"""Generation orchestration: plan work, generate, filter, write, with a budget cap.

Library code with IO at the edges; the Anthropic client is injected so tests mock
it. Idempotent and resumable via the manifest; a hard budget guard stops the run
before it exceeds the configured cap. Sync mode is the simple, cache-friendly path
(good for the pilot); batch mode uses the Batches API for the 50%-off full run.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import anthropic
import structlog
from anthropic.types.messages.batch_create_params import Request

from app.distill.errors import TeacherError
from app.distill.filters import faithfulness_check, seed_overlap
from app.distill.manifest import DistillManifest, item_key
from app.distill.sampling import SampleCandidate, secondary_subset, stratified_sample
from app.distill.seed import EXEMPLARS
from app.distill.teacher import (
    Generation,
    build_message_params,
    estimate_cost,
    generate_one,
    generation_cost,
    warm_cache,
)
from configs.distill import DistillConfig
from configs.personas import Persona, primary_persona, secondary_personas

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class WorkItem:
    """One (ball, persona) generation target."""

    candidate: SampleCandidate
    persona: Persona

    @property
    def key(self) -> str:
        return item_key(self.candidate.match_id, self.candidate.ball_id, self.persona.key)


@dataclass
class RunStats:
    """Accumulated outcome of a run."""

    processed: int = 0
    kept: int = 0
    rejected_faithfulness: int = 0
    rejected_seed_dedup: int = 0
    failed: int = 0
    spend_usd: float = 0.0
    cache_read_tokens: int = 0


def seed_lines() -> list[str]:
    """Every hand-authored seed line (for the seed-dedup filter)."""
    return [line for exemplar in EXEMPLARS for line in exemplar.lines.values()]


def plan_items(config: DistillConfig) -> list[WorkItem]:
    """The full work list: the primary persona over the stratified set, plus each
    secondary persona over a stratified subset of the SAME balls (per-ball A/B)."""
    primary = stratified_sample(config)
    items = [WorkItem(candidate, primary_persona()) for candidate in primary]
    subset = secondary_subset(
        primary, fraction=config.secondary_subset_fraction, seed=config.sampling_seed + 1
    )
    for persona in secondary_personas():
        items.extend(WorkItem(candidate, persona) for candidate in subset)
    return items


def _write_pair(config: DistillConfig, item: WorkItem, commentary: str) -> None:
    path = config.distill_dir / f"{item.persona.key}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "match_id": item.candidate.match_id,
        "ball_id": item.candidate.ball_id,
        "persona": item.persona.key,
        "bucket": item.candidate.bucket,
        "event": item.candidate.event,
        "state": item.candidate.state,
        "commentary": commentary,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")


def _record_result(
    config: DistillConfig,
    manifest: DistillManifest,
    item: WorkItem,
    gen: Generation,
    stats: RunStats,
    seeds: list[str],
) -> None:
    """Filter one generation and record it (faithfulness, then seed-dedup)."""
    faith = faithfulness_check(gen.text, item.candidate.event, item.candidate.state)
    if not faith.ok:
        stats.rejected_faithfulness += 1
        manifest.mark(item.key, "rejected_faithfulness")
        return
    if seed_overlap(gen.text, seeds):
        stats.rejected_seed_dedup += 1
        manifest.mark(item.key, "rejected_seed_dedup")
        return
    _write_pair(config, item, gen.text)
    stats.kept += 1
    manifest.mark(item.key, "kept")


def run_sync(
    client: anthropic.Anthropic,
    config: DistillConfig,
    items: list[WorkItem],
    *,
    budget_cap: float | None = None,
) -> RunStats:
    """Generate synchronously with a hard budget cap. Idempotent and resumable."""
    cap = budget_cap if budget_cap is not None else config.budget_usd_cap
    manifest = DistillManifest.load(config.manifest_path)
    seeds = seed_lines()
    stats = RunStats()
    for index, item in enumerate(items):
        if manifest.is_done(item.key):
            continue
        if stats.spend_usd >= cap:
            log.warning("budget cap reached", spend=round(stats.spend_usd, 4), cap=cap)
            break
        try:
            gen = generate_one(
                client, item.candidate.event, item.candidate.state, item.persona, config
            )
        except TeacherError as exc:
            stats.failed += 1
            manifest.mark(item.key, "failed")
            log.warning("generation failed", key=item.key, error=str(exc))
            continue
        stats.processed += 1
        stats.spend_usd += generation_cost(gen, config, batch=False)
        stats.cache_read_tokens += gen.cache_read_tokens
        _record_result(config, manifest, item, gen, stats, seeds)
        if (index + 1) % 50 == 0:
            manifest.save(config.manifest_path)
    manifest.save(config.manifest_path)
    log.info(
        "sync run complete",
        processed=stats.processed,
        kept=stats.kept,
        rejected_faithfulness=stats.rejected_faithfulness,
        rejected_seed_dedup=stats.rejected_seed_dedup,
        failed=stats.failed,
        spend_usd=round(stats.spend_usd, 4),
    )
    return stats


def _collect_batch_chunk(
    client: anthropic.Anthropic,
    config: DistillConfig,
    manifest: DistillManifest,
    chunk: list[WorkItem],
    stats: RunStats,
    seeds: list[str],
    *,
    poll_seconds: int,
) -> None:
    """Submit one chunk as a batch, wait, and collect/filter/record its results."""
    by_custom_id = {f"i{i}": item for i, item in enumerate(chunk)}
    requests = [
        Request(
            custom_id=custom_id,
            params=build_message_params(
                item.candidate.event, item.candidate.state, item.persona, config
            ),
        )
        for custom_id, item in by_custom_id.items()
    ]
    batch = client.messages.batches.create(requests=requests)
    log.info("batch chunk submitted", batch_id=batch.id, n=len(requests))
    while client.messages.batches.retrieve(batch.id).processing_status != "ended":
        time.sleep(poll_seconds)
    for result in client.messages.batches.results(batch.id):
        item = by_custom_id[result.custom_id]
        if result.result.type != "succeeded":
            stats.failed += 1
            manifest.mark(item.key, "failed")
            continue
        message = result.result.message
        usage = message.usage
        gen = Generation(
            text=next((b.text for b in message.content if b.type == "text"), "").strip(),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
        )
        stats.processed += 1
        stats.spend_usd += generation_cost(gen, config, batch=True)
        stats.cache_read_tokens += gen.cache_read_tokens
        _record_result(config, manifest, item, gen, stats, seeds)


def run_batch(
    client: anthropic.Anthropic,
    config: DistillConfig,
    items: list[WorkItem],
    *,
    budget_cap: float | None = None,
    poll_seconds: int = 30,
    chunk_size: int = 500,
) -> RunStats:
    """Generate via the Batches API (50% off), in chunks with a budget guard.

    The cache is warmed once; then chunks are submitted one at a time and the
    spend is checked BEFORE each chunk, so a cache miss (which makes a batch far
    more expensive than estimated) can overshoot the cap by at most one chunk.
    """
    cap = budget_cap if budget_cap is not None else config.budget_usd_cap
    manifest = DistillManifest.load(config.manifest_path)
    seeds = seed_lines()
    stats = RunStats()
    pending = [item for item in items if not manifest.is_done(item.key)]
    if not pending:
        return stats

    warm_cache(client, config)
    for start in range(0, len(pending), chunk_size):
        if stats.spend_usd >= cap:
            log.warning("budget cap reached", spend=round(stats.spend_usd, 4), cap=cap)
            break
        chunk = pending[start : start + chunk_size]
        _collect_batch_chunk(
            client, config, manifest, chunk, stats, seeds, poll_seconds=poll_seconds
        )
        manifest.save(config.manifest_path)
    manifest.save(config.manifest_path)
    log.info("batch run complete", kept=stats.kept, spend_usd=round(stats.spend_usd, 4))
    return stats


def dry_run(config: DistillConfig) -> tuple[int, float, float]:
    """Plan the run and return (n_items, batch_usd, sync_usd) without any API call.

    Uses the measured seed size and the measured per-call volatile/output sizes, so
    the estimate matches live cost. Batch is the recommended full-run mode.
    """
    from configs.distill import MEASURED_SEED_TOKENS

    n_items = len(plan_items(config))
    batch = estimate_cost(
        n_items,
        MEASURED_SEED_TOKENS,
        config=config,
        avg_volatile_input=190,
        avg_output=65,
        assume_batch=True,
    )
    sync = estimate_cost(
        n_items,
        MEASURED_SEED_TOKENS,
        config=config,
        avg_volatile_input=190,
        avg_output=65,
        assume_batch=False,
    )
    return n_items, batch.est_usd, sync.est_usd
