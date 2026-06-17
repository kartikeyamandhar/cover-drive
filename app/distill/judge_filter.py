"""Second-stage faithfulness filter: an Opus judge audits the heuristic survivors.

The boundary hunt showed the Phase 2 heuristic catches the mechanical defects but
misses subtle ones (a reworded chase equation, an invented partnership figure). The
chosen full-run pipeline therefore adds a stronger reader AFTER the heuristic: every
pair in a HIGH-RISK bucket (where round-0 defects concentrate) is judged by Opus and
dropped if unfaithful. Low-risk buckets (routine middle/death overs), which audited
near-clean, are kept unjudged to save spend.

Sync mode (concurrent) is for validation-size passes; batch mode (Batches API, 50%
off) is for the full run. Dropped pairs are logged, never silently discarded.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import anthropic
import structlog
from anthropic.types.messages.batch_create_params import Request

from app.eval.judge import build_judge_params, judge_cost, judge_one, parse_judge_message
from configs.distill import PRICING, DistillConfig
from configs.personas import persona_by_key

log = structlog.get_logger(__name__)

# Buckets whose round-0 defect rate justified paying the judge (>= ~15%). The
# near-clean routine buckets are trusted to the heuristic alone.
HIGH_RISK_BUCKETS: frozenset[str] = frozenset(
    {"wicket", "tight_finish", "death_chase_pressure", "batter_milestone", "four", "six"}
)

_DROPPED_FILE = "judge_dropped.jsonl"

# Measured cached prefix (rubric + tools) for the Opus judge, from count_tokens.
# Reads at 0.1x once warm; the volatile per-pair turn and the forced-tool output.
# Output was MEASURED at ~230 tokens on a real 3,119-pair run (the forced verdict +
# explanation), not the 95 first guessed -- that 2.4x miss caused a budget overrun,
# so the estimate and the guard below are now keyed off the measured value.
MEASURED_JUDGE_PREFIX_TOKENS = 4830
_JUDGE_VOLATILE_TOKENS = 280
_JUDGE_OUTPUT_TOKENS = 230


def per_pair_cost(model: str = "claude-opus-4-8", *, batch: bool = True) -> float:
    """Marginal USD cost of judging ONE more pair (cache-warm), excluding the one-off
    cache write. This is what the budget guard divides by to stay solvent."""
    in_price, out_price = PRICING[model]
    mult = 0.5 if batch else 1.0
    read = MEASURED_JUDGE_PREFIX_TOKENS * 0.1 * in_price
    volatile = _JUDGE_VOLATILE_TOKENS * in_price
    output = _JUDGE_OUTPUT_TOKENS * out_price
    return (read + volatile + output) / 1e6 * mult


# Risk order for budget-capped runs: when the budget cannot cover every high-risk
# pair, judge the highest-defect strata first (round-1 defect rates), so each dollar
# removes the most defects. Buckets not listed sort last.
_BUCKET_PRIORITY: tuple[str, ...] = (
    "tight_finish",
    "six",
    "batter_milestone",
    "wicket",
    "four",
    "death_chase_pressure",
)


def _affordable_targets(
    targets: list[tuple[int, _Pair]], *, budget_usd: float, model: str, batch: bool
) -> tuple[list[tuple[int, _Pair]], int]:
    """Risk-rank the targets and keep only as many as the budget can judge.

    Returns (kept, n_skipped). Guards against a single large batch blowing the cap:
    the cap is enforced BEFORE submission by the count, not only between chunks.
    """
    marginal = per_pair_cost(model, batch=batch)
    write = MEASURED_JUDGE_PREFIX_TOKENS * 2.0 * PRICING[model][0] / 1e6
    affordable = int(max(0.0, budget_usd - write) / marginal) if marginal else len(targets)
    if affordable >= len(targets):
        return targets, 0
    ranked = sorted(
        targets,
        key=lambda t: _BUCKET_PRIORITY.index(t[1].data.get("bucket", ""))
        if t[1].data.get("bucket", "") in _BUCKET_PRIORITY
        else len(_BUCKET_PRIORITY),
    )
    return ranked[:affordable], len(targets) - affordable


@dataclass
class JudgeFilterStats:
    """Outcome of a judge-filter pass."""

    pairs: int = 0
    high_risk: int = 0
    judged: int = 0
    kept: int = 0
    dropped: int = 0
    judge_failed: int = 0
    skipped_budget: int = 0
    spend_usd: float = 0.0


@dataclass
class _Pair:
    """One distilled pair plus where it came from (for rewrite)."""

    file: Path
    data: dict[str, str]


def _pair_files(config: DistillConfig) -> list[Path]:
    return [
        p for p in sorted(config.distill_dir.glob("*.jsonl")) if not p.name.startswith("judge_")
    ]


def _load_pairs(config: DistillConfig) -> list[_Pair]:
    pairs: list[_Pair] = []
    for path in _pair_files(config):
        for raw in path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                pairs.append(_Pair(file=path, data=json.loads(raw)))
    return pairs


def is_high_risk(pair: dict[str, str], buckets: frozenset[str]) -> bool:
    return pair.get("bucket", "") in buckets


def _judge_pair(
    client: anthropic.Anthropic, pair: dict[str, str], model: str
) -> tuple[bool, float]:
    """Judge one pair; return (faithful, cost). A judge failure keeps the pair."""
    persona = persona_by_key(pair["persona"])
    try:
        res = judge_one(
            client,
            persona_key=persona.key,
            persona_instruction=persona.instruction,
            event=pair["event"],
            state=pair["state"],
            line=pair["commentary"],
            model=model,
        )
    except anthropic.APIError as exc:
        log.warning("judge-filter call failed", error=str(exc))
        return True, 0.0  # fail open: do not drop a pair we could not judge
    if res.verdict is None:
        return True, judge_cost(res, model)
    return res.verdict.faithful, judge_cost(res, model)


def _rewrite(
    config: DistillConfig, pairs: list[_Pair], drop_ids: set[int], dropped_log: list[dict[str, str]]
) -> None:
    """Rewrite each persona file keeping every pair except the dropped indices."""
    keep_by_file: dict[Path, list[str]] = {}
    for idx, pair in enumerate(pairs):
        if idx in drop_ids:
            continue
        keep_by_file.setdefault(pair.file, []).append(
            json.dumps(pair.data, sort_keys=True, ensure_ascii=False)
        )
    for path in _pair_files(config):
        lines = keep_by_file.get(path, [])
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if dropped_log:
        with (config.distill_dir / _DROPPED_FILE).open("a", encoding="utf-8") as handle:
            for rec in dropped_log:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def estimate_cost(n_high_risk: int, *, model: str = "claude-opus-4-8", batch: bool = True) -> float:
    """Project the judge-filter USD cost for ``n_high_risk`` pairs (cache-warm)."""
    in_price, out_price = PRICING[model]
    mult = 0.5 if batch else 1.0
    write = MEASURED_JUDGE_PREFIX_TOKENS * 2.0 * in_price / 1e6
    reads = n_high_risk * MEASURED_JUDGE_PREFIX_TOKENS * 0.1 * in_price / 1e6 * mult
    volatile = n_high_risk * _JUDGE_VOLATILE_TOKENS * in_price / 1e6 * mult
    output = n_high_risk * _JUDGE_OUTPUT_TOKENS * out_price / 1e6 * mult
    return round(write + reads + volatile + output, 2)


def count_high_risk_pairs(
    config: DistillConfig, buckets: frozenset[str] = HIGH_RISK_BUCKETS
) -> int:
    """Count the already-generated pairs that the judge-filter would audit."""
    return sum(1 for p in _load_pairs(config) if is_high_risk(p.data, buckets))


def judge_filter_sync(
    client: anthropic.Anthropic,
    config: DistillConfig,
    *,
    model: str = "claude-opus-4-8",
    buckets: frozenset[str] = HIGH_RISK_BUCKETS,
    max_workers: int = 6,
    budget_usd: float = 25.0,
) -> JudgeFilterStats:
    """Judge the high-risk survivors concurrently and drop the unfaithful ones."""
    pairs = _load_pairs(config)
    targets = [(i, p) for i, p in enumerate(pairs) if is_high_risk(p.data, buckets)]
    stats = JudgeFilterStats(pairs=len(pairs), high_risk=len(targets))
    targets, skipped = _affordable_targets(targets, budget_usd=budget_usd, model=model, batch=False)
    stats.skipped_budget = skipped
    if skipped:
        log.warning("judge-filter capped to budget (risk-ranked)", skipped=skipped)
    drop_ids: set[int] = set()
    dropped_log: list[dict[str, str]] = []

    def work(idx_pair: tuple[int, _Pair]) -> tuple[int, bool, float]:
        idx, pair = idx_pair
        faithful, cost = _judge_pair(client, pair.data, model)
        return idx, faithful, cost

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for start in range(0, len(targets), max_workers):
            if stats.spend_usd >= budget_usd:
                log.warning("judge-filter budget reached", spend=round(stats.spend_usd, 4))
                break
            for idx, faithful, cost in pool.map(work, targets[start : start + max_workers]):
                stats.judged += 1
                stats.spend_usd += cost
                if faithful:
                    stats.kept += 1
                else:
                    stats.dropped += 1
                    drop_ids.add(idx)
                    dropped_log.append(pairs[idx].data)

    _rewrite(config, pairs, drop_ids, dropped_log)
    log.info("judge-filter sync done", **vars(stats))
    return stats


def _warm_judge_cache(client: anthropic.Anthropic, model: str) -> None:
    """Write the 4830-token rubric prefix to cache once, so batch requests read it
    at 0.1x instead of each paying (and re-writing) the full prefix."""
    try:
        judge_one(
            client,
            persona_key="broadcast",
            persona_instruction="lead caller",
            event="dot ball",
            state="T20 | Inns 1 | 0.1 | A 0/0 | powerplay",
            line="No run.",
            model=model,
        )
    except anthropic.APIError as exc:
        log.warning("judge cache warm failed", error=str(exc))


def judge_filter_batch(
    client: anthropic.Anthropic,
    config: DistillConfig,
    *,
    model: str = "claude-opus-4-8",
    buckets: frozenset[str] = HIGH_RISK_BUCKETS,
    poll_seconds: int = 30,
    chunk_size: int = 500,
    budget_usd: float = 15.0,
) -> JudgeFilterStats:
    """Judge the high-risk survivors via the Batches API (50% off) and drop defects.

    The rubric cache is warmed once before submitting; the spend is checked BEFORE
    each chunk so a cache miss (which would make each call pay the full 4830-token
    prefix) can overshoot the cap by at most one chunk, never the whole run.
    """
    pairs = _load_pairs(config)
    targets = [(i, p) for i, p in enumerate(pairs) if is_high_risk(p.data, buckets)]
    stats = JudgeFilterStats(pairs=len(pairs), high_risk=len(targets))
    targets, skipped = _affordable_targets(targets, budget_usd=budget_usd, model=model, batch=True)
    stats.skipped_budget = skipped
    if skipped:
        log.warning("judge-filter capped to budget (risk-ranked)", skipped=skipped)
    drop_ids: set[int] = set()
    dropped_log: list[dict[str, str]] = []
    in_price, out_price = PRICING[model]
    if targets:
        _warm_judge_cache(client, model)

    for start in range(0, len(targets), chunk_size):
        if stats.spend_usd >= budget_usd:
            log.warning("judge-filter budget reached", spend=round(stats.spend_usd, 4))
            break
        chunk = targets[start : start + chunk_size]
        by_cid = {f"j{i}": (idx, pair) for i, (idx, pair) in enumerate(chunk)}
        requests = [
            Request(
                custom_id=cid,
                params=build_judge_params(
                    persona_key=pair.data["persona"],
                    persona_instruction=persona_by_key(pair.data["persona"]).instruction,
                    event=pair.data["event"],
                    state=pair.data["state"],
                    line=pair.data["commentary"],
                    model=model,
                ),
            )
            for cid, (_idx, pair) in by_cid.items()
        ]
        batch = client.messages.batches.create(requests=requests)
        log.info("judge batch submitted", batch_id=batch.id, n=len(requests))
        while client.messages.batches.retrieve(batch.id).processing_status != "ended":
            time.sleep(poll_seconds)
        for result in client.messages.batches.results(batch.id):
            idx, pair = by_cid[result.custom_id]
            stats.judged += 1
            if result.result.type != "succeeded":
                stats.judge_failed += 1
                stats.kept += 1  # fail open
                continue
            verdict, judge_result = parse_judge_message(result.result.message)
            stats.spend_usd += (
                (
                    judge_result.input_tokens * in_price
                    + judge_result.cache_read_tokens * in_price * 0.1
                    + judge_result.output_tokens * out_price
                )
                / 1e6
                * 0.5
            )  # batch discount
            if verdict is not None and not verdict.faithful:
                stats.dropped += 1
                drop_ids.add(idx)
                dropped_log.append(pair.data)
            else:
                stats.kept += 1

    _rewrite(config, pairs, drop_ids, dropped_log)
    log.info("judge-filter batch done", **vars(stats))
    return stats
