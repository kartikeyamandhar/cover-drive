"""Assemble Phase 2 distill pairs into split train/val/test SFT JSONL + stats.

Pure data shaping: read the distill pairs, format each into an ``SFTExample``,
route it to its split by ``match_id`` (a match never spans two splits), and write
deterministic per-split JSONL plus a stats summary. No model, no network, no spend.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterator
from dataclasses import asdict, dataclass

import structlog

from app.dataset.format import SFTExample, format_pair
from app.dataset.split import Split, split_for_match
from configs.distill import DistillConfig

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class AssemblyStats:
    """Summary of an assembly run."""

    total: int
    by_split: dict[str, int]
    by_persona: dict[str, int]
    by_bucket: dict[str, int]
    assistant_words_mean: float
    assistant_words_p95: int


def read_pairs(config: DistillConfig) -> Iterator[dict[str, str]]:
    """Yield every distill pair from ``data/distill/*.jsonl``.

    Skips bookkeeping files written into the same directory (e.g. the judge-filter's
    ``judge_dropped.jsonl`` log): those hold the pairs that were DELETED, and reading
    them back would silently re-admit the very defects the filter removed.
    """
    for path in sorted(config.distill_dir.glob("*.jsonl")):
        if path.name.startswith("judge_"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield json.loads(line)


def split_of(config: DistillConfig, match_id: str) -> str:
    """The split a match belongs to (pure function of the id)."""
    return split_for_match(
        match_id, val_fraction=config.val_fraction, test_fraction=config.test_fraction
    ).value


def _assistant_words(example: SFTExample) -> int:
    content = next(turn.content for turn in example.messages if turn.role == "assistant")
    return len(content.split())


def _compute_stats(
    examples: list[SFTExample], by_split: dict[str, list[SFTExample]]
) -> AssemblyStats:
    counts_split = {split: len(items) for split, items in by_split.items()}
    counts_persona: dict[str, int] = {}
    counts_bucket: dict[str, int] = {}
    words: list[int] = []
    for example in examples:
        counts_persona[example.persona] = counts_persona.get(example.persona, 0) + 1
        counts_bucket[example.bucket] = counts_bucket.get(example.bucket, 0) + 1
        words.append(_assistant_words(example))
    if not words:
        return AssemblyStats(0, counts_split, counts_persona, counts_bucket, 0.0, 0)
    ordered = sorted(words)
    p95 = ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)]
    return AssemblyStats(
        total=len(examples),
        by_split=counts_split,
        by_persona=counts_persona,
        by_bucket=counts_bucket,
        assistant_words_mean=round(statistics.mean(words), 2),
        assistant_words_p95=p95,
    )


def assemble(config: DistillConfig) -> AssemblyStats:
    """Build train/val/test SFT JSONL + stats from the distill pairs."""
    by_split: dict[str, list[SFTExample]] = {split.value: [] for split in Split}
    for pair in read_pairs(config):
        split = split_of(config, pair["match_id"])
        by_split[split].append(format_pair(pair, split))

    config.dataset_dir.mkdir(parents=True, exist_ok=True)
    all_examples: list[SFTExample] = []
    for split, examples in by_split.items():
        examples.sort(key=lambda e: (e.match_id, e.ball_id, e.persona))
        body = "\n".join(e.model_dump_json() for e in examples)
        (config.dataset_dir / f"{split}.jsonl").write_text(
            body + "\n" if body else "", encoding="utf-8"
        )
        all_examples.extend(examples)

    stats = _compute_stats(all_examples, by_split)
    (config.dataset_dir / "stats.json").write_text(
        json.dumps(asdict(stats), indent=2, sort_keys=True), encoding="utf-8"
    )
    log.info("assembled", **stats.by_split, total=stats.total)
    return stats
