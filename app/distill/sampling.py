"""Situation-stratified sampling over the Phase 1 dataset.

Uniform sampling would drown in dot balls and routine singles; we instead
over-represent the rare, high-information situations a commentator reacts to.
Each ball is assigned to the FIRST bucket it matches (rare high-value buckets
first), so the buckets form a real partition. Sampling is deterministic (seeded)
and draws ONLY from training-split matches, so held-out matches stay clean for
Phase 5 evaluation.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import structlog

from app.dataset.split import Split, split_for_match
from app.distill.records import BallRecord, ball_event, has_real_wicket, parse_record
from configs.distill import DistillConfig

log = structlog.get_logger(__name__)

_BATTER_MARKS = (50, 100, 150)


@dataclass(frozen=True)
class SampleCandidate:
    """A ball selected for generation: enough to build the prompt and key the pair."""

    match_id: str
    ball_id: str
    bucket: str
    event: str
    state: str


def _crosses_batter_milestone(striker_runs: int, runs_batter: int) -> bool:
    before = striker_runs - runs_batter
    return any(before < mark <= striker_runs for mark in _BATTER_MARKS)


def bucket_of(record: BallRecord) -> str | None:
    """Assign a ball to its first-match situation bucket, or ``None`` if unbucketed.

    The order here MUST match ``configs.distill.STRATIFICATION``.
    """
    delivery = record.delivery
    state = record.state
    is_chase = record.innings >= 1
    no_wicket = not has_real_wicket(record)
    rrr = state.required_run_rate

    if not no_wicket:
        return "wicket"
    if state.bowler_on_hat_trick:
        return "hat_trick"
    if delivery.runs_batter == 4:
        return "four"
    if state.phase == "death" and is_chase and rrr is not None and rrr >= 12:
        return "death_chase_pressure"
    if delivery.runs_batter == 6:
        return "six"
    if any(m.startswith("batter_nearing") for m in state.milestones) or _crosses_batter_milestone(
        state.striker_runs, delivery.runs_batter
    ):
        return "batter_milestone"
    if (
        record.is_legal
        and delivery.runs_total == 0
        and (state.phase == "death" or (is_chase and rrr is not None and rrr >= 9))
    ):
        return "dot_under_pressure"
    if state.phase == "powerplay" and delivery.runs_batter in (0, 1, 2, 3):
        return "powerplay_routine"
    if (
        is_chase
        and state.runs_required is not None
        and 0 < state.runs_required <= 12
        and 0 < state.balls_left <= 12
    ):
        return "tight_finish"
    if any(m.startswith("team_nearing") for m in state.milestones):
        return "team_milestone"
    if state.phase == "middle" and delivery.runs_batter in (1, 2, 3):
        return "middle_routine"
    if state.phase == "death":
        return "death_routine"
    return None


def _scan_candidates(config: DistillConfig, split: Split) -> dict[str, list[SampleCandidate]]:
    """Bucket every ball of the given split in the processed dataset."""
    by_bucket: dict[str, list[SampleCandidate]] = defaultdict(list)
    files = sorted(config.processed_dir.glob("*.jsonl"))
    for path in files:
        match_id = path.stem
        if (
            split_for_match(
                match_id, val_fraction=config.val_fraction, test_fraction=config.test_fraction
            )
            is not split
        ):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = parse_record(line)
            bucket = bucket_of(record)
            if bucket is None:
                continue
            by_bucket[bucket].append(
                SampleCandidate(
                    match_id=record.match_id,
                    ball_id=record.ball_id,
                    bucket=bucket,
                    event=ball_event(record),
                    state=record.state_string,
                )
            )
    return by_bucket


def stratified_sample(
    config: DistillConfig, *, split: Split = Split.TRAIN, set_size: int | None = None
) -> list[SampleCandidate]:
    """Select a stratified set from the given split (train by default).

    ``set_size`` defaults to the configured primary size; pass a small value to
    draw a held-out val/test set at the boundary.
    """
    by_bucket = _scan_candidates(config, split)
    size = set_size if set_size is not None else config.primary_set_size
    rng = random.Random(config.sampling_seed)
    selected: list[SampleCandidate] = []
    for bucket in config.stratification():
        target = round(bucket.target_fraction * size)
        pool = sorted(by_bucket.get(bucket.name, []), key=lambda c: (c.match_id, c.ball_id))
        take = min(target, len(pool))
        if take < target:
            log.warning(
                "bucket undersupplied", bucket=bucket.name, target=target, available=len(pool)
            )
        selected.extend(rng.sample(pool, take) if take < len(pool) else pool)
    log.info("stratified sample", total=len(selected), buckets=len(by_bucket))
    return selected


def secondary_subset(
    primary: list[SampleCandidate], *, fraction: float, seed: int
) -> list[SampleCandidate]:
    """A stratified subset of the SAME primary balls (for the per-ball persona A/B)."""
    by_bucket: dict[str, list[SampleCandidate]] = defaultdict(list)
    for candidate in primary:
        by_bucket[candidate.bucket].append(candidate)
    rng = random.Random(seed)
    subset: list[SampleCandidate] = []
    for _bucket, pool in sorted(by_bucket.items()):
        ordered = sorted(pool, key=lambda c: (c.match_id, c.ball_id))
        take = round(fraction * len(ordered))
        subset.extend(rng.sample(ordered, take) if take < len(ordered) else ordered)
    return subset
