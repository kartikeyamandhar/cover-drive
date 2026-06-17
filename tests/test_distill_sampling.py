"""Tests for bucket assignment and stratified sampling."""

from __future__ import annotations

from pathlib import Path

from app.distill.records import BallRecord, ball_event
from app.distill.sampling import bucket_of, secondary_subset, stratified_sample
from app.features.build import build_match
from configs.data import FeatureConfig
from configs.distill import DistillConfig

FIXTURE = Path(__file__).parent / "fixtures" / "sample_match.json"


def _rec(
    *,
    innings: int = 0,
    runs_batter: int = 0,
    runs_total: int = 0,
    wickets: list[dict[str, object]] | None = None,
    phase: str = "middle",
    rrr: float | None = None,
    runs_required: int | None = None,
    balls_left: int = 120,
    milestones: list[str] | None = None,
    hat_trick: bool = False,
    striker_runs: int = 0,
    is_legal: bool = True,
    extras: dict[str, int] | None = None,
) -> BallRecord:
    return BallRecord.model_validate(
        {
            "match_id": "m",
            "ball_id": "0-0-1",
            "innings": innings,
            "is_legal": is_legal,
            "striker": {"id": "s1", "name": "X"},
            "bowler": {"id": "b1", "name": "Y"},
            "delivery": {
                "runs_batter": runs_batter,
                "runs_extras": 0,
                "runs_total": runs_total,
                "extras": extras or {},
                "wickets": wickets or [],
            },
            "state": {
                "phase": phase,
                "required_run_rate": rrr,
                "runs_required": runs_required,
                "balls_left": balls_left,
                "milestones": milestones or [],
                "bowler_on_hat_trick": hat_trick,
                "striker_runs": striker_runs,
            },
            "state_string": "STATE",
        }
    )


def test_bucket_priority() -> None:
    assert bucket_of(_rec(wickets=[{"kind": "bowled"}], runs_batter=0)) == "wicket"
    assert bucket_of(_rec(hat_trick=True)) == "hat_trick"
    assert bucket_of(_rec(runs_batter=4)) == "four"
    assert bucket_of(_rec(innings=1, phase="death", rrr=13.0)) == "death_chase_pressure"
    assert bucket_of(_rec(runs_batter=6)) == "six"
    assert bucket_of(_rec(milestones=["batter_nearing_50"])) == "batter_milestone"
    assert bucket_of(_rec(striker_runs=50, runs_batter=2)) == "batter_milestone"  # crossing 50
    assert bucket_of(_rec(phase="death", runs_total=0)) == "dot_under_pressure"
    assert bucket_of(_rec(phase="powerplay", runs_batter=1)) == "powerplay_routine"
    assert bucket_of(_rec(innings=1, runs_required=8, balls_left=6)) == "tight_finish"
    assert bucket_of(_rec(milestones=["team_nearing_200"])) == "team_milestone"
    assert bucket_of(_rec(phase="middle", runs_batter=2)) == "middle_routine"
    assert bucket_of(_rec(phase="death", innings=0, runs_batter=2, runs_total=2)) == "death_routine"


def test_retired_hurt_is_not_a_wicket_bucket() -> None:
    assert bucket_of(_rec(wickets=[{"kind": "retired hurt"}], runs_batter=2)) == "middle_routine"


def test_unbucketed_returns_none() -> None:
    assert bucket_of(_rec(phase="middle", runs_batter=0, runs_total=0)) is None


def test_ball_event_phrases() -> None:
    assert ball_event(_rec(runs_batter=6, runs_total=6)) == "SIX off the bat"
    assert ball_event(_rec(runs_batter=4, runs_total=4)) == "FOUR off the bat"
    assert ball_event(_rec(wickets=[{"kind": "caught"}])) == "WICKET, caught"
    assert ball_event(_rec(runs_total=0)) == "dot ball"
    assert ball_event(_rec(runs_total=1)) == "1 run"
    assert ball_event(_rec(runs_total=2)) == "2 runs"
    assert ball_event(_rec(runs_total=1, extras={"wides": 1})) == "wide"


def test_stratified_sample_and_subset(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    build_match(FIXTURE, processed, FeatureConfig())
    config = DistillConfig(
        data_dir=tmp_path / "data", val_fraction=0.0, test_fraction=0.0, primary_set_size=100
    )
    primary = stratified_sample(config)
    assert len(primary) > 0
    assert all(c.match_id == "sample_match" for c in primary)
    subset = secondary_subset(primary, fraction=0.5, seed=1)
    assert 0 < len(subset) <= len(primary)


def test_stratified_sample_respects_split(tmp_path: Path) -> None:
    from app.dataset.split import Split

    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    build_match(FIXTURE, processed, FeatureConfig())
    # val_fraction=test_fraction=0 -> the only match is TRAIN, so VAL/TEST draws are empty
    config = DistillConfig(
        data_dir=tmp_path / "data", val_fraction=0.0, test_fraction=0.0, primary_set_size=100
    )
    assert len(stratified_sample(config, split=Split.TRAIN)) > 0
    assert stratified_sample(config, split=Split.VAL, set_size=50) == []
