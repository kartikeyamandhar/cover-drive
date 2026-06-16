"""Featurization: every cricket rule from the Phase 1 red-team, hand-checked.

Expected values are computed by hand against tests/fixtures/sample_match.json.
"""

from __future__ import annotations

from pathlib import Path

from app.cricsheet.models import (
    Delivery,
    Extras,
    Innings,
    Match,
    MatchInfo,
    Over,
    Powerplay,
    Registry,
    Runs,
    Wicket,
)
from app.cricsheet.parse import parse_match
from app.features.serialize import ball_id
from app.features.state import (
    BallContext,
    compute_milestones,
    iter_ball_contexts,
    phase_of,
    powerplay_over_count,
)
from configs.data import FeatureConfig


def _match_with(deliveries: list[Delivery], powerplays: list[Powerplay] | None = None) -> Match:
    """Build a minimal one-innings, one-over match for edge-case tests."""
    info = MatchInfo(teams=["A", "B"], overs=20, balls_per_over=6, registry=Registry(people={}))
    innings = Innings(
        team="A",
        overs=[Over(over=0, deliveries=deliveries)],
        powerplays=powerplays or [],
    )
    return Match(info=info, innings=[innings])


def _delivery(
    *,
    total: int,
    batter_runs: int = 0,
    extras: Extras | None = None,
    wickets: list[Wicket] | None = None,
) -> Delivery:
    return Delivery(
        batter="X",
        bowler="Z",
        non_striker="Y",
        runs=Runs(batter=batter_runs, extras=total - batter_runs, total=total),
        extras=extras,
        wickets=wickets or [],
    )


FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample_match.json"


def _contexts() -> dict[str, BallContext]:
    match = parse_match(SAMPLE)
    return {ball_id(c): c for c in iter_ball_contexts(match, "sample_match", FeatureConfig())}


def test_total_balls_emitted() -> None:
    # innings 1 has 8 deliveries (incl. 1 wide + 1 no-ball), innings 2 has 3.
    assert len(_contexts()) == 11


def test_boundary_ball() -> None:
    ball = _contexts()["0-0-1"]
    assert ball.is_legal
    assert ball.runs_total == 4
    assert ball.score_runs == 4
    assert ball.legal_ball_number == 1
    assert ball.current_run_rate == 24.0
    assert ball.striker_id == "aaaa0001"
    assert ball.striker_runs == 4
    assert ball.striker_balls == 1


def test_wide_does_not_advance_legal_ball() -> None:
    ball = _contexts()["0-0-3"]
    assert ball.is_legal is False
    assert ball.legal_ball_number == 2  # unchanged from the previous legal ball
    assert ball.extras == {"wides": 1}
    assert ball.score_runs == 6
    assert ball.bowler_runs == 6  # the wide is charged to the bowler


def test_no_ball_credits_batter_runs_but_not_balls_faced() -> None:
    ball = _contexts()["0-0-5"]
    assert ball.is_legal is False
    assert ball.extras == {"noballs": 1}
    assert ball.striker_runs == 1  # 1 run off the bat on the no-ball
    assert ball.striker_balls == 1  # but only the one earlier legal ball is "faced"
    assert ball.bowler_runs == 8  # +2 conceded (batter 1 + no-ball 1)


def test_bowled_wicket_credited_and_partnership_resets() -> None:
    ctx = _contexts()
    out = ctx["0-0-6"]
    assert out.score_wickets == 1
    assert len(out.wickets) == 1
    assert out.wickets[0].kind == "bowled"
    assert out.wickets[0].credited_to_bowler is True
    assert out.wickets[0].player_out_id == "aaaa0002"
    assert out.bowler_wickets == 1
    assert out.partnership_runs == 8  # the stand that just ended
    assert out.partnership_balls == 4

    nxt = ctx["0-0-7"]  # fresh partnership starts
    assert nxt.partnership_runs == 6
    assert nxt.partnership_balls == 1
    assert nxt.score_runs == 14


def test_last_deliveries_trail_spans_over() -> None:
    ball = _contexts()["0-0-7"]
    assert ball.runs_total == 6
    assert ball.last_deliveries == ("1", "1", "0", "2", "W", "6")


def test_innings_one_totals() -> None:
    ball = _contexts()["0-0-8"]
    assert ball.score_runs == 15
    assert ball.score_wickets == 1
    assert ball.bowler_wickets == 1
    assert ball.bowler_runs == 15
    assert ball.bowler_balls == 6
    assert ball.balls_left == 0
    assert ball.current_run_rate == 15.0


def test_second_innings_target_and_required_rate() -> None:
    ball = _contexts()["1-0-1"]
    assert ball.target_runs == 16
    assert ball.runs_required == 15
    assert ball.balls_left == 5
    assert ball.required_run_rate == 18.0
    assert ball.current_run_rate == 6.0
    assert ball.batting_team == "Team Beta"
    assert ball.bowling_team == "Team Alpha"


def test_run_out_not_credited_to_bowler() -> None:
    ball = _contexts()["1-0-3"]
    assert ball.score_wickets == 1
    assert ball.wickets[0].kind == "run out"
    assert ball.wickets[0].credited_to_bowler is False
    assert ball.bowler_wickets == 0
    assert ball.runs_required == 11
    assert ball.required_run_rate == 22.0


def test_first_innings_has_no_required_rate() -> None:
    ball = _contexts()["0-0-1"]
    assert ball.required_run_rate is None
    assert ball.target_runs is None
    assert ball.runs_required is None


def test_phase_of_boundaries() -> None:
    assert phase_of(0, 20, 6, 5) == "powerplay"
    assert phase_of(5, 20, 6, 5) == "powerplay"
    assert phase_of(6, 20, 6, 5) == "middle"
    assert phase_of(14, 20, 6, 5) == "middle"
    assert phase_of(15, 20, 6, 5) == "death"
    assert phase_of(19, 20, 6, 5) == "death"


def test_powerplay_over_count_from_source() -> None:
    features = FeatureConfig()
    # no powerplays in source -> configured default
    assert powerplay_over_count(Innings(team="A"), features) == 6
    # a reduced (DLS) powerplay span 0.1..4.6 -> 5 overs
    reduced = Innings(
        team="A",
        powerplays=[Powerplay.model_validate({"from": 0.1, "to": 4.6, "type": "mandatory"})],
    )
    assert powerplay_over_count(reduced, features) == 5


def test_retired_hurt_is_not_a_fall_of_wicket() -> None:
    deliveries = [
        _delivery(total=0, wickets=[Wicket(player_out="X", kind="retired hurt")]),
        _delivery(total=0, wickets=[Wicket(player_out="W", kind="bowled")]),
    ]
    ctx = list(iter_ball_contexts(_match_with(deliveries), "m", FeatureConfig()))
    assert ctx[0].score_wickets == 0  # retired hurt: not out
    assert ctx[0].wickets[0].kind == "retired hurt"
    assert ctx[0].bowler_wickets == 0
    assert ctx[1].score_wickets == 1  # the bowled dismissal counts
    assert ctx[1].bowler_wickets == 1


def test_multiple_wickets_on_one_delivery() -> None:
    deliveries = [
        _delivery(
            total=0,
            wickets=[
                Wicket(player_out="X", kind="run out"),
                Wicket(player_out="Y", kind="run out"),
            ],
        )
    ]
    ctx = next(iter_ball_contexts(_match_with(deliveries), "m", FeatureConfig()))
    assert ctx.score_wickets == 2
    assert ctx.bowler_wickets == 0


def test_run_out_off_a_no_ball() -> None:
    deliveries = [
        _delivery(
            total=1,
            batter_runs=0,
            extras=Extras(noballs=1),
            wickets=[Wicket(player_out="Y", kind="run out")],
        )
    ]
    ctx = next(iter_ball_contexts(_match_with(deliveries), "m", FeatureConfig()))
    assert ctx.is_legal is False  # a no-ball
    assert ctx.legal_ball_number == 0  # does not advance the legal-ball count
    assert ctx.score_wickets == 1  # the run out is still a dismissal
    assert ctx.bowler_wickets == 0


def test_milestones() -> None:
    features = FeatureConfig()
    assert "batter_nearing_50" in compute_milestones(45, 10, on_hat_trick=False, features=features)
    assert "batter_nearing_100" in compute_milestones(92, 10, on_hat_trick=False, features=features)
    assert "team_nearing_200" in compute_milestones(10, 192, on_hat_trick=False, features=features)
    assert "bowler_on_hat_trick" in compute_milestones(0, 10, on_hat_trick=True, features=features)
    assert compute_milestones(10, 10, on_hat_trick=False, features=features) == ()
