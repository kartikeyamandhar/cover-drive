"""Tests for the deterministic scoreboard and the faithful fallback line (T6.2)."""

from __future__ import annotations

from typing import Any

from app.distill.filters import faithfulness_check
from app.serve.ball import ServeBall
from app.serve.scoreboard import fallback_line, scoreboard


def _ball(**over: Any) -> ServeBall:
    """A realistic per-ball record; override any field via kwargs (delivery/state merged)."""
    base: dict[str, Any] = {
        "match_id": "392232",
        "ball_id": "2-11-6",
        "innings": 1,
        "over": 11,
        "ball_in_over": 6,
        "legal_ball_number": 72,
        "is_legal": True,
        "batting_team": "Royal Challengers Bangalore",
        "bowling_team": "Rajasthan Royals",
        "striker": {"id": "s", "name": "V Kohli"},
        "non_striker": {"id": "n", "name": "AB de Villiers"},
        "bowler": {"id": "b", "name": "S Warne"},
        "delivery": {
            "runs_batter": 4,
            "runs_extras": 0,
            "runs_total": 4,
            "extras": {},
            "wickets": [],
        },
        "state": {
            "score_runs": 99,
            "score_wickets": 2,
            "balls_left": 48,
            "current_run_rate": 8.25,
            "required_run_rate": 9.1,
            "target_runs": 173,
            "runs_required": 74,
            "striker_runs": 41,
            "striker_balls": 28,
            "bowler_runs": 24,
            "bowler_balls": 18,
            "bowler_wickets": 1,
            "partnership_runs": 47,
            "partnership_balls": 26,
            "last_deliveries": ["1", "0", "4", "1", "2", "4"],
            "phase": "middle",
            "milestones": [],
        },
        "state_string": (
            "T20 Indian Premier League | Inns 2 | 11.6 | Royal Challengers Bangalore 99/2 | "
            "need 74 off 48 | CRR 8.25 RRR 9.1 | Striker V Kohli 41(28) | Bowler S Warne 1/24 | "
            "P'ship 47(26) | Last 1 0 4 1 2 4 | middle"
        ),
    }
    base["delivery"].update(over.pop("delivery", {}))
    base["state"].update(over.pop("state", {}))
    base.update(over)
    return ServeBall.model_validate(base)


def test_scoreboard_is_rendered_from_the_record() -> None:
    sb = scoreboard(_ball())
    assert sb.over == "11.6"  # over 0-based: legal_ball_number 72 - 6*11
    assert sb.innings == 2  # 0-based stored, 1-based displayed
    assert sb.score == "99/2"
    assert sb.bowler_figures == "1/24"
    assert sb.striker == "V Kohli"
    assert sb.event == "FOUR off the bat"
    assert sb.last_deliveries == ["1", "0", "4", "1", "2", "4"]


def test_fallback_line_is_faithful_for_every_event() -> None:
    cases = [
        _ball(),  # FOUR
        _ball(delivery={"runs_batter": 6, "runs_total": 6}),  # SIX
        _ball(delivery={"runs_batter": 0, "runs_total": 0}),  # dot
        _ball(delivery={"runs_batter": 2, "runs_total": 2}),  # 2 runs
        _ball(delivery={"runs_batter": 1, "runs_total": 1}),  # single
        _ball(delivery={"runs_batter": 0, "runs_total": 1, "extras": {"wides": 1}}),  # wide
        _ball(
            delivery={
                "runs_batter": 0,
                "runs_total": 0,
                "wickets": [{"player_out_name": "V Kohli", "kind": "caught"}],
            }
        ),  # WICKET
    ]
    for ball in cases:
        line = fallback_line(ball)
        result = faithfulness_check(line, ball.event, ball.state_string)
        assert result.ok, f"{ball.event!r}: {line!r} -> {result.reasons}"


def test_fallback_wicket_names_the_dismissed_batter() -> None:
    ball = _ball(
        delivery={
            "runs_batter": 0,
            "runs_total": 0,
            "wickets": [{"player_out_name": "AB de Villiers", "kind": "bowled"}],
        }
    )
    line = fallback_line(ball)
    assert "AB de Villiers" in line
    assert "bowled" in line
