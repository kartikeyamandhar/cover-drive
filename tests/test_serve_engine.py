"""Tests for the commentary engine: prompt parity + validate-or-fallback (T6.3)."""

from __future__ import annotations

from typing import Any

from app.serve.ball import ServeBall
from app.serve.engine import build_prompt, generate_commentary
from app.serve.runtime import StubRuntime
from configs.personas import persona_by_key

BROADCAST = persona_by_key("broadcast")


def _ball(**delivery: Any) -> ServeBall:
    d = {"runs_batter": 4, "runs_extras": 0, "runs_total": 4, "extras": {}, "wickets": []}
    d.update(delivery)
    return ServeBall.model_validate(
        {
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
            "delivery": d,
            "state": {"score_runs": 99, "score_wickets": 2, "balls_left": 48, "phase": "middle"},
            "state_string": (
                "T20 IPL | Inns 2 | 11.6 | Royal Challengers Bangalore 99/2 | "
                "CRR 8.25 | Striker V Kohli 41(28) | Bowler S Warne 1/24 | middle"
            ),
        }
    )


def test_build_prompt_matches_training_format() -> None:
    system, user = build_prompt(_ball(), BROADCAST)
    assert BROADCAST.display_name in system
    assert user == "BALL: FOUR off the bat\nSTATE: " + _ball().state_string


def test_faithful_model_line_passes_through() -> None:
    runtime = StubRuntime("Kohli pierces the field for four, RCB race to 99/2.")
    result = generate_commentary(runtime, _ball(), BROADCAST, retries=2)
    assert result.source == "model"
    assert result.faithful
    assert result.attempts == 1


def test_unfaithful_model_line_falls_back() -> None:
    # event is a FOUR; the model calls it a six -> unfaithful -> deterministic fallback
    runtime = StubRuntime("Kohli launches a colossal six into the stands!")
    result = generate_commentary(runtime, _ball(), BROADCAST, retries=2)
    assert result.source == "fallback"
    assert result.faithful  # the fallback is always faithful
    assert result.attempts == 3  # 1 initial + 2 retries, all failed
    assert result.reasons  # records why the model draws were rejected
    assert "four" in result.line.lower()  # the fallback describes the real four


def test_empty_model_output_falls_back() -> None:
    result = generate_commentary(StubRuntime(""), _ball(), BROADCAST, retries=1)
    assert result.source == "fallback"
    assert result.attempts == 2


def test_retries_are_bounded() -> None:
    runtime = StubRuntime("Bowled him! Kohli is gone!")  # phantom wicket on a FOUR
    result = generate_commentary(runtime, _ball(), BROADCAST, retries=0)
    assert result.source == "fallback"
    assert result.attempts == 1  # retries=0 => exactly one model draw, then fallback
