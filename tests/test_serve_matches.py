"""Tests for the bundled-match repository (T6.4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.serve.matches import MatchRepository, UnknownMatchError


def _rec(match_id: str, innings: int, ball_id: str, **state: Any) -> str:
    base_state = {"score_runs": 10, "score_wickets": 0, "balls_left": 100, "phase": "powerplay"}
    base_state.update(state)
    return json.dumps(
        {
            "match_id": match_id,
            "ball_id": ball_id,
            "innings": innings,
            "over": 0,
            "ball_in_over": 1,
            "legal_ball_number": 1,
            "is_legal": True,
            "batting_team": "Chennai Super Kings",
            "bowling_team": "Mumbai Indians",
            "striker": {"id": "s", "name": "MS Dhoni"},
            "non_striker": {"id": "n", "name": "S Raina"},
            "bowler": {"id": "b", "name": "L Malinga"},
            "delivery": {
                "runs_batter": 0,
                "runs_extras": 0,
                "runs_total": 0,
                "extras": {},
                "wickets": [],
            },
            "state": base_state,
            "state_string": "STATE",
        }
    )


def _write_match(dir_: Path, match_id: str, balls: int = 3, innings: int = 1) -> None:
    lines = [_rec(match_id, innings, f"{innings}-0-{i + 1}") for i in range(balls)]
    (dir_ / f"{match_id}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_list_and_load(tmp_path: Path) -> None:
    _write_match(tmp_path, "1001", balls=3)
    _write_match(tmp_path, "1002", balls=5)
    repo = MatchRepository(tmp_path)

    summaries = {s.match_id: s for s in repo.list_matches()}
    assert set(summaries) == {"1001", "1002"}
    assert summaries["1002"].balls == 5
    assert summaries["1001"].teams == ("Chennai Super Kings", "Mumbai Indians")

    balls = repo.load_balls("1001")
    assert len(balls) == 3
    assert balls[0].striker.name == "MS Dhoni"


def test_unknown_match_raises(tmp_path: Path) -> None:
    _write_match(tmp_path, "1001")
    repo = MatchRepository(tmp_path)
    with pytest.raises(UnknownMatchError):
        repo.load_balls("9999")


def test_allowed_ids_filters_the_listing(tmp_path: Path) -> None:
    _write_match(tmp_path, "1001")
    _write_match(tmp_path, "1002")
    repo = MatchRepository(tmp_path, allowed_ids=("1001",))
    assert [s.match_id for s in repo.list_matches()] == ["1001"]
    with pytest.raises(UnknownMatchError):
        repo.load_balls("1002")  # exists on disk but not allowed


def test_path_traversal_is_rejected(tmp_path: Path) -> None:
    _write_match(tmp_path, "1001")
    repo = MatchRepository(tmp_path)
    with pytest.raises(UnknownMatchError):
        repo.load_balls("../../etc/passwd")
