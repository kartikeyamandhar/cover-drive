"""Integration tests for the FastAPI serving app + SSE, via TestClient (T6.5).

All headless: a ``StubRuntime`` stands in for the model, a tmp dir for the bundled data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.serve.app import create_app
from app.serve.matches import MatchRepository
from app.serve.runtime import StubRuntime
from configs.serve import ServeConfig


def _rec(
    ball_id: str, runs_batter: int, runs_total: int, score_runs: int, score_wickets: int
) -> str:
    return json.dumps(
        {
            "match_id": "1082591",
            "ball_id": ball_id,
            "innings": 0,
            "over": 0,
            "ball_in_over": int(ball_id.split("-")[-1]),
            "legal_ball_number": int(ball_id.split("-")[-1]),
            "is_legal": True,
            "batting_team": "Sunrisers Hyderabad",
            "bowling_team": "Royal Challengers Bangalore",
            "striker": {"id": "s", "name": "D Warner"},
            "non_striker": {"id": "n", "name": "S Dhawan"},
            "bowler": {"id": "b", "name": "C de Grandhomme"},
            "delivery": {
                "runs_batter": runs_batter,
                "runs_extras": 0,
                "runs_total": runs_total,
                "extras": {},
                "wickets": [],
            },
            "state": {
                "score_runs": score_runs,
                "score_wickets": score_wickets,
                "balls_left": 120 - int(ball_id.split("-")[-1]),
                "current_run_rate": 7.5,
                "phase": "powerplay",
                "last_deliveries": ["4", "0"],
            },
            "state_string": (
                f"T20 IPL | Inns 1 | 0.{ball_id.split('-')[-1]} | "
                f"Sunrisers Hyderabad {score_runs}/{score_wickets} | CRR 7.5 | powerplay"
            ),
        }
    )


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    lines = [_rec("0-0-1", 4, 4, 4, 0), _rec("0-0-2", 0, 0, 4, 0)]
    (tmp_path / "1082591.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    app = create_app(
        runtime=StubRuntime("A good ball, well played."),
        repository=MatchRepository(tmp_path),
        config=ServeConfig(pacing_seconds=0.0),
    )
    return TestClient(app)


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def test_personas_lists_all_four_voices(client: TestClient) -> None:
    body = client.get("/personas").json()
    keys = {p["key"] for p in body}
    assert keys == {"broadcast", "radio", "analyst", "text"}


def test_matches_listing_and_detail(client: TestClient) -> None:
    listing = client.get("/matches").json()
    assert listing[0]["match_id"] == "1082591"
    assert listing[0]["balls"] == 2
    assert client.get("/matches/1082591").json()["teams"] == [
        "Sunrisers Hyderabad",
        "Royal Challengers Bangalore",
    ]
    assert client.get("/matches/9999").status_code == 404


def test_stream_emits_state_then_tokens_then_ball(client: TestClient) -> None:
    resp = client.get("/matches/1082591/stream?persona=broadcast")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text

    # Event ordering for the first delivery: state -> token(s) -> ball, then a final done.
    assert body.index("event: state") < body.index("event: token") < body.index("event: ball")
    assert body.count("event: state") == 2  # one per delivery
    assert (
        body.rstrip().endswith('data: {"match_id": "1082591", "persona": "broadcast"}')
        or "event: done" in body
    )

    # The first scoreboard reflects the structured ground truth.
    first_state = body.split("event: state\ndata: ", 1)[1].split("\n", 1)[0]
    assert json.loads(first_state)["score"] == "4/0"

    # Every ball event carries a validated, faithful line.
    for block in body.split("event: ball\ndata: ")[1:]:
        payload = json.loads(block.split("\n", 1)[0])
        assert payload["faithful"] is True
        assert payload["source"] == "model"


def test_stream_rejects_unknown_persona_and_match(client: TestClient) -> None:
    assert client.get("/matches/1082591/stream?persona=bogus").status_code == 404
    assert client.get("/matches/9999/stream?persona=broadcast").status_code == 404
