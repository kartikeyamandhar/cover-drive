"""Serialization: ball_id, state string, record schema, determinism, golden file."""

from __future__ import annotations

from pathlib import Path

from app.cricsheet.models import Match
from app.cricsheet.parse import parse_match
from app.features.serialize import (
    BALL_RECORD_SCHEMA_VERSION,
    ball_id,
    over_label,
    record_to_jsonl_line,
    state_string,
    to_record,
)
from app.features.state import BallContext, iter_ball_contexts
from configs.data import FeatureConfig

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample_match.json"
GOLDEN = FIXTURES / "sample_match.golden.jsonl"


def _match_and_contexts() -> tuple[Match, dict[str, BallContext]]:
    match = parse_match(SAMPLE)
    contexts = {ball_id(c): c for c in iter_ball_contexts(match, "sample_match", FeatureConfig())}
    return match, contexts


def test_ball_id_present() -> None:
    _, contexts = _match_and_contexts()
    assert "0-0-1" in contexts
    assert "1-0-3" in contexts


def test_record_schema_and_keys() -> None:
    match, contexts = _match_and_contexts()
    record = to_record(match.info, contexts["0-0-1"])
    assert record["schema_version"] == BALL_RECORD_SCHEMA_VERSION
    assert record["ball_id"] == "0-0-1"
    assert record["match_id"] == "sample_match"
    state = record["state"]
    assert isinstance(state, dict)
    assert state["score_runs"] == 4
    assert state["current_run_rate"] == 24.0


def test_record_line_is_deterministic() -> None:
    match, contexts = _match_and_contexts()
    line_a = record_to_jsonl_line(to_record(match.info, contexts["0-0-1"]))
    line_b = record_to_jsonl_line(to_record(match.info, contexts["0-0-1"]))
    assert line_a == line_b


def test_state_string_carries_facts() -> None:
    match, contexts = _match_and_contexts()
    text = state_string(match.info, contexts["1-0-1"])
    assert "Team Beta 1/0" in text
    assert "need 15 off 5" in text
    assert "RRR 18.0" in text


def test_over_label_marks_wide() -> None:
    _, contexts = _match_and_contexts()
    assert over_label(contexts["0-0-3"]).endswith("+wide")
    assert over_label(contexts["0-0-1"]) == "0.1"  # 0-indexed over.ball


def test_serialization_matches_golden() -> None:
    match, _ = _match_and_contexts()
    lines = [
        record_to_jsonl_line(to_record(match.info, c))
        for c in iter_ball_contexts(match, "sample_match", FeatureConfig())
    ]
    produced = "\n".join(lines) + "\n"
    assert produced == GOLDEN.read_text(encoding="utf-8")
