"""Parsing: schema validation, registry resolution, and typed errors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.cricsheet.errors import CricsheetParseError
from app.cricsheet.parse import match_id_from_path, parse_match, resolve_person_id

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample_match.json"


def test_parse_sample_match() -> None:
    match = parse_match(SAMPLE)
    assert match.info.teams == ["Team Alpha", "Team Beta"]
    assert match.info.balls_per_over == 6
    assert match.info.overs == 1
    assert len(match.innings) == 2
    assert match.innings[0].overs[0].deliveries[0].runs.batter == 4
    assert match.innings[1].target is not None
    assert match.innings[1].target.runs == 16


def test_registry_resolution() -> None:
    match = parse_match(SAMPLE)
    assert resolve_person_id(match.info, "Alpha One") == "aaaa0001"
    # missing registry entry falls back to the name (still distinct per person)
    assert resolve_person_id(match.info, "Nobody") == "Nobody"


def test_match_id_from_path() -> None:
    assert match_id_from_path(Path("/data/extracted/1234567.json")) == "1234567"


def test_missing_file_raises() -> None:
    with pytest.raises(CricsheetParseError):
        parse_match(FIXTURES / "does_not_exist.json")


def test_malformed_json_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(CricsheetParseError):
        parse_match(bad)


def test_schema_violation_raises(tmp_path: Path) -> None:
    # info present but missing the required `teams` field
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"info": {"registry": {"people": {}}}}), encoding="utf-8")
    with pytest.raises(CricsheetParseError):
        parse_match(bad)


def test_unknown_fields_are_ignored(tmp_path: Path) -> None:
    payload = {
        "info": {"teams": ["A", "B"], "surprise_field": 1},
        "innings": [],
    }
    path = tmp_path / "ok.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    match = parse_match(path)
    assert match.info.teams == ["A", "B"]
