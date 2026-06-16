"""Parse a Cricsheet match JSON file into typed domain models.

IO at the edge: this module reads a file and returns a validated ``Match`` or
raises a typed ``CricsheetParseError``. All downstream code works on the typed
models, never on raw dicts.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from app.cricsheet.errors import CricsheetParseError
from app.cricsheet.models import Match, MatchInfo


def match_id_from_path(path: Path) -> str:
    """Return the match id (the Cricsheet filename stem, e.g. ``1234567``)."""
    return path.stem


def parse_match(path: Path) -> Match:
    """Load and validate one match file.

    Raises:
        CricsheetParseError: if the file is missing, not valid JSON, or does not
            match the expected schema.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CricsheetParseError(f"cannot read match file {path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CricsheetParseError(f"invalid JSON in {path}: {exc}") from exc

    try:
        return Match.model_validate(data)
    except ValidationError as exc:
        raise CricsheetParseError(f"schema validation failed for {path}: {exc}") from exc


def resolve_person_id(info: MatchInfo, name: str) -> str:
    """Resolve a display name to its stable registry id.

    Falls back to the name itself when the registry has no entry, so stat keying
    stays deterministic on imperfect source data. The name is still distinct per
    person within a match, so the fallback does not merge people.
    """
    return info.registry.people.get(name, name)
