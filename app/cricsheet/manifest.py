"""Persisted job state for the dataset build, enabling idempotent resumption.

The manifest records, per match, the input file hash and the produced record/innings
counts. A re-run skips any match whose input hash is unchanged; changing one match
reprocesses only that match. State lives on disk, not in memory (CLAUDE.md S7).
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

MANIFEST_SCHEMA_VERSION = 1


class MatchEntry(BaseModel):
    """Recorded build state for one match."""

    input_sha256: str
    n_records: int
    n_innings: int
    status: str = "built"


class Manifest(BaseModel):
    """The full build manifest, keyed by match id."""

    schema_version: int = MANIFEST_SCHEMA_VERSION
    matches: dict[str, MatchEntry] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        """Load a manifest, or return an empty one if absent or unreadable."""
        if not path.exists():
            return cls()
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError):
            return cls()

    def save(self, path: Path) -> None:
        """Write the manifest deterministically (sorted keys)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def needs_build(self, match_id: str, input_sha256: str) -> bool:
        """True if this match has not been built, or its input changed."""
        entry = self.matches.get(match_id)
        return entry is None or entry.input_sha256 != input_sha256

    def mark_built(self, match_id: str, input_sha256: str, n_records: int, n_innings: int) -> None:
        """Record a successful build for a match."""
        self.matches[match_id] = MatchEntry(
            input_sha256=input_sha256, n_records=n_records, n_innings=n_innings
        )
