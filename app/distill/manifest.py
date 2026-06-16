"""Resumable generation state for distillation, keyed on (match, ball, persona).

Lets a long teacher run resume after a crash without re-spending on completed
items. State is on disk, not in memory.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

MANIFEST_SCHEMA_VERSION = 1


def item_key(match_id: str, ball_id: str, persona: str) -> str:
    """The stable key for one generation work item."""
    return f"{match_id}|{ball_id}|{persona}"


class DistillEntry(BaseModel):
    """The recorded outcome for one item."""

    status: str  # "kept" | "rejected_faithfulness" | "rejected_seed_dedup" | "failed"


class DistillManifest(BaseModel):
    """Per-item generation state."""

    schema_version: int = MANIFEST_SCHEMA_VERSION
    items: dict[str, DistillEntry] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> DistillManifest:
        """Load a manifest, or an empty one if absent or unreadable."""
        if not path.exists():
            return cls()
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError):
            return cls()

    def save(self, path: Path) -> None:
        """Write deterministically (sorted keys)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def is_done(self, key: str) -> bool:
        """True if this item was already processed."""
        return key in self.items

    def mark(self, key: str, status: str) -> None:
        """Record an item's outcome."""
        self.items[key] = DistillEntry(status=status)
