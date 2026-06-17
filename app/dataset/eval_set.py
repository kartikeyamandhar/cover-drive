"""Held-out real-commentary eval reference (used by the Phase 5 style judge).

LICENSING: the real broadcast corpus is gray-license. The reference is local-only,
git-ignored, and NEVER committed or pushed. This module only loads a local file; a
small hand-authored CLEAN fixture is committed for tests. The actual gray-corpus
held-out slice is assembled locally at eval time from held-out (val/test) matches.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_reference(path: Path) -> list[str]:
    """Load a held-out real-commentary reference, or an empty list if absent.

    Accepts either bare lines or JSONL objects with a ``text`` field.
    """
    if not path.exists():
        return []
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("{"):
            lines.append(str(json.loads(stripped).get("text", "")).strip())
        else:
            lines.append(stripped)
    return [line for line in lines if line]
