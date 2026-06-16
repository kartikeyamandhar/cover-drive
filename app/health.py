"""Trivial, dependency-free smoke module.

Phase 0 contains no product logic. This module exists only to give the
toolchain a real, typed target: pytest has something to assert, mypy has a
typed signature to check, and ruff has formatted code to lint. It is the
canary that proves the quality gates are wired before any feature lands.
"""

from __future__ import annotations

PROJECT_SLUG = "cricket-commentary"


def project_slug() -> str:
    """Return the project's stable identifier slug.

    Pure and deterministic: no IO, no globals mutated, same output every call.
    """
    return PROJECT_SLUG


def is_blank(text: str) -> bool:
    """Return ``True`` if ``text`` is empty or only whitespace.

    A small branching helper so branch coverage has something real to measure.
    """
    return len(text.strip()) == 0
