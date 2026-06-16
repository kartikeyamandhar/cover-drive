"""Smoke tests proving pytest, coverage, and imports are wired."""

from __future__ import annotations

from app.health import is_blank, project_slug


def test_project_slug_is_stable() -> None:
    assert project_slug() == "cricket-commentary"


def test_is_blank_true_cases() -> None:
    assert is_blank("")
    assert is_blank("   \t\n  ")


def test_is_blank_false_case() -> None:
    assert not is_blank("over")
