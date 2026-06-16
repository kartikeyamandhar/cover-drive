"""Typed errors for the distillation (target-generation) layer."""

from __future__ import annotations


class DistillError(Exception):
    """Base class for all distillation failures."""


class TeacherError(DistillError):
    """Raised when a teacher (Anthropic) API interaction fails irrecoverably."""


class BudgetExceededError(DistillError):
    """Raised when a run would exceed the configured teacher budget cap."""
