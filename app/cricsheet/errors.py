"""Typed errors for the Cricsheet acquisition and parsing layer.

Errors are typed and surfaced, never swallowed (CLAUDE.md Section 7). Each failure
mode in the Phase 1 red-team maps to one of these.
"""

from __future__ import annotations


class CricsheetError(Exception):
    """Base class for all Cricsheet acquisition and parsing failures."""


class AcquireError(CricsheetError):
    """Raised when downloading or locating a source archive fails."""


class UnsafeArchiveError(AcquireError):
    """Raised when an archive entry is unsafe to extract.

    Covers absolute paths, parent-directory (``..``) traversal, and archives that
    exceed the configured uncompressed-size or entry-count caps (zip-bomb guard).
    """


class CricsheetParseError(CricsheetError):
    """Raised when a match file is missing, malformed, or fails schema validation."""
