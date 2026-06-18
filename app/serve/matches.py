"""The bundled-match repository: the demo's source of deliveries to replay.

Reads the Phase 1 per-ball JSONL (one file per match) under a configured directory and
exposes a list of matches plus the ordered deliveries for one. Inputs are validated against
the set of known match ids, so a client-supplied id is never turned into a filesystem path
it could traverse with.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from app.serve.ball import ServeBall, parse_serve_ball


class MatchSummary(BaseModel):
    """A short, listable description of a bundled match, derived from its deliveries."""

    model_config = ConfigDict(frozen=True)

    match_id: str
    teams: tuple[str, str]  # (batting-first, bowling-first)
    innings: int
    balls: int


class UnknownMatchError(KeyError):
    """Raised when a match id is not among the bundled matches."""


class MatchRepository:
    """Lists and loads bundled matches from a directory of per-ball JSONL files."""

    def __init__(
        self, processed_dir: Path, allowed_ids: Sequence[str] = (), max_listed: int | None = None
    ) -> None:
        self._dir = processed_dir
        self._allowed = tuple(allowed_ids)
        self._max_listed = max_listed
        self._summaries: dict[str, MatchSummary] | None = None

    def _all_ids(self) -> set[str]:
        """Every match present on disk (the set that can be replayed)."""
        return {p.stem for p in self._dir.glob("*.jsonl")}

    def _known_ids(self) -> list[str]:
        """The subset shown by ``/matches`` (allow-list, else a capped discovery)."""
        ids = sorted(self._all_ids())
        if self._allowed:
            # An explicit allow-list is authoritative and never capped.
            allowed = set(self._allowed)
            return [match_id for match_id in ids if match_id in allowed]
        if self._max_listed is not None:
            ids = ids[: self._max_listed]
        return ids

    def _summarize(self, match_id: str) -> MatchSummary:
        lines = [
            line
            for line in (self._dir / f"{match_id}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        first = parse_serve_ball(lines[0])
        last = parse_serve_ball(lines[-1])
        return MatchSummary(
            match_id=match_id,
            teams=(first.batting_team, first.bowling_team),
            innings=last.innings + 1,
            balls=len(lines),
        )

    def list_matches(self) -> list[MatchSummary]:
        """All bundled matches (memoized after first call)."""
        if self._summaries is None:
            self._summaries = {mid: self._summarize(mid) for mid in self._known_ids()}
        return list(self._summaries.values())

    def load_balls(self, match_id: str) -> list[ServeBall]:
        """The ordered deliveries for any match present on disk (not just the listed set).

        Validated against the full processed set, which both serves the whole catalog and
        blocks path traversal (an id is only honoured if a ``<id>.jsonl`` exists)."""
        if match_id not in self._all_ids():
            raise UnknownMatchError(match_id)
        lines = (self._dir / f"{match_id}.jsonl").read_text(encoding="utf-8").splitlines()
        return [parse_serve_ball(line) for line in lines if line.strip()]
