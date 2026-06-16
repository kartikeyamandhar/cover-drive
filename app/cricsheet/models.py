"""Typed domain models for a parsed Cricsheet match.

These mirror the subset of the Cricsheet JSON schema that featurization needs.
Unknown fields are ignored (``extra="ignore"``) so forward schema drift does not
break parsing. Names in deliveries are resolved to stable registry ids by the
featurizer, never used directly as stat keys.

Schema reference: https://cricsheet.org/format/json/
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    """Base model: ignore unknown keys for forward compatibility."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class Registry(_Base):
    """Maps a person's display name to a stable 8-character identifier."""

    people: dict[str, str] = Field(default_factory=dict)


class Toss(_Base):
    """Toss result."""

    winner: str | None = None
    decision: str | None = None


class Event(_Base):
    """Competition/event context (optional in the source)."""

    name: str | None = None
    match_number: int | None = None


class MatchInfo(_Base):
    """The ``info`` block: match-level context featurization depends on."""

    teams: list[str]
    players: dict[str, list[str]] = Field(default_factory=dict)
    registry: Registry = Field(default_factory=Registry)
    dates: list[str] = Field(default_factory=list)
    gender: str | None = None
    match_type: str | None = None
    season: str | int | None = None
    team_type: str | None = None
    balls_per_over: int = 6
    overs: int | None = None
    city: str | None = None
    venue: str | None = None
    event: Event | None = None
    toss: Toss | None = None


class Extras(_Base):
    """Extra runs on a delivery, by type. Absent fields are zero."""

    wides: int = 0
    byes: int = 0
    legbyes: int = 0
    noballs: int = 0
    penalty: int = 0


class Runs(_Base):
    """Runs scored on a delivery."""

    batter: int
    extras: int
    total: int
    non_boundary: bool = False


class Fielder(_Base):
    """A fielder involved in a dismissal (object form in the source)."""

    name: str | None = None
    substitute: bool = False


class Wicket(_Base):
    """A dismissal recorded on a delivery."""

    player_out: str
    kind: str
    fielders: list[Fielder] = Field(default_factory=list)


class Delivery(_Base):
    """A single delivery. ``batter``/``non_striker``/``bowler`` are names.

    Strike is given directly by the source: the on-strike batter is ``batter``;
    do not re-derive strike rotation.
    """

    batter: str
    bowler: str
    non_striker: str
    runs: Runs
    extras: Extras | None = None
    wickets: list[Wicket] = Field(default_factory=list)


class Over(_Base):
    """One over: a 0-indexed number and its ordered deliveries."""

    over: int
    deliveries: list[Delivery] = Field(default_factory=list)


class Powerplay(_Base):
    """A powerplay span within an innings.

    ``from``/``to`` are over.ball markers encoded as numbers in the source
    (for example ``0.1`` and ``5.6``), not strings.
    """

    from_: float = Field(alias="from")
    to: float
    type: str


class Target(_Base):
    """The chasing innings target (set, and DLS-adjusted when rain-affected)."""

    runs: int
    overs: float | None = None


class Innings(_Base):
    """One innings: batting team, ordered overs, and optional context."""

    team: str
    overs: list[Over] = Field(default_factory=list)
    powerplays: list[Powerplay] = Field(default_factory=list)
    target: Target | None = None
    super_over: bool = False


class Meta(_Base):
    """File metadata."""

    data_version: str | None = None
    created: str | None = None
    revision: int | None = None


class Match(_Base):
    """A full parsed match: metadata, info, and innings."""

    meta: Meta | None = None
    info: MatchInfo
    innings: list[Innings] = Field(default_factory=list)
