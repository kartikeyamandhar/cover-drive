"""Typed reader for the Phase 1 per-ball JSONL records.

A lightweight pydantic view over the serialized record (``extra="ignore"`` so only
the fields distillation needs are declared). Gives the sampler and filters typed
field access instead of dict-spelunking.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.features.state import NON_DISMISSAL_KINDS


class _Person(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    name: str


class _Wicket(BaseModel):
    model_config = ConfigDict(extra="ignore")
    player_out_name: str | None = None
    kind: str
    credited_to_bowler: bool = False


class _Delivery(BaseModel):
    model_config = ConfigDict(extra="ignore")
    runs_batter: int
    runs_extras: int
    runs_total: int
    extras: dict[str, int] = Field(default_factory=dict)
    wickets: list[_Wicket] = Field(default_factory=list)


class _State(BaseModel):
    model_config = ConfigDict(extra="ignore")
    phase: str
    required_run_rate: float | None = None
    runs_required: int | None = None
    balls_left: int
    milestones: list[str] = Field(default_factory=list)
    bowler_on_hat_trick: bool = False
    striker_runs: int = 0


class BallRecord(BaseModel):
    """The subset of a per-ball record that distillation reads."""

    model_config = ConfigDict(extra="ignore")

    match_id: str
    ball_id: str
    innings: int
    is_legal: bool
    striker: _Person
    bowler: _Person
    delivery: _Delivery
    state: _State
    state_string: str


def parse_record(line: str) -> BallRecord:
    """Parse one JSONL line into a typed ``BallRecord``."""
    return BallRecord.model_validate_json(line)


def has_real_wicket(record: BallRecord) -> bool:
    """True if a real fall of wicket (excludes retired hurt / not out)."""
    return any(w.kind not in NON_DISMISSAL_KINDS for w in record.delivery.wickets)


def delivery_event(
    *, runs_batter: int, runs_total: int, extras: dict[str, int], wicket_kinds: list[str]
) -> str:
    """The short factual phrase for a delivery, from its raw outcome fields.

    The single source of truth for the BALL: event, shared by distillation
    (``ball_event``) and serving (``app.serve.ball.ServeBall.event``) so the prompt
    the student is served is byte-identical to the prompt it was trained on.
    """
    real = [k for k in wicket_kinds if k not in NON_DISMISSAL_KINDS]
    if real:
        return f"WICKET, {real[0]}"
    if runs_batter == 6:
        return "SIX off the bat"
    if runs_batter == 4:
        return "FOUR off the bat"
    if extras.get("wides"):
        return "wide"
    if extras.get("noballs"):
        return "no-ball"
    if runs_total == 0:
        return "dot ball"
    return f"{runs_total} run" if runs_total == 1 else f"{runs_total} runs"


def ball_event(record: BallRecord) -> str:
    """A short factual phrase for what just happened on this ball (no color)."""
    delivery = record.delivery
    return delivery_event(
        runs_batter=delivery.runs_batter,
        runs_total=delivery.runs_total,
        extras=delivery.extras,
        wicket_kinds=[w.kind for w in delivery.wickets],
    )
