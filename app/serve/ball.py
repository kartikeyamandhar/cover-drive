"""The serving-side typed view of a Phase 1 per-ball record (the full state).

Parallel to ``app.distill.records.BallRecord``, which intentionally exposes only the subset
distillation reads. Serving renders the entire scoreboard, so this view declares the full
``state``. ``extra="ignore"`` keeps it tolerant of schema growth. The ground-truth source is
``app.features.serialize.to_record``; this is its read side for Phase 6.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.distill.records import delivery_event


class Person(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    name: str


class Wicket(BaseModel):
    model_config = ConfigDict(extra="ignore")
    player_out_name: str | None = None
    kind: str
    credited_to_bowler: bool = False


class Delivery(BaseModel):
    model_config = ConfigDict(extra="ignore")
    runs_batter: int
    runs_extras: int
    runs_total: int
    extras: dict[str, int] = Field(default_factory=dict)
    wickets: list[Wicket] = Field(default_factory=list)


class BallState(BaseModel):
    model_config = ConfigDict(extra="ignore")
    score_runs: int
    score_wickets: int
    balls_left: int
    current_run_rate: float | None = None
    required_run_rate: float | None = None
    target_runs: int | None = None
    runs_required: int | None = None
    striker_runs: int = 0
    striker_balls: int = 0
    bowler_runs: int = 0
    bowler_balls: int = 0
    bowler_wickets: int = 0
    bowler_on_hat_trick: bool = False
    partnership_runs: int = 0
    partnership_balls: int = 0
    last_deliveries: list[str] = Field(default_factory=list)
    phase: str
    milestones: list[str] = Field(default_factory=list)


class ServeBall(BaseModel):
    """Full typed per-ball record for serving (the scoreboard plus the prompt)."""

    model_config = ConfigDict(extra="ignore")

    match_id: str
    ball_id: str
    innings: int
    over: int  # 0-based over number (scorecard notation: first over is 0.1..0.6)
    ball_in_over: int
    legal_ball_number: int  # cumulative legal balls this innings, after this ball
    is_legal: bool
    batting_team: str
    bowling_team: str
    striker: Person
    non_striker: Person
    bowler: Person
    delivery: Delivery
    state: BallState
    state_string: str

    @property
    def event(self) -> str:
        """The factual BALL: phrase, identical to the training/distillation event."""
        return delivery_event(
            runs_batter=self.delivery.runs_batter,
            runs_total=self.delivery.runs_total,
            extras=self.delivery.extras,
            wicket_kinds=[w.kind for w in self.delivery.wickets],
        )


def parse_serve_ball(line: str) -> ServeBall:
    """Parse one Phase 1 JSONL record line into a typed ``ServeBall``."""
    return ServeBall.model_validate_json(line)
