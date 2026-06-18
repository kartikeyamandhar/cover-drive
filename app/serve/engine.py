"""The commentary engine: turn a ball + persona into a validated line.

This is where the facts-versus-voice invariant is enforced at serving time. The model is
asked for voice (the exact training prompt), the finished line is checked against the
ground truth, and on failure the engine regenerates up to a bounded number of times before
substituting the deterministic, always-faithful fallback. A line that contradicts the facts
never leaves this function: ``faithful`` is always True on return (model or fallback).

Validation runs on the COMPLETE line (a score or chase equation spans the whole sentence),
so the engine generates fully and validates before the API replays the line to the client
token by token. The client only ever sees text that already passed the check.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.dataset.format import system_prompt, user_turn
from app.distill.filters import faithfulness_check
from app.serve.ball import ServeBall
from app.serve.runtime import RuntimeAdapter
from app.serve.scoreboard import fallback_line
from configs.personas import Persona


class CommentaryResult(BaseModel):
    """The validated line plus provenance for observability and the client."""

    model_config = ConfigDict(frozen=True)

    line: str
    faithful: bool
    source: str  # "model" | "fallback"
    attempts: int  # model draws made before returning (>= 1)
    reasons: list[str] = Field(default_factory=list)  # why model draws failed, if any


def build_prompt(ball: ServeBall, persona: Persona) -> tuple[str, str]:
    """The exact (system, user) prompt the student was trained on (train/serve parity)."""
    return system_prompt(persona), user_turn(ball.event, ball.state_string)


def _draw(runtime: RuntimeAdapter, system: str, user: str) -> str:
    """One model draw, reduced to a single trimmed line."""
    text = "".join(runtime.stream(system, user)).strip()
    return text.splitlines()[0].strip() if text else ""


def generate_commentary(
    runtime: RuntimeAdapter, ball: ServeBall, persona: Persona, *, retries: int = 2
) -> CommentaryResult:
    """Generate a validated line: the model's voice if faithful, else the fallback.

    Makes up to ``retries + 1`` model draws; the first faithful one wins. If none is
    faithful the deterministic ``fallback_line`` is returned (itself always faithful), so
    the result is faithful regardless of model behaviour.
    """
    system, user = build_prompt(ball, persona)
    reasons: list[str] = []
    for attempt in range(1, retries + 2):
        line = _draw(runtime, system, user)
        if line:
            check = faithfulness_check(line, ball.event, ball.state_string)
            if check.ok:
                return CommentaryResult(line=line, faithful=True, source="model", attempts=attempt)
            reasons = check.reasons
    return CommentaryResult(
        line=fallback_line(ball),
        faithful=True,
        source="fallback",
        attempts=retries + 1,
        reasons=reasons,
    )
