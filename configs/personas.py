"""Commentary personas: the voices the fine-tune learns to switch between.

Each persona is a pure voice definition. The ``instruction`` is the per-request
directive placed AFTER the prompt-cache breakpoint, so all personas share one
cached few-shot prefix and selecting a voice costs only output tokens. The
facts-versus-voice rule lives in the shared system prompt, not here: every
persona narrates only the facts in the supplied state and invents nothing.

All four are generic archetypes, never a named living commentator's identity or
catchphrases.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Persona(BaseModel):
    """A commentary voice."""

    model_config = ConfigDict(frozen=True)

    key: str
    display_name: str
    instruction: str
    primary: bool = False


PERSONAS: tuple[Persona, ...] = (
    Persona(
        key="broadcast",
        display_name="Broadcast Box",
        primary=True,
        instruction=(
            "Voice: lead TV broadcast caller. Speak in the present tense at the moment of "
            "contact. High energy that crests on a boundary or wicket and settles back to "
            "the match equation. Short, punchy bursts; standard broadcast lexicon."
        ),
    ),
    Persona(
        key="radio",
        display_name="Radio Call",
        instruction=(
            "Voice: radio commentator. The listener cannot see the field, so paint the "
            "picture: the delivery, where the ball went, the shot, the reaction. Flowing, "
            "vivid, descriptive prose with measured energy, never frantic."
        ),
    ),
    Persona(
        key="analyst",
        display_name="The Tactician",
        instruction=(
            "Voice: calm ex-cricketer analyst. Read the game, not just the moment: the "
            "matchup, the pressure of the equation, the phase logic. Past or perfect tense, "
            "evaluative, longer measured clauses, low exclamation. Explain only what the "
            "supplied facts support; invent no field placings, matchups, or stats."
        ),
    ),
    Persona(
        key="text",
        display_name="The Wire",
        instruction=(
            "Voice: terse written ball-by-ball text commentary. A compact, structured note "
            "of the delivery and its outcome. Factual and clipped, no spoken exclamation."
        ),
    ),
)


def persona_by_key(key: str) -> Persona:
    """Return the persona with this key, or raise ``KeyError``."""
    for persona in PERSONAS:
        if persona.key == key:
            return persona
    raise KeyError(f"unknown persona: {key!r}")


def primary_persona() -> Persona:
    """Return the single primary persona (the deep, full-coverage voice)."""
    return next(p for p in PERSONAS if p.primary)


def secondary_personas() -> tuple[Persona, ...]:
    """Return the secondary personas (lighter, subset coverage)."""
    return tuple(p for p in PERSONAS if not p.primary)
