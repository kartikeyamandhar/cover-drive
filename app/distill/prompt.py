"""Build the teacher prompt: a cached few-shot prefix + a volatile per-ball turn.

The stable prefix (system rules + the few-shot voice examples) is byte-identical
across every request, so it caches once and is read at ~0.1x cost thereafter. The
volatile part (the persona directive, the ball event, the state) goes AFTER the
cache breakpoint, in the user turn, so selecting a voice costs only output tokens.
"""

from __future__ import annotations

from anthropic.types import MessageParam, TextBlockParam

from app.distill.seed import EXEMPLARS
from configs.personas import PERSONAS, Persona

SYSTEM_PROMPT = (
    "You generate exactly one line of live cricket commentary for a single delivery.\n"
    "RULES:\n"
    "- Narrate only the delivery and the match situation given in BALL and STATE.\n"
    "- You may add natural descriptive color (the shot, the mood, the pressure) as voice.\n"
    "- Never state or contradict a hard fact you were not given: do not invent or change "
    "any number (runs, score, wickets, run rate, over) or the outcome of the ball, and use "
    "only the player and team names that appear in STATE.\n"
    "- Output exactly one line. No preamble, no quotation marks, no labels.\n"
    "- Write in the requested PERSONA's voice. Examples of each voice follow."
)


def _fewshot_text() -> str:
    """Render the deterministic few-shot block, grouped by persona."""
    parts: list[str] = ["VOICE EXAMPLES"]
    for persona in PERSONAS:
        parts.append(f"\n## {persona.display_name} ({persona.key})")
        for exemplar in EXEMPLARS:
            line = exemplar.lines.get(persona.key)
            if line is None:
                continue
            parts.append(f"BALL: {exemplar.event}\nSTATE: {exemplar.state}\nLINE: {line}")
    return "\n".join(parts)


def system_blocks() -> list[TextBlockParam]:
    """The stable system + few-shot prefix, with the cache breakpoint on the last block."""
    system: TextBlockParam = {"type": "text", "text": SYSTEM_PROMPT}
    fewshot: TextBlockParam = {
        "type": "text",
        "text": _fewshot_text(),
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }
    return [system, fewshot]


def user_message(event: str, state: str, persona: Persona) -> MessageParam:
    """The volatile per-ball turn (after the cache breakpoint)."""
    text = (
        f"PERSONA: {persona.key} - {persona.instruction}\n"
        f"BALL: {event}\n"
        f"STATE: {state}\n\n"
        f"Write one line of commentary in the {persona.display_name} voice."
    )
    return {"role": "user", "content": text}
