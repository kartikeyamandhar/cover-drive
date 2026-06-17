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
    "- STATE is the situation AFTER this ball. The score, the wickets, the chase equation "
    "(need R off B), the striker's runs(balls) and the bowler's wickets/runs ALREADY "
    "include this delivery. Read these numbers as final and say them exactly as given. Do "
    "NOT add this ball's runs again, do NOT add one to any wicket tally, and do NOT take a "
    "ball off the equation.\n"
    "- This holds EVEN as you describe the ball. After 'worked away for one', the equation "
    "is STILL exactly what STATE says: the single is already counted. If STATE says 'need 3 "
    "off 3', say 3 off 3 -- never 2 off 2. If STATE says 'need 8 off 8', it is 8 off 8, "
    "never 8 off 7. The balls-left number is the balls remaining AFTER this one.\n"
    "- Wicket count: the '/W' in the team score is the number already out INCLUDING this "
    "wicket. If STATE shows 19/1 on a wicket ball, this IS the first wicket -- say 'their "
    "first' or 'one down', never 'their second' or 'two down'. Wickets still in hand = 10 "
    "minus that number. If the bowler shows 2 wickets, this ball is their 2nd, not a 3rd.\n"
    "- The striker's runs are final: if STATE shows the striker on 42, they are on 42 (not "
    "46), even on a boundary. A milestone listed under 'nearing' has NOT been reached yet: "
    "do not say a fifty/hundred/team-hundred was brought up when STATE only flags 'nearing'.\n"
    "- The over label O.B means O completed overs and B balls, so play is in over number "
    "O+1 (a label of 15.x is the 16th over; 1.6 is the end of the 2nd over).\n"
    "- Fielders' names are not given; on a catch, credit only the bowler in STATE or say "
    "'caught' without naming the fielder.\n"
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
