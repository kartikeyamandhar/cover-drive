"""Instruction formatting: a distill pair becomes a chat-style SFT example.

The student is fine-tuned, so unlike the teacher it needs no few-shot block: it
carries the voice in its weights. Its prompt is just a system turn (role + the
persona voice + the facts-versus-voice rule) and a user turn (the ball + state).
The persona lives in the system prompt, so at serving (Phase 6) switching the
persona descriptor switches the voice - that is the product's headline demo.

The output is the portable ``messages`` format (a list of role/content turns).
Phase 4 applies the Qwen tokenizer's chat template to it; we do not hand-roll
ChatML tokens here, to avoid drifting from the official template.
"""

from __future__ import annotations

from pydantic import BaseModel

from configs.personas import Persona, persona_by_key

SFT_SCHEMA_VERSION = 1

_SYSTEM_TEMPLATE = (
    "You are a cricket commentator giving exactly one line of live ball-by-ball "
    "commentary in the {display_name} voice.\n"
    "{instruction}\n"
    "Narrate only the delivery and the match situation you are given. You may add "
    "natural descriptive color, but never invent or change a number (runs, score, "
    "wickets, rate, over), a player or team name, or the outcome of the ball. "
    "Output exactly one line."
)


class ChatTurn(BaseModel):
    """One turn in the conversation."""

    role: str
    content: str


class SFTExample(BaseModel):
    """A single supervised fine-tuning example, plus metadata for analysis."""

    schema_version: int = SFT_SCHEMA_VERSION
    match_id: str
    ball_id: str
    persona: str
    bucket: str
    split: str
    messages: list[ChatTurn]


def system_prompt(persona: Persona) -> str:
    """The student's system prompt for a persona (no few-shot)."""
    return _SYSTEM_TEMPLATE.format(
        display_name=persona.display_name, instruction=persona.instruction
    )


def user_turn(event: str, state: str) -> str:
    """The student's user turn: the factual ball event and state."""
    return f"BALL: {event}\nSTATE: {state}"


def format_pair(pair: dict[str, str], split: str) -> SFTExample:
    """Turn a Phase 2 distill pair into an ``SFTExample`` for the given split."""
    persona = persona_by_key(pair["persona"])
    messages = [
        ChatTurn(role="system", content=system_prompt(persona)),
        ChatTurn(role="user", content=user_turn(pair["event"], pair["state"])),
        ChatTurn(role="assistant", content=pair["commentary"]),
    ]
    return SFTExample(
        match_id=pair["match_id"],
        ball_id=pair["ball_id"],
        persona=pair["persona"],
        bucket=pair["bucket"],
        split=split,
        messages=messages,
    )
