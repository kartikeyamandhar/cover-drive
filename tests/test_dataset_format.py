"""Tests for SFT instruction formatting."""

from __future__ import annotations

from app.dataset.format import SFT_SCHEMA_VERSION, format_pair, system_prompt, user_turn
from configs.personas import persona_by_key

_PAIR = {
    "match_id": "1082592",
    "ball_id": "1-19-4",
    "persona": "broadcast",
    "bucket": "six",
    "event": "SIX off the bat",
    "state": "T20 IPL | RPS 181/3 | need 4 off 2 | death",
    "commentary": "Smith launches it into the night!",
}


def test_system_prompt_names_the_persona_voice() -> None:
    prompt = system_prompt(persona_by_key("analyst"))
    assert "The Tactician" in prompt
    assert "never invent" in prompt.lower()


def test_user_turn_carries_event_and_state() -> None:
    turn = user_turn("SIX off the bat", "RPS 181/3")
    assert "BALL: SIX off the bat" in turn
    assert "STATE: RPS 181/3" in turn


def test_format_pair_builds_three_role_conversation() -> None:
    example = format_pair(_PAIR, "train")
    assert example.split == "train"
    assert example.schema_version == SFT_SCHEMA_VERSION
    assert [turn.role for turn in example.messages] == ["system", "user", "assistant"]
    assert "Broadcast Box" in example.messages[0].content
    assert "SIX off the bat" in example.messages[1].content
    assert "RPS 181/3" in example.messages[1].content
    assert example.messages[2].content == "Smith launches it into the night!"
    assert example.match_id == "1082592"
