"""Tests for personas and the cached-prefix prompt builder."""

from __future__ import annotations

from app.distill.prompt import system_blocks, user_message
from configs.personas import (
    PERSONAS,
    persona_by_key,
    primary_persona,
    secondary_personas,
)


def test_four_distinct_personas() -> None:
    keys = [p.key for p in PERSONAS]
    assert keys == ["broadcast", "radio", "analyst", "text"]
    assert len(set(keys)) == 4


def test_primary_and_secondary_split() -> None:
    assert primary_persona().key == "broadcast"
    assert {p.key for p in secondary_personas()} == {"radio", "analyst", "text"}


def test_persona_by_key() -> None:
    assert persona_by_key("analyst").display_name == "The Tactician"


def test_cache_breakpoint_on_last_system_block() -> None:
    blocks = system_blocks()
    assert len(blocks) == 2
    assert "cache_control" not in blocks[0]
    assert blocks[1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    # the few-shot block names every persona
    fewshot = blocks[1]["text"]
    for persona in PERSONAS:
        assert persona.display_name in fewshot


def test_user_message_carries_persona_and_state() -> None:
    message = user_message("SIX off the bat", "CSK 162/4 | death", persona_by_key("broadcast"))
    assert message["role"] == "user"
    content = message["content"]
    assert isinstance(content, str)
    assert "PERSONA: broadcast" in content
    assert "SIX off the bat" in content
    assert "CSK 162/4" in content
