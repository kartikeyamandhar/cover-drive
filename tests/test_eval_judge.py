"""Tests for the LLM judge. The Anthropic SDK is mocked; no live calls."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import anthropic

from app.eval.judge import (
    FAILURE_MODES,
    JudgeResult,
    Verdict,
    judge_cost,
    judge_one,
    system_blocks,
    user_message,
)

_VERDICT = {
    "faithful": False,
    "severity": "major",
    "failure_modes": ["hallucinated_name"],
    "persona_match": True,
    "confidence": "high",
    "explanation": "names a catcher not present in STATE",
}


def _fake_message(*, tool: dict[str, object] | None, text: str = "") -> SimpleNamespace:
    content: list[SimpleNamespace] = []
    if tool is not None:
        content.append(SimpleNamespace(type="tool_use", name="record_verdict", input=tool))
    if text:
        content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(
        content=content,
        usage=SimpleNamespace(
            input_tokens=900,
            output_tokens=80,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


class _FakeMessages:
    def __init__(self, message: SimpleNamespace) -> None:
        self.message = message
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.message


class _FakeClient:
    def __init__(self, message: SimpleNamespace) -> None:
        self.messages = _FakeMessages(message)


def _client(message: SimpleNamespace) -> tuple[anthropic.Anthropic, _FakeClient]:
    fake = _FakeClient(message)
    return cast(anthropic.Anthropic, fake), fake


def _judge(message: SimpleNamespace) -> JudgeResult:
    client, _ = _client(message)
    return judge_one(
        client,
        persona_key="broadcast",
        persona_instruction="lead caller",
        event="WICKET, caught",
        state="... 88/3 | Striker S Iyer 41(29) | Bowler R Jadeja 1/22 ...",
        line="Caught by Rohit Sharma at deep midwicket!",
    )


def test_judge_parses_tool_verdict() -> None:
    result = _judge(_fake_message(tool=_VERDICT))
    assert result.verdict is not None
    assert result.verdict.faithful is False
    assert result.verdict.failure_modes == ["hallucinated_name"]
    assert result.input_tokens == 900


def test_bulk_mode_forces_the_tool() -> None:
    client, fake = _client(_fake_message(tool=_VERDICT))
    judge_one(
        client,
        persona_key="broadcast",
        persona_instruction="x",
        event="dot ball",
        state="STATE",
        line="No run.",
    )
    call = fake.messages.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "record_verdict"}
    assert "thinking" not in call


def test_thinking_mode_leaves_tool_choice_auto() -> None:
    client, fake = _client(_fake_message(tool=_VERDICT))
    judge_one(
        client,
        persona_key="broadcast",
        persona_instruction="x",
        event="dot ball",
        state="STATE",
        line="No run.",
        use_thinking=True,
    )
    call = fake.messages.calls[0]
    assert call["thinking"] == {"type": "adaptive"}
    assert "tool_choice" not in call


def test_judge_falls_back_to_json_text() -> None:
    msg = _fake_message(
        tool=None,
        text='reasoning... {"faithful": true, "severity": "none", '
        '"failure_modes": [], "persona_match": true, "confidence": "high", "explanation": "ok"}',
    )
    result = _judge(msg)
    assert result.verdict is not None
    assert result.verdict.faithful is True


def test_judge_returns_none_on_garbage() -> None:
    result = _judge(_fake_message(tool=None, text="no structured output here"))
    assert result.verdict is None


def test_judge_rejects_invalid_tool_payload() -> None:
    bad = {**_VERDICT, "severity": "catastrophic"}  # not in the enum
    result = _judge(_fake_message(tool=bad))
    assert result.verdict is None


def test_system_block_is_cacheable() -> None:
    blocks = system_blocks()
    assert blocks[-1].get("cache_control") == {"type": "ephemeral", "ttl": "1h"}


def test_user_message_carries_all_fields() -> None:
    msg = user_message("radio", "paint the picture", "FOUR off the bat", "RCB 162/4", "Four!")
    content = msg["content"]
    assert isinstance(content, str)
    assert "PERSONA: radio" in content
    assert "EVENT: FOUR off the bat" in content
    assert "LINE: Four!" in content


def test_judge_cost_formula() -> None:
    result = JudgeResult(
        verdict=Verdict.model_validate(_VERDICT),
        input_tokens=1000,
        output_tokens=100,
        cache_read_tokens=0,
        cache_write_tokens=0,
        raw="",
    )
    # opus: (1000*5 + 100*25) / 1e6
    assert abs(judge_cost(result) - (1000 * 5 + 100 * 25) / 1e6) < 1e-12


def test_taxonomy_has_the_core_modes() -> None:
    for mode in ("hallucinated_name", "invented_event", "wrong_wicket_count", "persona_bleed"):
        assert mode in FAILURE_MODES
