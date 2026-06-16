"""Tests for the teacher client. The Anthropic SDK is mocked; no live calls."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import anthropic

from app.distill.teacher import (
    Generation,
    approx_seed_tokens,
    build_message_params,
    count_seed_tokens,
    estimate_cost,
    generate_one,
    generation_cost,
    warm_cache,
)
from configs.distill import DistillConfig
from configs.personas import primary_persona


def _fake_message(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=20,
            cache_read_input_tokens=100,
            cache_creation_input_tokens=0,
        ),
    )


class _FakeMessages:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return _fake_message(self.text)

    def count_tokens(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(input_tokens=4524)


class _FakeClient:
    def __init__(self, text: str = "A line of commentary.") -> None:
        self.messages = _FakeMessages(text)


def _client(text: str = "A line of commentary.") -> anthropic.Anthropic:
    return cast(anthropic.Anthropic, _FakeClient(text))


def test_generate_one_extracts_text_and_usage() -> None:
    gen = generate_one(
        _client("Bowled him!"), "WICKET, bowled", "STATE", primary_persona(), DistillConfig()
    )
    assert gen.text == "Bowled him!"
    assert gen.input_tokens == 10
    assert gen.cache_read_tokens == 100


def test_sonnet_sends_sampling_params() -> None:
    params = build_message_params("dot ball", "STATE", primary_persona(), DistillConfig())
    assert params["model"] == "claude-sonnet-4-6"
    assert "temperature" in params
    assert params.get("thinking") == {"type": "disabled"}
    # cache breakpoint present on the few-shot block
    system = params["system"]
    assert isinstance(system, list)
    assert system[1].get("cache_control") == {"type": "ephemeral", "ttl": "1h"}


def test_opus_omits_sampling_params() -> None:
    config = DistillConfig(teacher_model="claude-opus-4-8")
    params = build_message_params("dot ball", "STATE", primary_persona(), config)
    assert "temperature" not in params
    assert "thinking" not in params


def test_estimate_cost_cache_beats_no_cache() -> None:
    config = DistillConfig()
    cached = estimate_cost(10_000, 4524, config=config, assume_cache_hits=True, assume_batch=True)
    uncached = estimate_cost(
        10_000, 4524, config=config, assume_cache_hits=False, assume_batch=True
    )
    assert cached.est_usd < uncached.est_usd
    assert cached.est_usd > 0


def test_generation_cost_matches_formula() -> None:
    gen = Generation(
        text="x", input_tokens=10, output_tokens=20, cache_read_tokens=100, cache_write_tokens=0
    )
    # sonnet: (10*3 + 100*3*0.1 + 0 + 20*15) / 1e6
    expected = (10 * 3 + 100 * 3 * 0.1 + 20 * 15) / 1e6
    assert abs(generation_cost(gen, DistillConfig(), batch=False) - expected) < 1e-12


def test_approx_seed_tokens_clears_sonnet_floor() -> None:
    assert approx_seed_tokens() > 2048


def test_count_seed_tokens_uses_count_tokens() -> None:
    assert count_seed_tokens(_client(), DistillConfig()) == 4524


def test_warm_cache_uses_one_output_token() -> None:
    client = _FakeClient()
    warm_cache(cast(anthropic.Anthropic, client), DistillConfig())
    assert client.messages.calls[0]["max_tokens"] == 1
