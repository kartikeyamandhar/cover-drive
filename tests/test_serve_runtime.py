"""Tests for the serving config and the stub runtime (Phase 6 T6.1)."""

from __future__ import annotations

from app.serve.runtime import RuntimeAdapter, StubRuntime
from configs.serve import ServeConfig


def test_serve_config_defaults_target_local_fp16() -> None:
    cfg = ServeConfig()
    # The LoRA applies to the canonical base on any device; bnb-4bit is CUDA-only.
    assert cfg.base_model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.adapter_repo.endswith("qwen2.5-1.5b-lora")
    assert cfg.device == "auto"
    assert cfg.faithfulness_retries >= 1
    assert cfg.eos_token == "<|im_end|>"


def test_stub_runtime_streams_fixed_line() -> None:
    stub = StubRuntime("Driven through the covers for four.")
    chunks = list(stub.stream("sys", "user"))
    assert "".join(chunks) == "Driven through the covers for four."
    assert len(chunks) > 1  # actually streamed, not one blob


def test_stub_runtime_callable_varies_by_prompt() -> None:
    stub = StubRuntime(lambda system, user: "SIX!" if "SIX" in user else "dot.")
    assert "".join(stub.stream("s", "BALL: SIX off the bat")) == "SIX!"
    assert "".join(stub.stream("s", "BALL: dot ball")) == "dot."


def test_stub_runtime_satisfies_protocol() -> None:
    assert isinstance(StubRuntime("x"), RuntimeAdapter)
