"""Tests for the GPU-free training data path. No network, no tokenizer download."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.train.dataset import (
    check,
    load_examples,
    split_for_monitoring,
    trained_span,
)
from configs.train import QWEN_RESPONSE_PART, TrainConfig


def _example(match_id: str, ball: str, line: str = "A line.") -> dict[str, Any]:
    return {
        "match_id": match_id,
        "ball_id": ball,
        "persona": "broadcast",
        "bucket": "wicket",
        "messages": [
            {"role": "system", "content": "You are a commentator..."},
            {"role": "user", "content": "BALL: WICKET, bowled\nSTATE: CSK 45/2"},
            {"role": "assistant", "content": line},
        ],
    }


class _FakeTokenizer:
    """Mimics the bits of a HF tokenizer that the data path uses."""

    def apply_chat_template(self, messages: list[dict[str, str]], *, tokenize: bool = False) -> Any:
        text = ""
        for m in messages:
            text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        text += QWEN_RESPONSE_PART  # the generation prompt the template appends
        # for the assistant turn, fold its content in after the marker
        assistant = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        rendered = text + assistant + "<|im_end|>\n"
        if tokenize:
            return list(range(len(rendered.split())))  # crude token count
        return rendered


def test_trained_span_extracts_after_marker() -> None:
    rendered = f"<|im_start|>user\nX<|im_end|>\n{QWEN_RESPONSE_PART}the line<|im_end|>\n"
    assert trained_span(rendered) == "the line<|im_end|>\n"


def test_trained_span_empty_when_marker_absent() -> None:
    assert trained_span("no marker here") == ""


def test_split_for_monitoring_is_match_level_and_disjoint() -> None:
    examples = [_example(f"m{i}", "0-0-1") for i in range(200)]
    train, held = split_for_monitoring(examples, 0.1)
    assert len(train) + len(held) == 200
    train_matches = {e["match_id"] for e in train}
    held_matches = {e["match_id"] for e in held}
    assert train_matches.isdisjoint(held_matches)  # a match never straddles
    assert 0 < len(held) < len(train)  # roughly the requested fraction, not all/none


def test_split_is_deterministic() -> None:
    examples = [_example(f"m{i}", "0-0-1") for i in range(50)]
    a, _ = split_for_monitoring(examples, 0.1)
    b, _ = split_for_monitoring(examples, 0.1)
    assert [e["match_id"] for e in a] == [e["match_id"] for e in b]


def test_load_examples_local(tmp_path: Path) -> None:
    config = TrainConfig(data_dir=tmp_path / "data")
    config.local_train_jsonl.parent.mkdir(parents=True)
    rows = [_example("m1", "0-0-1"), _example("m1", "0-0-2")]
    config.local_train_jsonl.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    got = load_examples(config, source="local")
    assert len(got) == 2
    assert got[0]["match_id"] == "m1"


def test_check_renders_and_finds_loss_mask(tmp_path: Path) -> None:
    config = TrainConfig(data_dir=tmp_path / "data")
    config.local_train_jsonl.parent.mkdir(parents=True)
    rows = [_example(f"m{i}", "0-0-1", line="Bowled him!") for i in range(20)]
    config.local_train_jsonl.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    summary = check(config, tokenizer=_FakeTokenizer())
    assert summary.n_total == 20
    assert summary.n_train + summary.n_eval == 20
    assert summary.marker_ok is True
    assert "Bowled him!" in summary.sample_trained
    assert "<|im_start|>system" in summary.sample_context
    assert summary.token_max > 0
