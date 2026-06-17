"""Tests for the second-stage Opus judge-filter. The SDK is mocked; no live calls."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import anthropic

from app.distill.judge_filter import (
    HIGH_RISK_BUCKETS,
    is_high_risk,
    judge_filter_sync,
)
from configs.distill import DistillConfig

_DROP = "DROPME"


def _verdict(line: str) -> dict[str, object]:
    faithful = _DROP not in line
    return {
        "faithful": faithful,
        "severity": "none" if faithful else "major",
        "failure_modes": [] if faithful else ["chase_equation_error"],
        "persona_match": True,
        "confidence": "high",
        "explanation": "ok" if faithful else "off-by-one equation",
    }


class _Messages:
    def create(self, **kwargs: object) -> SimpleNamespace:
        messages = cast(list[dict[str, str]], kwargs["messages"])
        line = ""
        for piece in messages[0]["content"].splitlines():
            if piece.startswith("LINE: "):
                line = piece[len("LINE: ") :]
        return SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", name="record_verdict", input=_verdict(line))],
            usage=SimpleNamespace(
                input_tokens=900,
                output_tokens=80,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
        )


class _Client:
    def __init__(self) -> None:
        self.messages = _Messages()


def _client() -> anthropic.Anthropic:
    return cast(anthropic.Anthropic, _Client())


def _pair(persona: str, bucket: str, commentary: str, ball: str) -> dict[str, str]:
    return {
        "match_id": "m1",
        "ball_id": ball,
        "persona": persona,
        "bucket": bucket,
        "event": "1 run",
        "state": "T20 | Inns 2 | CSK 134/6 | need 3 off 3 | death",
        "commentary": commentary,
    }


def _write(config: DistillConfig, pairs: list[dict[str, str]]) -> None:
    config.distill_dir.mkdir(parents=True, exist_ok=True)
    by_persona: dict[str, list[dict[str, str]]] = {}
    for pair in pairs:
        by_persona.setdefault(pair["persona"], []).append(pair)
    for persona, items in by_persona.items():
        (config.distill_dir / f"{persona}.jsonl").write_text(
            "\n".join(json.dumps(p) for p in items) + "\n", encoding="utf-8"
        )


def test_is_high_risk() -> None:
    assert is_high_risk({"bucket": "wicket"}, HIGH_RISK_BUCKETS)
    assert not is_high_risk({"bucket": "middle_routine"}, HIGH_RISK_BUCKETS)


def test_judge_filter_drops_only_unfaithful_high_risk(tmp_path: Path) -> None:
    config = DistillConfig(data_dir=tmp_path / "data")
    pairs = [
        _pair("broadcast", "wicket", "Clean wicket call.", "0-1-1"),  # high-risk, faithful -> keep
        _pair("broadcast", "wicket", f"Bad call {_DROP}.", "0-1-2"),  # high-risk, bad -> drop
        _pair(
            "broadcast", "middle_routine", f"Low risk {_DROP}.", "0-1-3"
        ),  # low-risk -> not judged
    ]
    _write(config, pairs)
    stats = judge_filter_sync(_client(), config, max_workers=3)

    assert stats.pairs == 3
    assert stats.high_risk == 2
    assert stats.judged == 2
    assert stats.dropped == 1
    assert stats.kept == 1

    kept = (config.distill_dir / "broadcast.jsonl").read_text(encoding="utf-8").splitlines()
    balls = {json.loads(line)["ball_id"] for line in kept}
    assert balls == {"0-1-1", "0-1-3"}  # the bad wicket dropped; the low-risk kept unjudged

    dropped = (config.distill_dir / "judge_dropped.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(dropped) == 1
    assert json.loads(dropped[0])["ball_id"] == "0-1-2"


def test_judge_filter_keeps_all_when_clean(tmp_path: Path) -> None:
    config = DistillConfig(data_dir=tmp_path / "data")
    _write(config, [_pair("radio", "six", "A clean maximum description.", "0-2-1")])
    stats = judge_filter_sync(_client(), config)
    assert stats.dropped == 0
    assert stats.kept == 1
    assert (config.distill_dir / "radio.jsonl").read_text(encoding="utf-8").strip() != ""


def test_judge_filter_budget_guard(tmp_path: Path) -> None:
    config = DistillConfig(data_dir=tmp_path / "data")
    _write(config, [_pair("broadcast", "wicket", f"bad {_DROP} {i}", f"0-1-{i}") for i in range(6)])
    stats = judge_filter_sync(_client(), config, max_workers=2, budget_usd=0.0)
    assert stats.judged == 0  # budget tripped before any wave
