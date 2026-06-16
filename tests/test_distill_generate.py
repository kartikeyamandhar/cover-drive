"""Tests for generation orchestration. The Anthropic SDK is mocked; no live calls."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import anthropic

from app.distill.generate import WorkItem, dry_run, plan_items, run_batch, run_sync
from app.distill.sampling import SampleCandidate
from app.features.build import build_match
from configs.data import FeatureConfig
from configs.distill import DistillConfig
from configs.personas import primary_persona

FIXTURE = Path(__file__).parent / "fixtures" / "sample_match.json"
_FAITHFUL = "Patel launches it into the night! CSK 162/4, need 13 off 9."


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


class _FakeBatches:
    def __init__(self, text: str) -> None:
        self.text = text
        self._requests: list[object] = []

    def create(self, *, requests: list[object]) -> SimpleNamespace:
        self._requests = requests
        return SimpleNamespace(id="batch_1")

    def retrieve(self, _batch_id: str) -> SimpleNamespace:
        return SimpleNamespace(processing_status="ended")

    def results(self, _batch_id: str) -> list[SimpleNamespace]:
        out = []
        for req in self._requests:
            cid = cast(dict[str, object], req)["custom_id"]
            out.append(
                SimpleNamespace(
                    custom_id=cid,
                    result=SimpleNamespace(type="succeeded", message=_fake_message(self.text)),
                )
            )
        return out


class _FakeMessages:
    def __init__(self, text: str) -> None:
        self.text = text
        self.batches = _FakeBatches(text)

    def create(self, **_kwargs: object) -> SimpleNamespace:
        return _fake_message(self.text)


class _FakeClient:
    def __init__(self, text: str = _FAITHFUL) -> None:
        self.messages = _FakeMessages(text)


def _client(text: str = _FAITHFUL) -> anthropic.Anthropic:
    return cast(anthropic.Anthropic, _FakeClient(text))


def _item(ball: str = "0-0-1") -> WorkItem:
    candidate = SampleCandidate(
        match_id="m",
        ball_id=ball,
        bucket="six",
        event="SIX off the bat",
        state="T20 IPL | CSK 162/4 | need 13 off 9 | death",
    )
    return WorkItem(candidate, primary_persona())


def test_run_sync_keeps_faithful_pair(tmp_path: Path) -> None:
    config = DistillConfig(data_dir=tmp_path / "data")
    stats = run_sync(_client(), config, [_item()])
    assert stats.kept == 1
    assert stats.processed == 1
    out = config.distill_dir / "broadcast.jsonl"
    assert out.exists()
    assert "commentary" in out.read_text(encoding="utf-8")


def test_run_sync_rejects_unfaithful(tmp_path: Path) -> None:
    config = DistillConfig(data_dir=tmp_path / "data")
    stats = run_sync(_client("Driven for four, what a shot!"), config, [_item()])
    assert stats.kept == 0
    assert stats.rejected_faithfulness == 1


def test_run_sync_is_idempotent(tmp_path: Path) -> None:
    config = DistillConfig(data_dir=tmp_path / "data")
    run_sync(_client(), config, [_item()])
    again = run_sync(_client(), config, [_item()])
    assert again.processed == 0  # manifest skips the completed item


def test_budget_cap_stops_the_run(tmp_path: Path) -> None:
    config = DistillConfig(data_dir=tmp_path / "data")
    items = [_item(f"0-0-{i}") for i in range(6)]
    # each fake generation costs ~$0.00036; cap stops after the 2nd
    stats = run_sync(_client(), config, items, budget_cap=0.0005)
    assert stats.processed == 2


def test_run_batch_collects_pairs(tmp_path: Path) -> None:
    config = DistillConfig(data_dir=tmp_path / "data")
    stats = run_batch(_client(), config, [_item(), _item("0-0-2")])
    assert stats.kept == 2


def test_run_batch_budget_guard_stops_between_chunks(tmp_path: Path) -> None:
    config = DistillConfig(data_dir=tmp_path / "data")
    items = [_item(f"0-0-{i}") for i in range(6)]
    # one item per chunk, a tiny cap: the guard stops partway, not all 6
    stats = run_batch(_client(), config, items, budget_cap=0.0005, chunk_size=1)
    assert 0 < stats.processed < 6


def test_dry_run_estimates_without_api(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    build_match(FIXTURE, processed, FeatureConfig())
    config = DistillConfig(
        data_dir=tmp_path / "data", val_fraction=0.0, test_fraction=0.0, primary_set_size=50
    )
    n_items, batch_usd, sync_usd = dry_run(config)
    assert n_items > 0
    assert 0 < batch_usd < sync_usd  # batch is cheaper than sync


def test_plan_items_includes_secondaries(tmp_path: Path) -> None:
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    build_match(FIXTURE, processed, FeatureConfig())
    config = DistillConfig(
        data_dir=tmp_path / "data", val_fraction=0.0, test_fraction=0.0, primary_set_size=50
    )
    items = plan_items(config)
    personas = {item.persona.key for item in items}
    assert "broadcast" in personas
    assert len(personas) >= 2  # primary + at least one secondary
