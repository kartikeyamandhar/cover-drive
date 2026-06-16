"""Tests for the distillation manifest."""

from __future__ import annotations

from pathlib import Path

from app.distill.manifest import DistillManifest, item_key


def test_item_key() -> None:
    assert item_key("123", "0-1-2", "broadcast") == "123|0-1-2|broadcast"


def test_mark_and_is_done() -> None:
    manifest = DistillManifest()
    key = item_key("m", "0-0-1", "radio")
    assert not manifest.is_done(key)
    manifest.mark(key, "kept")
    assert manifest.is_done(key)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "distill" / "manifest.json"
    manifest = DistillManifest()
    manifest.mark(item_key("m", "0-0-1", "text"), "rejected_faithfulness")
    manifest.save(path)
    loaded = DistillManifest.load(path)
    assert loaded.items[item_key("m", "0-0-1", "text")].status == "rejected_faithfulness"


def test_load_missing_is_empty(tmp_path: Path) -> None:
    assert DistillManifest.load(tmp_path / "nope.json").items == {}
