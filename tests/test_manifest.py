"""Manifest: build-state tracking, deterministic save, resumability logic."""

from __future__ import annotations

from pathlib import Path

from app.cricsheet.manifest import Manifest


def test_needs_build_and_mark() -> None:
    manifest = Manifest()
    assert manifest.needs_build("m", "sha1") is True
    manifest.mark_built("m", "sha1", n_records=10, n_innings=2)
    assert manifest.needs_build("m", "sha1") is False
    assert manifest.needs_build("m", "sha2") is True  # input changed


def test_save_load_roundtrip_and_sorted(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = Manifest()
    manifest.mark_built("b", "shab", n_records=5, n_innings=1)
    manifest.mark_built("a", "shaa", n_records=7, n_innings=2)
    manifest.save(path)

    loaded = Manifest.load(path)
    assert loaded.matches["a"].n_records == 7
    assert loaded.matches["b"].n_innings == 1

    text = path.read_text(encoding="utf-8")
    assert text.index('"a"') < text.index('"b"')  # deterministic, sorted


def test_load_missing_returns_empty(tmp_path: Path) -> None:
    assert Manifest.load(tmp_path / "absent.json").matches == {}


def test_load_corrupt_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{ not valid", encoding="utf-8")
    assert Manifest.load(path).matches == {}
