"""End-to-end: acquire from a local archive, build, and verify resumability."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from app.cricsheet.acquire import acquire
from app.features.build import build_dataset, build_match, select_matches
from configs.data import DataConfig

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample_match.json"


def _config(tmp_path: Path) -> DataConfig:
    return DataConfig(data_dir=tmp_path / "data")


def test_build_match_writes_jsonl(tmp_path: Path) -> None:
    config = _config(tmp_path)
    n_records, n_innings = build_match(SAMPLE, config.processed_dir, config.features)
    assert n_records == 11
    assert n_innings == 2
    out = config.processed_dir / "sample_match.jsonl"
    assert out.exists()
    assert len(out.read_text(encoding="utf-8").strip().splitlines()) == 11


def test_build_dataset_idempotent_and_force(tmp_path: Path) -> None:
    config = _config(tmp_path)
    files = [SAMPLE]

    first = build_dataset(files, config)
    assert first.matches_built == 1
    assert first.records_written == 11

    second = build_dataset(files, config)
    assert second.matches_built == 0
    assert second.matches_skipped == 1

    forced = build_dataset(files, config, force=True)
    assert forced.matches_built == 1


def test_partial_rebuild_only_touches_changed_match(tmp_path: Path) -> None:
    config = _config(tmp_path)
    m1 = tmp_path / "m1.json"
    m2 = tmp_path / "m2.json"
    shutil.copy(SAMPLE, m1)
    shutil.copy(SAMPLE, m2)
    files = [m1, m2]

    first = build_dataset(files, config)
    assert first.matches_built == 2

    # Change m2's bytes (trailing whitespace keeps it valid JSON); m1 unchanged.
    m2.write_text(m2.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    second = build_dataset(files, config)
    assert second.matches_built == 1
    assert second.matches_skipped == 1


def test_selection_limit(tmp_path: Path) -> None:
    config = DataConfig(data_dir=tmp_path / "data", limit=0)
    stats = build_dataset([SAMPLE], config)
    assert stats.matches_built == 0


def test_select_matches_by_id_and_season(tmp_path: Path) -> None:
    m1 = tmp_path / "m1.json"
    m2 = tmp_path / "m2.json"
    shutil.copy(SAMPLE, m1)
    shutil.copy(SAMPLE, m2)
    files = [m1, m2]

    by_id = DataConfig(data_dir=tmp_path / "d", match_ids=("m1",))
    assert select_matches(files, by_id) == [m1]

    # the fixture's season is "2023"
    matching_season = DataConfig(data_dir=tmp_path / "d", season="2023")
    assert select_matches(files, matching_season) == files

    other_season = DataConfig(data_dir=tmp_path / "d", season="1999")
    assert select_matches(files, other_season) == []


def test_acquire_then_build(tmp_path: Path) -> None:
    archive = tmp_path / "ipl.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(SAMPLE, arcname="sample_match.json")

    config = DataConfig(data_dir=tmp_path / "data")
    result = acquire(
        dest_dir=config.raw_dir,
        local_archive=archive,
        allowed_hosts=config.allowed_hosts,
        max_download_bytes=config.max_download_bytes,
        max_uncompressed_bytes=config.max_uncompressed_bytes,
        max_entries=config.max_entries,
    )
    stats = build_dataset(result.match_files, config)
    assert stats.records_written == 11
