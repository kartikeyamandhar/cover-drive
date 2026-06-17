"""Tests for assembly: split-by-match routing, determinism, stats."""

from __future__ import annotations

import json
from pathlib import Path

from app.dataset.assemble import assemble, split_of
from configs.distill import DistillConfig


def _write_pairs(config: DistillConfig, pairs: list[dict[str, str]]) -> None:
    config.distill_dir.mkdir(parents=True, exist_ok=True)
    by_persona: dict[str, list[dict[str, str]]] = {}
    for pair in pairs:
        by_persona.setdefault(pair["persona"], []).append(pair)
    for persona, items in by_persona.items():
        (config.distill_dir / f"{persona}.jsonl").write_text(
            "\n".join(json.dumps(p) for p in items) + "\n", encoding="utf-8"
        )


def _pair(match_id: str, ball_id: str, persona: str, bucket: str = "six") -> dict[str, str]:
    return {
        "match_id": match_id,
        "ball_id": ball_id,
        "persona": persona,
        "bucket": bucket,
        "event": "SIX off the bat",
        "state": f"T20 IPL | {match_id} 162/4 | death",
        "commentary": "Smashed into the stands, what a hit.",
    }


def test_assemble_routes_and_co_locates_by_match(tmp_path: Path) -> None:
    config = DistillConfig(data_dir=tmp_path / "data")
    pairs = [
        _pair("matchA", "0-0-1", "broadcast"),
        _pair("matchA", "0-0-2", "radio"),  # same match, different persona
        _pair("matchB", "0-0-1", "broadcast"),
        _pair("matchC", "0-0-1", "broadcast"),
    ]
    _write_pairs(config, pairs)
    stats = assemble(config)
    assert stats.total == 4

    # both matchA pairs land in the same split file (a match never spans splits)
    split_a = split_of(config, "matchA")
    lines = (config.dataset_dir / f"{split_a}.jsonl").read_text(encoding="utf-8").splitlines()
    assert sum(1 for line in lines if "matchA" in line) == 2

    # every split file's examples actually belong to that split
    for split_name in ("train", "val", "test"):
        path = config.dataset_dir / f"{split_name}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            example = json.loads(line)
            assert split_of(config, example["match_id"]) == split_name


def test_assemble_is_deterministic(tmp_path: Path) -> None:
    config = DistillConfig(data_dir=tmp_path / "data")
    _write_pairs(config, [_pair("matchA", "0-0-2", "broadcast"), _pair("matchA", "0-0-1", "radio")])
    assemble(config)
    first = (config.dataset_dir / f"{split_of(config, 'matchA')}.jsonl").read_text(encoding="utf-8")
    assemble(config)
    second = (config.dataset_dir / f"{split_of(config, 'matchA')}.jsonl").read_text(
        encoding="utf-8"
    )
    assert first == second


def test_assemble_ignores_judge_dropped_log(tmp_path: Path) -> None:
    # the judge-filter writes judge_dropped.jsonl into data/distill; assemble must NOT
    # read it back, or the very defects the filter removed re-enter the training set.
    config = DistillConfig(data_dir=tmp_path / "data")
    _write_pairs(config, [_pair("matchA", "0-0-1", "broadcast")])
    (config.distill_dir / "judge_dropped.jsonl").write_text(
        json.dumps(_pair("matchA", "0-0-9", "broadcast")) + "\n", encoding="utf-8"
    )
    stats = assemble(config)
    assert stats.total == 1  # only the kept pair, not the dropped one
    split = split_of(config, "matchA")
    balls = {
        json.loads(line)["ball_id"]
        for line in (config.dataset_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert balls == {"0-0-1"}


def test_stats_file_written(tmp_path: Path) -> None:
    config = DistillConfig(data_dir=tmp_path / "data")
    _write_pairs(config, [_pair("matchA", "0-0-1", "broadcast")])
    stats = assemble(config)
    assert stats.total == 1
    assert stats.by_persona["broadcast"] == 1
    saved = json.loads((config.dataset_dir / "stats.json").read_text(encoding="utf-8"))
    assert saved["total"] == 1
