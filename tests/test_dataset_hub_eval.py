"""Tests for the HF push scaffolding (mocked) and the eval-set loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.dataset.eval_set import load_reference
from app.dataset.hub import plan_push, push
from configs.distill import DistillConfig

FIXTURES = Path(__file__).parent / "fixtures"


def test_plan_push_is_inspectable(tmp_path: Path) -> None:
    config = DistillConfig(data_dir=tmp_path / "data")
    config.dataset_dir.mkdir(parents=True)
    (config.dataset_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (config.dataset_dir / "val.jsonl").write_text("{}\n", encoding="utf-8")
    plan = plan_push(config, "user/cricket-sft")
    assert plan.repo_id == "user/cricket-sft"
    assert plan.private is True
    assert set(plan.files) == {"train.jsonl", "val.jsonl"}


def test_push_uploads_each_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    calls = {"repo": 0, "files": 0}

    class _FakeApi:
        def __init__(self, token: str | None = None) -> None: ...

        def create_repo(self, *args: object, **kwargs: object) -> None:
            calls["repo"] += 1

        def upload_file(self, **kwargs: object) -> None:
            calls["files"] += 1

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    config = DistillConfig(data_dir=tmp_path / "data")
    config.dataset_dir.mkdir(parents=True)
    (config.dataset_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (config.dataset_dir / "test.jsonl").write_text("{}\n", encoding="utf-8")
    push(config, "user/cricket-sft", "tok")
    assert calls["repo"] == 1
    assert calls["files"] == 2


def test_load_eval_reference_jsonl() -> None:
    lines = load_reference(FIXTURES / "eval_reference.jsonl")
    assert len(lines) == 3
    assert all(isinstance(line, str) and line for line in lines)


def test_load_eval_reference_missing_is_empty(tmp_path: Path) -> None:
    assert load_reference(tmp_path / "absent.jsonl") == []


def test_load_eval_reference_bare_lines(tmp_path: Path) -> None:
    path = tmp_path / "ref.txt"
    path.write_text("one line\n\nanother line\n", encoding="utf-8")
    assert load_reference(path) == ["one line", "another line"]
