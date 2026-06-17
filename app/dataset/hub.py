"""Push the assembled dataset to a private Hugging Face Hub dataset.

The Hub is the artifact store and the Mac-to-RunPod transfer path. The push is
gated on ``HF_TOKEN`` and never runs in ``make check``. Only the Claude-authored
synthetic train/val/test are pushed; the gray-license real eval reference is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from configs.distill import DistillConfig

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class PushPlan:
    """What a push would upload (for inspection / dry-run)."""

    repo_id: str
    files: list[str]
    private: bool


def plan_push(config: DistillConfig, repo_id: str) -> PushPlan:
    """Describe the push without performing it (no network)."""
    files = [p.name for p in sorted(config.dataset_dir.glob("*.jsonl"))]
    return PushPlan(repo_id=repo_id, files=files, private=True)


def push(config: DistillConfig, repo_id: str, token: str) -> PushPlan:
    """Create the private dataset repo and upload the split JSONL files (live)."""
    from huggingface_hub import HfApi

    plan = plan_push(config, repo_id)
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    for path in sorted(config.dataset_dir.glob("*.jsonl")):
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            repo_type="dataset",
        )
    log.info("pushed dataset", repo_id=repo_id, files=len(plan.files))
    return plan
