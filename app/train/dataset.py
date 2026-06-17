"""Load the SFT dataset and render it through the Qwen chat template for training.

This module is the data path, and it is deliberately GPU-free: it runs on the Mac
(transformers tokenizer only, no torch/unsloth) so the most error-prone part of
Phase 4 -- the chat-template rendering and the assistant-span the loss is masked to --
is validated BEFORE any GPU is rented. ``train.py`` reuses it on the pod.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from configs.train import QWEN_RESPONSE_PART, TrainConfig

log = structlog.get_logger(__name__)


def load_examples(config: TrainConfig, *, source: str = "local") -> list[dict[str, Any]]:
    """Load SFT examples: ``local`` reads the assembled train.jsonl; ``hub`` pulls the
    private dataset (the pod path)."""
    if source == "local":
        path = config.local_train_jsonl
        text = path.read_text(encoding="utf-8")
    else:
        from huggingface_hub import hf_hub_download

        from configs.settings import get_settings

        settings = get_settings()
        token = settings.hf_token.get_secret_value() if settings.hf_token else None
        fp = hf_hub_download(
            repo_id=config.dataset_repo,
            filename="train.jsonl",
            repo_type="dataset",
            token=token,
        )
        text = Path(fp).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _is_eval_match(match_id: str, fraction: float) -> bool:
    """Deterministic match-level holdout for loss monitoring (a match never straddles)."""
    h = int(hashlib.sha256(f"evalholdout-{match_id}".encode()).hexdigest(), 16) % 1000
    return h < fraction * 1000


def split_for_monitoring(
    examples: list[dict[str, Any]], fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split into (train, eval) by match id for in-training loss monitoring."""
    train = [e for e in examples if not _is_eval_match(e["match_id"], fraction)]
    held = [e for e in examples if _is_eval_match(e["match_id"], fraction)]
    return train, held


def load_tokenizer(config: TrainConfig) -> Any:
    """The Qwen tokenizer (chat template). Loads on the Mac without torch/GPU."""
    from transformers import AutoTokenizer

    auto: Any = AutoTokenizer  # transformers is untyped; treat as Any under strict mypy
    return auto.from_pretrained(config.tokenizer_id)


def render(example: dict[str, Any], tokenizer: Any) -> str:
    """Apply the Qwen chat template to one example's messages (no tokenization)."""
    text: str = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return text


def trained_span(rendered: str) -> str:
    """The substring the loss is actually computed on: the assistant turn after the
    Qwen response marker. If the marker is absent, masking would be wrong -- callers
    must treat an empty return as a hard error."""
    idx = rendered.rfind(QWEN_RESPONSE_PART)
    return "" if idx < 0 else rendered[idx + len(QWEN_RESPONSE_PART) :]


@dataclass(frozen=True)
class CheckSummary:
    """A no-GPU validation of the data path."""

    n_total: int
    n_train: int
    n_eval: int
    token_p50: int
    token_p95: int
    token_max: int
    over_max_seq: int
    marker_ok: bool
    sample_context: str
    sample_trained: str


def check(
    config: TrainConfig, *, tokenizer: Any = None, sample_for_lengths: int = 500
) -> CheckSummary:
    """Render + measure the dataset with no GPU, asserting the loss-mask marker exists.

    ``tokenizer`` is injectable so tests can pass a stub and stay network-free; in
    normal use it loads the real Qwen tokenizer.
    """
    examples = load_examples(config, source="local")
    train, held = split_for_monitoring(examples, config.eval_match_fraction)
    if tokenizer is None:
        tokenizer = load_tokenizer(config)

    step = max(1, len(examples) // sample_for_lengths)
    lengths = sorted(
        len(tokenizer.apply_chat_template(e["messages"], tokenize=True)) for e in examples[::step]
    )
    p50 = lengths[len(lengths) // 2]
    p95 = lengths[int(0.95 * len(lengths))]
    over = sum(1 for n in lengths if n > config.max_seq_len)

    sample = train[0]
    rendered = render(sample, tokenizer)
    trained = trained_span(rendered)
    marker_ok = bool(trained)
    context = rendered[: len(rendered) - len(trained)] if marker_ok else rendered

    return CheckSummary(
        n_total=len(examples),
        n_train=len(train),
        n_eval=len(held),
        token_p50=p50,
        token_p95=p95,
        token_max=lengths[-1],
        over_max_seq=over,
        marker_ok=marker_ok,
        sample_context=context,
        sample_trained=trained,
    )
