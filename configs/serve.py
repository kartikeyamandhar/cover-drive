"""Phase 6 serving configuration: the model artifact, the runtime, and the demo replay.

A plain typed config (not settings); the HF token, if ever needed for a private artifact,
is read separately via ``configs.settings``. The defaults target local dev on the Mac (no
CUDA): the CANONICAL fp16 base plus the LoRA adapter. The adapter's recorded base is
``unsloth/...-bnb-4bit``, but bitsandbytes 4-bit is CUDA-only and that repo ships an
incomplete tokenizer config, so locally we load the canonical base in fp16 on MPS/CPU and
apply the adapter (the LoRA fits the same architecture on any device). A CUDA host can flip
``load_in_4bit_on_cuda`` on.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from configs.train import DEFAULT_BASE_MODEL, QWEN_EOS_TOKEN, QWEN_PAD_TOKEN

DEFAULT_ADAPTER_REPO = "kattymandy/cricket-commentary-qwen2.5-1.5b-lora"


class ServeConfig(BaseModel):
    """Serving and demo-replay configuration."""

    model_config = ConfigDict(frozen=True)

    # Model artifact. The adapter repo also carries the trained tokenizer.
    base_model: str = DEFAULT_BASE_MODEL
    adapter_repo: str = DEFAULT_ADAPTER_REPO
    tokenizer_id: str = DEFAULT_ADAPTER_REPO
    eos_token: str = QWEN_EOS_TOKEN
    pad_token: str = QWEN_PAD_TOKEN

    # Runtime / device. "auto" resolves to cuda if available, else mps, else cpu.
    device: str = "auto"
    load_in_4bit_on_cuda: bool = True

    # Decoding (one short line of commentary).
    max_new_tokens: int = 80
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    seed: int | None = None  # set for a reproducible demo

    # Faithfulness enforcement: regenerate at most this many times before the
    # deterministic fallback. The fallback always terminates, so output is always faithful.
    faithfulness_retries: int = 2

    # Demo replay.
    processed_dir: Path = Path("data/processed")
    demo_match_ids: tuple[str, ...] = ()  # explicit allow-list; empty => discover from dir
    max_listed: int = 8  # cap the discovered listing (ignored when demo_match_ids is set)
    pacing_seconds: float = 1.8  # default delay between balls (the client can override via ?pace)
    max_pacing_seconds: float = 8.0  # clamp on the client-supplied pace

    # API.
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)
