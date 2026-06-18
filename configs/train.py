"""Phase 4 training configuration: the student model, QLoRA, optimizer, artifacts.

A plain typed config (not settings); secrets (HF_TOKEN, WANDB_API_KEY) are read
separately via ``configs.settings``. The GPU-side defaults are tuned for the cheapest
pinned 24GB RunPod card: a 4-bit base, a short sequence length (our examples are
~320 tokens), a small per-device batch with gradient accumulation.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

# Decided in Phase 4 (ADR 0007): Qwen2.5-1.5B for budget, training speed, and demo
# latency. 3B is the fallback if the model fails the Phase 5 gate.
# NOTE: we use the CANONICAL repo, not unsloth/...-bnb-4bit. The 4-bit repo ships a
# broken tokenizer config (eos_token = the literal '<EOS_TOKEN>', no pad token), which
# TRL 0.24 rejects. The canonical repo has a complete config (eos = <|im_end|>) and
# Unsloth still quantizes it to 4-bit on load (load_in_4bit=True) -- a ~3GB one-time
# download, cached on the network volume.
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_TOKENIZER = "Qwen/Qwen2.5-1.5B-Instruct"

# Qwen2.5 ChatML markers, used by TRL's train_on_responses_only to mask the loss to
# the assistant turn only (so the model learns to PRODUCE the line, not echo the state).
QWEN_INSTRUCTION_PART = "<|im_start|>user\n"
QWEN_RESPONSE_PART = "<|im_start|>assistant\n"

# Special tokens. The 4-bit repo's special-token config is incomplete, so pin these:
# the chat template ends each turn with <|im_end|> (the real eos); <|endoftext|> is a
# valid base-vocab token used for padding. Both are in vocab (no embedding resize).
QWEN_EOS_TOKEN = "<|im_end|>"
QWEN_PAD_TOKEN = "<|endoftext|>"


class TrainConfig(BaseModel):
    """QLoRA fine-tune configuration for Qwen2.5."""

    model_config = ConfigDict(frozen=True)

    base_model: str = DEFAULT_BASE_MODEL
    tokenizer_id: str = DEFAULT_TOKENIZER
    dataset_repo: str = "kattymandy/cricket-commentary-sft"
    # Measured on the real set: max rendered length is 350 tokens (p95 327), so 512
    # truncates nothing and halves the sequence memory/compute vs 1024.
    max_seq_len: int = 512

    # LoRA: train a small low-rank diff, not the full weights.
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # Optimizer / schedule.
    epochs: float = 2.0
    learning_rate: float = 2e-4
    per_device_batch: int = 8
    grad_accum: int = 2
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    lr_scheduler: str = "linear"
    seed: int = 17

    # Hold out a deterministic slice of TRAIN matches for loss monitoring (no synthetic
    # val set yet; the real held-out eval is the Phase 5 gray corpus). Match-level so a
    # match never straddles train/eval.
    eval_match_fraction: float = 0.05

    # Artifacts.
    data_dir: Path = Path("data")
    output_dir: Path = Path("data/train")  # local on the pod
    adapter_repo: str = "kattymandy/cricket-commentary-qwen2.5-1.5b-lora"
    gguf_repo: str = "kattymandy/cricket-commentary-qwen2.5-1.5b-gguf"
    # Off by default: the GGUF export builds llama.cpp on the pod and can fail at the END
    # of a paid run on a thin image. The adapter is the irreplaceable artifact; build the
    # GGUF off-pod (or enable this only after setup.sh installs build-essential + cmake).
    push_gguf: bool = False
    gguf_quant: str = "q4_k_m"

    @property
    def local_train_jsonl(self) -> Path:
        """The assembled train split on disk (used by the no-GPU --check path)."""
        return self.data_dir / "dataset" / "train.jsonl"

    @property
    def effective_batch(self) -> int:
        """Tokens-per-optimizer-step batch (per-device x accumulation)."""
        return self.per_device_batch * self.grad_accum
