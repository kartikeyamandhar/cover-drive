"""CLI: Phase 4 QLoRA fine-tune.

uv run python -m scripts.train --check                 # no-GPU data-path validation (Mac)
uv run python -m scripts.train --train                 # pod: run the QLoRA fine-tune (GPU)
uv run python -m scripts.train --check --base-model Qwen/Qwen2.5-3B-Instruct --epochs 3
"""

from __future__ import annotations

import argparse
import textwrap

from app.logging_config import configure_logging
from app.train.train import run_check, run_train
from configs.train import TrainConfig


def _config(args: argparse.Namespace) -> TrainConfig:
    overrides: dict[str, object] = {}
    if args.base_model is not None:
        overrides["base_model"] = args.base_model
    if args.tokenizer is not None:
        overrides["tokenizer_id"] = args.tokenizer
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    return TrainConfig().model_copy(update=overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Qwen2.5 on the SFT set.")
    parser.add_argument("--check", action="store_true", help="validate the data path (no GPU)")
    parser.add_argument("--train", action="store_true", help="run the fine-tune (GPU, on the pod)")
    parser.add_argument("--base-model", type=str, default=None, help="override base model id")
    parser.add_argument("--tokenizer", type=str, default=None, help="override tokenizer id")
    parser.add_argument("--epochs", type=float, default=None, help="override epochs")
    parser.add_argument("--json-logs", action="store_true", help="emit JSON logs")
    args = parser.parse_args()

    configure_logging(json_logs=args.json_logs)
    config = _config(args)

    if args.check:
        s = run_check(config)
        print("=== DATA-PATH CHECK (no GPU) ===")
        print(f"examples: {s.n_total}  (train {s.n_train} / eval-holdout {s.n_eval})")
        print(
            f"token length: p50={s.token_p50} p95={s.token_p95} max={s.token_max} "
            f"| over max_seq({config.max_seq_len})={s.over_max_seq}"
        )
        print(f"loss-mask marker present: {s.marker_ok}  (assistant turn is the training target)")
        print("\n--- ONE EXAMPLE, rendered through the Qwen chat template ---")
        print("[CONTEXT - loss MASKED]")
        print(textwrap.indent(s.sample_context, "  "))
        print("[ASSISTANT - loss COMPUTED HERE (what the model learns to produce)]")
        print(textwrap.indent(s.sample_trained, "  "))
        if not s.marker_ok:
            raise SystemExit("FATAL: Qwen response marker not found; loss mask would be wrong.")
        return

    if args.train:
        run_train(config)
        return

    parser.error("choose --check or --train")


if __name__ == "__main__":
    main()
