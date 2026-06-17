"""CLI: generate the synthetic SFT set from the Claude teacher.

uv run python -m scripts.distill --dry-run            # cost estimate, no API calls
uv run python -m scripts.distill --pilot 400          # opt-in, live (spends credit)
uv run python -m scripts.distill --full --mode batch  # full run via Batches (50% off)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import anthropic

from app.distill.errors import DistillError
from app.distill.generate import RunStats, dry_run, plan_items, run_batch, run_sync
from app.logging_config import configure_logging
from configs.distill import DistillConfig
from configs.settings import get_settings


def _make_client() -> anthropic.Anthropic:
    settings = get_settings()
    if settings.anthropic_api_key is None:
        raise DistillError("ANTHROPIC_API_KEY is not set; add it to .env before a live run")
    return anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())


def _config(args: argparse.Namespace) -> DistillConfig:
    overrides: dict[str, object] = {}
    if args.data_dir is not None:
        overrides["data_dir"] = args.data_dir
    if args.model is not None:
        overrides["teacher_model"] = args.model
    if args.budget is not None:
        overrides["budget_usd_cap"] = args.budget
    if args.primary_set_size is not None:
        overrides["primary_set_size"] = args.primary_set_size
    return DistillConfig().model_copy(update=overrides)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill the synthetic commentary SFT set.")
    parser.add_argument("--dry-run", action="store_true", help="estimate cost, no API calls")
    parser.add_argument("--pilot", type=int, default=None, help="generate N sampled items (live)")
    parser.add_argument("--full", action="store_true", help="generate the full planned set (live)")
    parser.add_argument("--mode", choices=("sync", "batch"), default="sync", help="run mode")
    parser.add_argument("--data-dir", type=Path, default=None, help="root data directory")
    parser.add_argument("--model", type=str, default=None, help="override the teacher model id")
    parser.add_argument("--budget", type=float, default=None, help="override the USD budget cap")
    parser.add_argument(
        "--primary-set-size", type=int, default=None, help="override primary persona set size"
    )
    parser.add_argument("--json-logs", action="store_true", help="emit JSON logs")
    args = parser.parse_args()

    configure_logging(json_logs=args.json_logs)
    config = _config(args)

    if args.dry_run:
        n_items, batch_usd, sync_usd = dry_run(config)
        print(
            f"planned items={n_items} | model={config.teacher_model} | "
            f"est batch ~${batch_usd} | est sync ~${sync_usd} | "
            f"budget cap ${config.budget_usd_cap} (hard stop)"
        )
        return

    if args.pilot is None and not args.full:
        parser.error("choose one of --dry-run, --pilot N, or --full")

    client = _make_client()
    items = plan_items(config)
    if args.pilot is not None:
        rng = random.Random(config.sampling_seed)
        items = rng.sample(items, min(args.pilot, len(items)))

    stats: RunStats = (
        run_batch(client, config, items)
        if args.mode == "batch"
        else run_sync(client, config, items)
    )
    print(
        f"kept={stats.kept} processed={stats.processed} "
        f"rejected_faithfulness={stats.rejected_faithfulness} "
        f"rejected_seed_dedup={stats.rejected_seed_dedup} failed={stats.failed} "
        f"spend=${round(stats.spend_usd, 4)} cache_read_tokens={stats.cache_read_tokens}"
    )


if __name__ == "__main__":
    main()
