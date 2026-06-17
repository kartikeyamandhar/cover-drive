"""CLI: second-stage Opus judge-filter over the distilled pairs (boundary pipeline).

uv run python -m scripts.judge_filter --dry-run                 # count high-risk + cost
uv run python -m scripts.judge_filter --mode sync               # judge live (spends)
uv run python -m scripts.judge_filter --mode batch              # full run via Batches (50% off)

Runs AFTER scripts.distill (generation + heuristic) and BEFORE scripts.assemble.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anthropic

from app.distill.errors import DistillError
from app.distill.judge_filter import (
    HIGH_RISK_BUCKETS,
    JudgeFilterStats,
    count_high_risk_pairs,
    estimate_cost,
    judge_filter_batch,
    judge_filter_sync,
)
from app.logging_config import configure_logging
from configs.distill import DistillConfig
from configs.settings import get_settings


def _make_client() -> anthropic.Anthropic:
    settings = get_settings()
    if settings.anthropic_api_key is None:
        raise DistillError("ANTHROPIC_API_KEY is not set; add it to .env before a live run")
    return anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())


def main() -> None:
    parser = argparse.ArgumentParser(description="Opus judge-filter over distilled pairs.")
    parser.add_argument("--dry-run", action="store_true", help="count high-risk pairs + cost")
    parser.add_argument("--mode", choices=("sync", "batch"), default=None, help="run mode (live)")
    parser.add_argument("--model", type=str, default="claude-opus-4-8", help="judge model id")
    parser.add_argument("--workers", type=int, default=6, help="sync concurrency")
    parser.add_argument("--budget", type=float, default=25.0, help="USD budget guard")
    parser.add_argument(
        "--chunk-size", type=int, default=500, help="batch chunk size (larger = fewer, faster)"
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="root data directory")
    parser.add_argument("--json-logs", action="store_true", help="emit JSON logs")
    args = parser.parse_args()

    configure_logging(json_logs=args.json_logs)
    config = DistillConfig() if args.data_dir is None else DistillConfig(data_dir=args.data_dir)

    if args.dry_run:
        n = count_high_risk_pairs(config)
        print(
            f"high-risk pairs in {config.distill_dir}: {n} "
            f"(buckets: {sorted(HIGH_RISK_BUCKETS)})\n"
            f"est judge cost: batch ~${estimate_cost(n, model=args.model, batch=True)} | "
            f"sync ~${estimate_cost(n, model=args.model, batch=False)}"
        )
        return

    if args.mode is None:
        parser.error("choose --dry-run or --mode {sync,batch}")

    client = _make_client()
    stats: JudgeFilterStats = (
        judge_filter_batch(
            client, config, model=args.model, budget_usd=args.budget, chunk_size=args.chunk_size
        )
        if args.mode == "batch"
        else judge_filter_sync(
            client, config, model=args.model, max_workers=args.workers, budget_usd=args.budget
        )
    )
    print(
        f"high_risk={stats.high_risk} judged={stats.judged} kept={stats.kept} "
        f"dropped={stats.dropped} judge_failed={stats.judge_failed} "
        f"spend=${round(stats.spend_usd, 4)}"
    )


if __name__ == "__main__":
    main()
