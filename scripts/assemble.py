"""CLI: assemble the SFT dataset from the distill pairs; optionally push to the Hub.

uv run python -m scripts.assemble --dry-run                 # count splits, write nothing
uv run python -m scripts.assemble                           # write train/val/test + stats
uv run python -m scripts.assemble --push user/cricket-sft   # assemble then push (live)
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from app.dataset.assemble import assemble, read_pairs, split_of
from app.dataset.hub import push
from app.logging_config import configure_logging
from configs.distill import DistillConfig
from configs.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the SFT dataset.")
    parser.add_argument("--data-dir", type=Path, default=None, help="root data directory")
    parser.add_argument("--dry-run", action="store_true", help="count splits, write nothing")
    parser.add_argument(
        "--push", type=str, default=None, metavar="REPO_ID", help="push to a private HF dataset"
    )
    parser.add_argument("--json-logs", action="store_true", help="emit JSON logs")
    args = parser.parse_args()

    configure_logging(json_logs=args.json_logs)
    config = DistillConfig() if args.data_dir is None else DistillConfig(data_dir=args.data_dir)

    if args.dry_run:
        counts = Counter(split_of(config, pair["match_id"]) for pair in read_pairs(config))
        summary = " ".join(f"{split}={n}" for split, n in sorted(counts.items()))
        print(f"would assemble: {summary} (total {sum(counts.values())})")
        return

    stats = assemble(config)
    splits = " ".join(f"{split}={n}" for split, n in sorted(stats.by_split.items()))
    print(
        f"assembled total={stats.total} {splits} | "
        f"assistant words mean={stats.assistant_words_mean} p95={stats.assistant_words_p95}"
    )

    if args.push is not None:
        settings = get_settings()
        if settings.hf_token is None:
            raise SystemExit("HF_TOKEN is not set; cannot push")
        plan = push(config, args.push, settings.hf_token.get_secret_value())
        print(f"pushed {len(plan.files)} files to {plan.repo_id}")


if __name__ == "__main__":
    main()
