"""CLI entrypoint: build the per-ball match-state dataset.

Reads extracted match JSON, featurizes each ball, and writes one JSONL file per
match keyed by ``(match_id, ball_id)``. Idempotent and resumable via the manifest:
unchanged matches are skipped; ``--force`` rebuilds all.

    uv run python -m scripts.build_dataset                 # build from data/raw/extracted
    uv run python -m scripts.build_dataset --season 2023 --limit 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.features.build import BuildStats, build_dataset
from app.logging_config import configure_logging
from configs.data import DataConfig


def run(args: argparse.Namespace) -> BuildStats:
    """Execute the build from parsed args and return the run stats."""
    data_dir: Path = args.data_dir if args.data_dir is not None else Path("data")
    config = DataConfig(
        data_dir=data_dir,
        match_ids=tuple(args.match_id or ()),
        season=args.season,
        limit=args.limit,
    )
    source_dir: Path = args.from_dir if args.from_dir is not None else config.raw_dir / "extracted"
    files = sorted(source_dir.rglob("*.json"))
    return build_dataset(files, config, force=args.force)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the per-ball dataset.")
    parser.add_argument("--data-dir", type=Path, default=None, help="root data directory")
    parser.add_argument("--from-dir", type=Path, default=None, help="dir of extracted match JSON")
    parser.add_argument("--season", type=str, default=None, help="only this season")
    parser.add_argument("--match-id", action="append", default=None, help="only these match ids")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of matches")
    parser.add_argument("--force", action="store_true", help="rebuild even if unchanged")
    parser.add_argument("--json-logs", action="store_true", help="emit JSON logs")
    args = parser.parse_args()

    configure_logging(json_logs=args.json_logs)
    stats = run(args)
    print(
        f"built={stats.matches_built} skipped={stats.matches_skipped} "
        f"failed={stats.matches_failed} records={stats.records_written}"
    )


if __name__ == "__main__":
    main()
