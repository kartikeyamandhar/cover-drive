"""CLI entrypoint: acquire Cricsheet IPL data.

Idempotent and resumable. Sources, in priority order: a local directory of match
JSON (``--from-dir``), a local archive (``--from-local``), or a download from the
configured URL. Safe to re-run.

    uv run python -m scripts.acquire                       # download IPL archive
    uv run python -m scripts.acquire --from-local data.zip # use a local archive
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.cricsheet.acquire import AcquireResult, acquire
from app.logging_config import configure_logging
from configs.data import DataConfig


def run(args: argparse.Namespace) -> AcquireResult:
    """Execute acquisition from parsed args and return the result."""
    config = DataConfig() if args.data_dir is None else DataConfig(data_dir=args.data_dir)
    local_archive: Path | None = args.from_local
    local_dir: Path | None = args.from_dir
    url = args.url if args.url is not None else config.archive_url
    return acquire(
        dest_dir=config.raw_dir,
        url=None if (local_archive or local_dir) else url,
        local_archive=local_archive,
        local_dir=local_dir,
        allowed_hosts=config.allowed_hosts,
        max_download_bytes=config.max_download_bytes,
        max_uncompressed_bytes=config.max_uncompressed_bytes,
        max_entries=config.max_entries,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire Cricsheet IPL data.")
    parser.add_argument("--data-dir", type=Path, default=None, help="root data directory")
    parser.add_argument("--url", type=str, default=None, help="override the archive URL")
    parser.add_argument("--from-local", type=Path, default=None, help="local archive (.zip)")
    parser.add_argument("--from-dir", type=Path, default=None, help="local dir of match JSON")
    parser.add_argument("--json-logs", action="store_true", help="emit JSON logs")
    args = parser.parse_args()

    configure_logging(json_logs=args.json_logs)
    result = run(args)
    print(f"acquired {len(result.match_files)} match files (cache={result.from_cache})")


if __name__ == "__main__":
    main()
