"""Build the per-ball dataset: parse, featurize, serialize, write JSONL.

Orchestrates the Cricsheet parser and the featurizer over a selection of matches,
streaming one match at a time (memory is O(one match)) and writing one JSONL file
per match keyed by ``(match_id, ball_id)``. Idempotent and resumable via the
manifest. This is library code with IO at the edges; CLI wiring lives in
``scripts/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog

from app.cricsheet.acquire import sha256_file
from app.cricsheet.errors import CricsheetError, CricsheetParseError
from app.cricsheet.manifest import Manifest
from app.cricsheet.parse import match_id_from_path, parse_match
from app.features.serialize import record_to_jsonl_line, to_record
from app.features.state import iter_ball_contexts
from configs.data import DataConfig, FeatureConfig

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class BuildStats:
    """Summary of a build run."""

    matches_built: int
    matches_skipped: int
    records_written: int
    matches_failed: int = 0


def build_match(match_path: Path, processed_dir: Path, features: FeatureConfig) -> tuple[int, int]:
    """Build one match into ``{match_id}.jsonl``; return (records, innings)."""
    match = parse_match(match_path)
    match_id = match_id_from_path(match_path)
    lines: list[str] = []
    innings_seen: set[int] = set()
    for ctx in iter_ball_contexts(match, match_id, features):
        lines.append(record_to_jsonl_line(to_record(match.info, ctx)))
        innings_seen.add(ctx.innings_index)
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"{match_id}.jsonl"
    body = "\n".join(lines)
    out_path.write_text(body + "\n" if body else "", encoding="utf-8")
    return len(lines), len(innings_seen)


def _season_of(path: Path) -> str | None:
    try:
        match = parse_match(path)
    except CricsheetParseError:
        return None
    return None if match.info.season is None else str(match.info.season)


def select_matches(files: list[Path], config: DataConfig) -> list[Path]:
    """Apply the configured match selection (ids, season, limit) to file paths."""
    selected = list(files)
    if config.match_ids:
        wanted = set(config.match_ids)
        selected = [p for p in selected if match_id_from_path(p) in wanted]
    if config.season is not None:
        selected = [p for p in selected if _season_of(p) == config.season]
    if config.limit is not None:
        selected = selected[: config.limit]
    return selected


def build_dataset(files: list[Path], config: DataConfig, *, force: bool = False) -> BuildStats:
    """Build all selected matches, skipping unchanged ones via the manifest."""
    manifest = Manifest.load(config.manifest_path)
    selected = select_matches(files, config)
    built = skipped = records = failed = 0
    for path in selected:
        match_id = match_id_from_path(path)
        input_sha = sha256_file(path)
        out_path = config.processed_dir / f"{match_id}.jsonl"
        # Skip only when the input is unchanged AND the output is still present, so
        # a deleted JSONL is regenerated even though the manifest says "built".
        if not force and not manifest.needs_build(match_id, input_sha) and out_path.exists():
            skipped += 1
            continue
        # Isolate per-match failures: one unparseable match must not abort a
        # full-archive build. The bad match is logged and left unbuilt so a later
        # run retries it once the cause is fixed.
        try:
            n_records, n_innings = build_match(path, config.processed_dir, config.features)
        except CricsheetError as exc:
            failed += 1
            log.warning("skipped unbuildable match", match_id=match_id, error=str(exc))
            continue
        manifest.mark_built(match_id, input_sha, n_records, n_innings)
        built += 1
        records += n_records
        log.info("built match", match_id=match_id, records=n_records, innings=n_innings)
    manifest.save(config.manifest_path)
    log.info("build complete", built=built, skipped=skipped, records=records, failed=failed)
    return BuildStats(
        matches_built=built,
        matches_skipped=skipped,
        records_written=records,
        matches_failed=failed,
    )
