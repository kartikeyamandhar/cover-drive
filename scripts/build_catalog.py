"""Build the web app's match catalog from the local Cricsheet data.

Scans the parsed raw match info (season, date, playoff stage, teams, winner) and the
processed per-ball JSONL (ball count, innings) for every IPL match, groups by season, and
writes a single factual JSON the frontend bundles to render the season selector and the
per-season bracket. Idempotent: re-run any time the data changes.

  uv run python -m scripts.build_catalog
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

IPL = "Indian Premier League"

# Playoff stage -> bracket round (0 = group). One renderer then covers both eras: the
# classic Semi-Final era and the modern Qualifier/Eliminator era.
STAGE_ROUND: dict[str, int] = {
    "Semi Final": 1,
    "Qualifier 1": 1,
    "Eliminator": 1,
    "Elimination Final": 1,
    "3rd Place Play-Off": 1,
    "Qualifier 2": 2,
    "Final": 3,
}


def _fmt_date(iso: str) -> str:
    try:
        y, m, d = (int(x) for x in iso.split("-"))
        return date(y, m, d).strftime("%-d %b %Y")
    except (ValueError, TypeError):
        return iso


def _ball_stats(path: Path) -> tuple[int, int]:
    """(ball count, innings count) from a processed JSONL file, or (0, 0)."""
    if not path.exists():
        return (0, 0)
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return (0, 0)
    last = json.loads(lines[-1])
    return (len(lines), int(last.get("innings", 0)) + 1)


def build(raw_dir: Path, processed_dir: Path) -> dict[str, Any]:
    """Group every IPL match by season into the catalog structure."""
    by_season: dict[str, list[dict[str, Any]]] = {}
    for raw in sorted(raw_dir.glob("*.json")):
        try:
            info = json.loads(raw.read_text(encoding="utf-8")).get("info", {})
        except (ValueError, OSError):
            continue
        event = info.get("event") or {}
        if event.get("name") != IPL:
            continue
        match_id = raw.stem
        balls, innings = _ball_stats(processed_dir / f"{match_id}.jsonl")
        if balls == 0:
            continue  # only matches we can actually replay
        stage = event.get("stage")
        dates = info.get("dates") or [""]
        outcome = info.get("outcome") or {}
        record = {
            "id": match_id,
            "teams": list(info.get("teams", []))[:2],
            "date": _fmt_date(dates[0]),
            "iso": dates[0],
            "balls": balls,
            "innings": innings,
            "stage": stage if stage else "league",
            "round": STAGE_ROUND.get(stage, 1) if stage else 0,
            "matchNo": event.get("match_number"),
            "winner": outcome.get("winner", ""),
        }
        by_season.setdefault(str(info.get("season")), []).append(record)

    seasons: list[dict[str, Any]] = []
    for season in sorted(by_season):
        matches = sorted(by_season[season], key=lambda m: (m["round"], m["matchNo"] or 0, m["iso"]))
        final = next((m for m in matches if m["stage"] == "Final"), None)
        seasons.append(
            {
                "season": season,
                "label": f"IPL {season}",
                "champion": final["winner"] if final else "",
                "matches": matches,
            }
        )
    return {"competition": IPL, "seasons": seasons}


def main() -> None:
    parser = argparse.ArgumentParser(description="build the web match catalog")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/extracted"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("web/public/catalog.json"))
    args = parser.parse_args()

    catalog = build(args.raw_dir, args.processed_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(catalog, separators=(",", ":")), encoding="utf-8")
    n = sum(len(s["matches"]) for s in catalog["seasons"])
    print(f"wrote {args.out}: {len(catalog['seasons'])} seasons, {n} matches")


if __name__ == "__main__":
    main()
