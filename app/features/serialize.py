"""Deterministic serialization of a ``BallContext``.

Produces two artifacts per ball:
- a compact human-readable ``state_string`` the teacher and student consume, and
- a versioned structured ``record`` that the Phase 6 fact-fill and Phase 5
  faithfulness check verify against.

Determinism is a hard requirement (caching, diffing, and the golden test depend on
it): floats are rounded to a fixed precision and JSON is dumped with sorted keys.
"""

from __future__ import annotations

import json

from app.cricsheet.models import MatchInfo
from app.features.state import BallContext

BALL_RECORD_SCHEMA_VERSION = 1


def ball_id(ctx: BallContext) -> str:
    """Stable per-ball key: ``innings-over-ballInOver`` (illegal balls get ids too)."""
    return f"{ctx.innings_index}-{ctx.over}-{ctx.ball_in_over}"


def _round2(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


def over_label(ctx: BallContext) -> str:
    """Render the over.ball label, marking wides/no-balls.

    Uses scorecard notation: overs completed plus the legal ball within the
    current over, with the over 0-indexed (the first over's balls are 0.1..0.6,
    the last ball of a T20 is 19.6).
    """
    base = f"{ctx.over}.{ctx.legal_ball_in_over}"
    if ctx.is_legal:
        return base
    if "wides" in ctx.extras:
        return f"{base}+wide"
    if "noballs" in ctx.extras:
        return f"{base}+no-ball"
    return f"{base}+extra"


def _competition_label(info: MatchInfo) -> str:
    parts: list[str] = []
    if info.match_type:
        parts.append(info.match_type)
    if info.event is not None and info.event.name:
        parts.append(info.event.name)
    elif info.season is not None:
        parts.append(str(info.season))
    return " ".join(parts) if parts else "match"


def state_string(info: MatchInfo, ctx: BallContext) -> str:
    """A compact, deterministic one-line match state for the model to narrate."""
    segments: list[str] = [
        _competition_label(info),
        f"Inns {ctx.innings_index + 1}",
        over_label(ctx),
        f"{ctx.batting_team} {ctx.score_runs}/{ctx.score_wickets}",
    ]
    if ctx.runs_required is not None and ctx.target_runs is not None:
        segments.append(f"need {ctx.runs_required} off {ctx.balls_left}")
    crr = _round2(ctx.current_run_rate)
    rate = f"CRR {crr}"
    if ctx.required_run_rate is not None:
        rate += f" RRR {_round2(ctx.required_run_rate)}"
    segments.append(rate)
    segments.append(f"Striker {ctx.striker_name} {ctx.striker_runs}({ctx.striker_balls})")
    segments.append(f"Bowler {ctx.bowler_name} {ctx.bowler_wickets}/{ctx.bowler_runs}")
    segments.append(f"P'ship {ctx.partnership_runs}({ctx.partnership_balls})")
    if ctx.last_deliveries:
        segments.append("Last " + " ".join(ctx.last_deliveries))
    segments.append(ctx.phase)
    if ctx.milestones:
        segments.append("nearing: " + ",".join(ctx.milestones))
    return " | ".join(segments)


def to_record(info: MatchInfo, ctx: BallContext) -> dict[str, object]:
    """Build the versioned, JSON-serializable per-ball record."""
    wickets = [
        {
            "player_out_id": w.player_out_id,
            "player_out_name": w.player_out_name,
            "kind": w.kind,
            "credited_to_bowler": w.credited_to_bowler,
        }
        for w in ctx.wickets
    ]
    return {
        "schema_version": BALL_RECORD_SCHEMA_VERSION,
        "match_id": ctx.match_id,
        "ball_id": ball_id(ctx),
        "innings": ctx.innings_index,
        "over": ctx.over,
        "ball_in_over": ctx.ball_in_over,
        "legal_ball_number": ctx.legal_ball_number,
        "is_legal": ctx.is_legal,
        "batting_team": ctx.batting_team,
        "bowling_team": ctx.bowling_team,
        "striker": {"id": ctx.striker_id, "name": ctx.striker_name},
        "non_striker": {"id": ctx.non_striker_id, "name": ctx.non_striker_name},
        "bowler": {"id": ctx.bowler_id, "name": ctx.bowler_name},
        "delivery": {
            "runs_batter": ctx.runs_batter,
            "runs_extras": ctx.runs_extras,
            "runs_total": ctx.runs_total,
            "extras": dict(ctx.extras),
            "wickets": wickets,
        },
        "state": {
            "score_runs": ctx.score_runs,
            "score_wickets": ctx.score_wickets,
            "balls_left": ctx.balls_left,
            "current_run_rate": _round2(ctx.current_run_rate),
            "required_run_rate": _round2(ctx.required_run_rate),
            "target_runs": ctx.target_runs,
            "runs_required": ctx.runs_required,
            "striker_runs": ctx.striker_runs,
            "striker_balls": ctx.striker_balls,
            "bowler_runs": ctx.bowler_runs,
            "bowler_balls": ctx.bowler_balls,
            "bowler_wickets": ctx.bowler_wickets,
            "bowler_on_hat_trick": ctx.bowler_on_hat_trick,
            "partnership_runs": ctx.partnership_runs,
            "partnership_balls": ctx.partnership_balls,
            "last_deliveries": list(ctx.last_deliveries),
            "phase": ctx.phase,
            "milestones": list(ctx.milestones),
        },
        "state_string": state_string(info, ctx),
    }


def record_to_jsonl_line(record: dict[str, object]) -> str:
    """Serialize a record to one deterministic JSON line."""
    return json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
