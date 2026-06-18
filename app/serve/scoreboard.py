"""Deterministic fact-fill: the displayed scoreboard and the faithful fallback line.

Both are built ONLY from the structured ground-truth record, never from the model. The
scoreboard is what the client shows (score, rates, striker/bowler, last six); the fallback
line is the safety net the engine substitutes when a model draw fails the faithfulness
check, so a wrong fact never reaches the user. Every fallback line is itself constructed to
pass ``faithfulness_check`` against the same ground truth (a test asserts this).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from app.serve.ball import ServeBall


class Scoreboard(BaseModel):
    """The per-ball facts the client renders, all sourced from the structured record."""

    model_config = ConfigDict(frozen=True)

    match_id: str
    ball_id: str
    innings: int  # 1-based for display
    over: str  # scorecard notation, e.g. "11.6"
    batting_team: str
    bowling_team: str
    score: str  # "162/4"
    runs: int
    wickets: int
    striker: str
    striker_runs: int
    striker_balls: int
    bowler: str
    bowler_figures: str  # wickets/runs, e.g. "1/24"
    current_run_rate: float | None
    required_run_rate: float | None
    target: int | None
    runs_required: int | None
    balls_left: int
    last_deliveries: list[str]
    phase: str
    event: str  # the factual BALL: phrase


def _over_label(ball: ServeBall) -> str:
    """Scorecard over label "over.legal-ball-in-over" (the over is 0-based)."""
    legal_in_over = ball.legal_ball_number - 6 * ball.over
    return f"{ball.over}.{legal_in_over}"


def scoreboard(ball: ServeBall) -> Scoreboard:
    """Render the deterministic scoreboard for a ball, straight from its record."""
    s = ball.state
    return Scoreboard(
        match_id=ball.match_id,
        ball_id=ball.ball_id,
        innings=ball.innings + 1,
        over=_over_label(ball),
        batting_team=ball.batting_team,
        bowling_team=ball.bowling_team,
        score=f"{s.score_runs}/{s.score_wickets}",
        runs=s.score_runs,
        wickets=s.score_wickets,
        striker=ball.striker.name,
        striker_runs=s.striker_runs,
        striker_balls=s.striker_balls,
        bowler=ball.bowler.name,
        bowler_figures=f"{s.bowler_wickets}/{s.bowler_runs}",
        current_run_rate=s.current_run_rate,
        required_run_rate=s.required_run_rate,
        target=s.target_runs,
        runs_required=s.runs_required,
        balls_left=s.balls_left,
        last_deliveries=list(s.last_deliveries),
        phase=s.phase,
        event=ball.event,
    )


def fallback_line(ball: ServeBall) -> str:
    """A deterministic, always-faithful commentary line for a ball.

    Used when a model draw fails the faithfulness check (or as a no-model mode). States only
    facts present in the ground truth and is phrased to pass ``faithfulness_check``: the team
    score is the only "X/Y" in the line, no chase equation, and no "Nth wicket" claim.
    """
    s = ball.state
    score = f"{ball.batting_team} {s.score_runs}/{s.score_wickets}"
    striker = ball.striker.name
    event = ball.event

    if event.startswith("WICKET"):
        kind = event.split(", ", 1)[1] if ", " in event else "out"
        out = ball.delivery.wickets[0].player_out_name if ball.delivery.wickets else striker
        return f"Wicket. {out or striker} {kind}. {score}."
    if event == "SIX off the bat":
        return f"{striker} clears the rope for six. {score}."
    if event == "FOUR off the bat":
        return f"{striker} finds the boundary for four. {score}."
    if event in ("wide", "no-ball"):
        return f"{event.capitalize()}, an extra to the total. {score}."
    if event == "dot ball":
        return f"Beaten, no run there. {score}."
    total = ball.delivery.runs_total
    if total == 1:
        return f"{striker} pushes a single. {score}."
    return f"{striker} works it for {total} runs. {score}."
