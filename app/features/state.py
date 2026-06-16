"""Per-ball match-state featurization.

Pure functions over typed models: fold an innings' deliveries into the incremental
state a commentator uses, emitting one ``BallContext`` per delivery. No IO. The
model may later reference only what is computed here, so the cricket rules are
explicit and individually tested.

Key rules (Phase 1 red-team):
- Legal balls exclude wides and no-balls; over/ball math uses legal balls.
- Bowler conceded runs = batter runs + wides + no-balls (byes, leg-byes, and
  penalty are charged to nobody).
- Required rate uses the chasing innings ``target`` (DLS-adjusted) when present.
- Super-over innings are excluded from the main spine.
- Stats key on stable registry ids, never on display names.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field

from app.cricsheet.models import Delivery, Innings, Match, MatchInfo
from app.cricsheet.parse import resolve_person_id
from configs.data import FeatureConfig

# Dismissal kinds credited to the bowler. Run outs, retirements, obstruction, etc.
# are dismissals but not the bowler's wicket.
BOWLER_WICKET_KINDS: frozenset[str] = frozenset(
    {"bowled", "caught", "lbw", "stumped", "caught and bowled", "hit wicket"}
)

# "Wicket" entries that are not a fall of wicket: the batter leaves but is not out
# (and may return). These end the partnership but do not increment the score's
# wicket count. "retired out" IS a dismissal and is not listed here.
NON_DISMISSAL_KINDS: frozenset[str] = frozenset({"retired hurt", "retired not out"})


@dataclass
class _BatterStat:
    runs: int = 0
    balls: int = 0


@dataclass
class _BowlerStat:
    runs: int = 0
    balls: int = 0
    wickets: int = 0


@dataclass(frozen=True)
class WicketEvent:
    """A dismissal on a delivery, with bowler-credit resolved."""

    player_out_id: str
    player_out_name: str
    kind: str
    credited_to_bowler: bool


@dataclass(frozen=True)
class BallContext:
    """All facts and computed features for one delivery (state after the ball)."""

    match_id: str
    innings_index: int
    batting_team: str
    bowling_team: str
    over: int  # 0-based over number
    ball_in_over: int  # 1-based position within the over's deliveries
    legal_ball_in_over: int  # legal balls bowled so far this over
    legal_ball_number: int  # legal balls bowled so far this innings
    is_legal: bool

    striker_id: str
    striker_name: str
    non_striker_id: str
    non_striker_name: str
    bowler_id: str
    bowler_name: str

    runs_batter: int
    runs_extras: int
    runs_total: int
    extras: dict[str, int]  # only non-zero types
    wickets: tuple[WicketEvent, ...]

    score_runs: int
    score_wickets: int
    balls_left: int
    current_run_rate: float
    required_run_rate: float | None
    target_runs: int | None
    runs_required: int | None

    striker_runs: int
    striker_balls: int
    bowler_runs: int
    bowler_balls: int
    bowler_wickets: int
    bowler_on_hat_trick: bool

    partnership_runs: int
    partnership_balls: int
    last_deliveries: tuple[str, ...]
    phase: str
    milestones: tuple[str, ...]


def is_legal_delivery(delivery: Delivery) -> bool:
    """A delivery counts as a legal ball unless it is a wide or a no-ball."""
    if delivery.extras is None:
        return True
    return delivery.extras.wides == 0 and delivery.extras.noballs == 0


def bowler_conceded(delivery: Delivery) -> int:
    """Runs charged to the bowler: off the bat plus wides and no-balls only."""
    conceded = delivery.runs.batter
    if delivery.extras is not None:
        conceded += delivery.extras.wides + delivery.extras.noballs
    return conceded


def extras_breakdown(delivery: Delivery) -> dict[str, int]:
    """Return the non-zero extra types on a delivery."""
    if delivery.extras is None:
        return {}
    raw = {
        "wides": delivery.extras.wides,
        "byes": delivery.extras.byes,
        "legbyes": delivery.extras.legbyes,
        "noballs": delivery.extras.noballs,
        "penalty": delivery.extras.penalty,
    }
    return {name: value for name, value in raw.items() if value}


def delivery_token(delivery: Delivery) -> str:
    """A compact token for the last-N-deliveries trail."""
    if delivery.wickets:
        return "W"
    return str(delivery.runs.total)


def phase_of(
    over_zero_based: int, total_overs: int, powerplay_overs: int, death_overs_from_end: int
) -> str:
    """Classify an over as powerplay, middle, or death.

    ``powerplay_overs`` is the count of opening powerplay overs for this innings
    (derived from the source ``powerplays`` when present, so DLS-reduced innings
    are handled), not a global constant.
    """
    if over_zero_based < powerplay_overs:
        return "powerplay"
    if over_zero_based >= total_overs - death_overs_from_end:
        return "death"
    return "middle"


def powerplay_over_count(innings: Innings, features: FeatureConfig) -> int:
    """Number of opening powerplay overs for an innings.

    Uses the source ``powerplays`` (a span like 0.1 to 5.6 in over.ball form) when
    present, so a rain-reduced powerplay is honored; otherwise the configured
    default. Only spans starting at the innings' beginning are considered, so a
    later (non-opening) powerplay does not inflate the count.
    """
    opening_tos = [pp.to for pp in innings.powerplays if pp.from_ <= 1.0]
    if opening_tos:
        return int(max(opening_tos)) + 1  # to is 0-indexed over.ball; +1 for count
    return features.powerplay_overs


def compute_milestones(
    striker_runs: int,
    team_runs: int,
    on_hat_trick: bool,
    features: FeatureConfig,
) -> tuple[str, ...]:
    """Approaching-milestone flags (state after the ball). Explicit thresholds."""
    out: list[str] = []
    window = features.batter_milestone_window
    for mark in (50, 100, 150):
        if mark - window <= striker_runs < mark:
            out.append(f"batter_nearing_{mark}")
            break
    step = features.team_milestone_step
    if team_runs > 0:
        nxt = ((team_runs // step) + 1) * step
        if nxt - team_runs <= features.team_milestone_window:
            out.append(f"team_nearing_{nxt}")
    if on_hat_trick:
        out.append("bowler_on_hat_trick")
    return tuple(out)


@dataclass
class _InningsAccumulator:
    """Mutable fold state for one innings."""

    score_runs: int = 0
    score_wickets: int = 0
    legal_balls: int = 0
    partnership_runs: int = 0
    partnership_balls: int = 0
    batters: dict[str, _BatterStat] = field(default_factory=dict)
    bowlers: dict[str, _BowlerStat] = field(default_factory=dict)
    bowler_streak: dict[str, int] = field(default_factory=dict)


def _innings_max_balls(info: MatchInfo, target_overs: float | None, bpo: int) -> int:
    """Balls allotted to this innings: the (possibly reduced) chase, else the cap."""
    overs = target_overs if target_overs is not None else (info.overs or 20)
    return round(overs * bpo)


def iter_ball_contexts(
    match: Match, match_id: str, features: FeatureConfig
) -> Iterator[BallContext]:
    """Yield a ``BallContext`` for every delivery in every main (non-super) innings."""
    info = match.info
    bpo = info.balls_per_over or features.balls_per_over_default
    main_innings = [inn for inn in match.innings if not inn.super_over]

    for innings_index, innings in enumerate(main_innings):
        batting_team = innings.team
        bowling_team = next((t for t in info.teams if t != batting_team), "")
        target = innings.target
        target_overs = target.overs if target is not None else None
        innings_max_balls = _innings_max_balls(info, target_overs, bpo)
        total_overs = innings_max_balls // bpo if bpo else 0

        pp_overs = powerplay_over_count(innings, features)
        acc = _InningsAccumulator()
        last: deque[str] = deque(maxlen=features.last_n_deliveries)

        for over in innings.overs:
            legal_in_over = 0
            for ball_in_over, delivery in enumerate(over.deliveries, start=1):
                legal = is_legal_delivery(delivery)
                striker_id = resolve_person_id(info, delivery.batter)
                non_striker_id = resolve_person_id(info, delivery.non_striker)
                bowler_id = resolve_person_id(info, delivery.bowler)

                on_hat_trick = acc.bowler_streak.get(bowler_id, 0) >= 2

                acc.score_runs += delivery.runs.total
                acc.partnership_runs += delivery.runs.total
                if legal:
                    acc.legal_balls += 1
                    legal_in_over += 1
                    acc.partnership_balls += 1

                batter = acc.batters.setdefault(striker_id, _BatterStat())
                batter.runs += delivery.runs.batter
                if legal:
                    batter.balls += 1

                bowler = acc.bowlers.setdefault(bowler_id, _BowlerStat())
                bowler.runs += bowler_conceded(delivery)
                if legal:
                    bowler.balls += 1

                wicket_events: list[WicketEvent] = []
                credited_this_ball = False
                for wicket in delivery.wickets:
                    credited = wicket.kind in BOWLER_WICKET_KINDS
                    if credited:
                        bowler.wickets += 1
                        credited_this_ball = True
                    # Retired hurt / not out is not a fall of wicket (the batter
                    # may return); it still ends the partnership below.
                    if wicket.kind not in NON_DISMISSAL_KINDS:
                        acc.score_wickets += 1
                    wicket_events.append(
                        WicketEvent(
                            player_out_id=resolve_person_id(info, wicket.player_out),
                            player_out_name=wicket.player_out,
                            kind=wicket.kind,
                            credited_to_bowler=credited,
                        )
                    )
                acc.bowler_streak[bowler_id] = (
                    acc.bowler_streak.get(bowler_id, 0) + 1 if credited_this_ball else 0
                )

                last.append(delivery_token(delivery))

                crr = acc.score_runs * bpo / acc.legal_balls if acc.legal_balls else 0.0
                balls_left = max(innings_max_balls - acc.legal_balls, 0)
                runs_required: int | None
                target_runs: int | None
                rrr: float | None
                if target is not None:
                    required = max(target.runs - acc.score_runs, 0)
                    runs_required = required
                    target_runs = target.runs
                    rrr = required * bpo / balls_left if balls_left else 0.0
                else:
                    runs_required = None
                    target_runs = None
                    rrr = None

                phase = phase_of(over.over, total_overs, pp_overs, features.death_overs_from_end)
                milestones = compute_milestones(batter.runs, acc.score_runs, on_hat_trick, features)

                yield BallContext(
                    match_id=match_id,
                    innings_index=innings_index,
                    batting_team=batting_team,
                    bowling_team=bowling_team,
                    over=over.over,
                    ball_in_over=ball_in_over,
                    legal_ball_in_over=legal_in_over,
                    legal_ball_number=acc.legal_balls,
                    is_legal=legal,
                    striker_id=striker_id,
                    striker_name=delivery.batter,
                    non_striker_id=non_striker_id,
                    non_striker_name=delivery.non_striker,
                    bowler_id=bowler_id,
                    bowler_name=delivery.bowler,
                    runs_batter=delivery.runs.batter,
                    runs_extras=delivery.runs.extras,
                    runs_total=delivery.runs.total,
                    extras=extras_breakdown(delivery),
                    wickets=tuple(wicket_events),
                    score_runs=acc.score_runs,
                    score_wickets=acc.score_wickets,
                    balls_left=balls_left,
                    current_run_rate=crr,
                    required_run_rate=rrr,
                    target_runs=target_runs,
                    runs_required=runs_required,
                    striker_runs=batter.runs,
                    striker_balls=batter.balls,
                    bowler_runs=bowler.runs,
                    bowler_balls=bowler.balls,
                    bowler_wickets=bowler.wickets,
                    bowler_on_hat_trick=on_hat_trick,
                    partnership_runs=acc.partnership_runs,
                    partnership_balls=acc.partnership_balls,
                    last_deliveries=tuple(last),
                    phase=phase,
                    milestones=milestones,
                )

                # A dismissal ends the current partnership; the next ball starts a
                # fresh one. Report-then-reset so this ball reflects the stand that
                # just ended.
                if delivery.wickets:
                    acc.partnership_runs = 0
                    acc.partnership_balls = 0
