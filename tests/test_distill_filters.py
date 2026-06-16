"""Tests for the quality filters: faithfulness, diversity, seed-dedup."""

from __future__ import annotations

from app.distill.filters import diversity_metrics, faithfulness_check, seed_overlap
from app.distill.seed import EXEMPLARS

_STATE = (
    "T20 IPL | Inns 2 | 18.3 | CSK 162/4 | need 13 off 9 | CRR 8.76 RRR 8.67 | "
    "Striker R Patel 41(24) | Bowler M Khan 1/34 | P'ship 47(26) | death"
)


def test_faithful_line_passes() -> None:
    result = faithfulness_check(
        "Patel launches it into the night - CSK 162/4, need 13 off 9.", "SIX off the bat", _STATE
    )
    assert result.ok


def test_six_called_a_four_is_caught() -> None:
    assert not faithfulness_check("Driven away for four!", "SIX off the bat", _STATE).ok


def test_phantom_boundary_is_caught() -> None:
    assert not faithfulness_check("Smashed for six!", "dot ball", _STATE).ok


def test_phantom_wicket_is_caught() -> None:
    result = faithfulness_check("Bowled him, the stumps go flying!", "dot ball", _STATE)
    assert not result.ok
    assert any("wicket" in r for r in result.reasons)


def test_wrong_score_and_equation_caught() -> None:
    result = faithfulness_check("CSK 99/9, need 5 off 3.", "dot ball", _STATE)
    assert not result.ok
    assert len(result.reasons) == 2


def test_wickets_count_is_not_a_boundary() -> None:
    # "six down" means six wickets, not a six off the bat
    assert faithfulness_check("Hyderabad are six down and reeling.", "dot ball", _STATE).ok


def test_batter_score_off_y_is_not_the_equation() -> None:
    # "27 off 24" is the batter's score; the real equation "83 off 55" is correct
    state = "T20 IPL | Mumbai Indians 56/2 | need 83 off 55 | middle"
    line = "Sharma moves to 27 off 24, but Mumbai still need 83 off 55."
    assert faithfulness_check(line, "FOUR off the bat", state).ok


def test_inside_six_overs_is_not_a_boundary() -> None:
    state = "T20 IPL | Sunrisers Hyderabad 37/2 | need 118 off 87 | powerplay"
    line = "Bowled him! Three down inside six and the chase is under pressure."
    assert faithfulness_check(line, "WICKET, bowled", state).ok


def test_for_four_wickets_is_not_a_boundary() -> None:
    state = "T20 IPL | Punjab Kings 55/4 | need 138 off 66 | middle"
    line = "Caught! Punjab in tatters at 55 for four, needing 138 off 66."
    assert faithfulness_check(line, "WICKET, caught", state).ok


def test_off_by_one_equation_is_caught() -> None:
    state = "T20 IPL | Pune Warriors 126/9 | need 48 off 8 | death"
    assert not faithfulness_check("They need 48 off 7 now.", "1 run", state).ok


def test_all_seed_lines_are_faithful() -> None:
    for exemplar in EXEMPLARS:
        for line in exemplar.lines.values():
            assert faithfulness_check(line, exemplar.event, exemplar.state).ok


def test_diversity_metrics() -> None:
    result = diversity_metrics(["the ball is gone", "the ball is gone", "a brand new shot"])
    assert result.n_lines == 3
    assert result.duplicate_rate > 0  # two identical lines
    assert 0 < result.distinct_1 <= 1


def test_diversity_empty() -> None:
    assert diversity_metrics([]).n_lines == 0


def test_seed_overlap_flags_near_copy() -> None:
    seeds = [exemplar.lines["broadcast"] for exemplar in EXEMPLARS]
    near_copy = EXEMPLARS[0].lines["broadcast"]
    assert seed_overlap(near_copy, seeds)
    assert not seed_overlap("a completely original and different sentence entirely", seeds)
