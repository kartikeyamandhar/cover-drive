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
