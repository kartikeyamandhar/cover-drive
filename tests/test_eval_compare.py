"""Tests for the Phase 5 comparison/gate scoring."""

from __future__ import annotations

from app.eval.compare import comparison_table, gate_passed, score_system

_STATE = "T20 IPL | CSK 162/4 | need 13 off 9 | death"


def _rec(line: str, event: str = "FOUR off the bat") -> dict[str, str]:
    return {"event": event, "state": _STATE, "line": line}


def test_score_system_counts_faithful_and_defects() -> None:
    records = [
        _rec("Driven through the covers for four, CSK 162/4."),  # faithful
        _rec("Smashed for SIX into the crowd!", event="dot ball"),  # phantom six -> defect
    ]
    score = score_system("finetuned", records)
    assert score.n == 2
    assert score.faithful == 1
    assert abs(score.faithfulness_rate - 0.5) < 1e-9
    assert len(score.defects) == 1
    assert "six" in score.defects[0][1][0]


def test_comparison_table_shape() -> None:
    s = score_system("base", [_rec("Four runs.")])
    table = comparison_table([s])
    assert table[0]["system"] == "base"
    assert "faithfulness" in table[0]
    assert "distinct_2" in table[0]


def test_gate_passes_when_finetuned_beats_base_with_diversity() -> None:
    finetuned = score_system(
        "ft",
        [
            _rec("Kohli drives it crisply through cover for four, a lovely shot."),
            _rec("Punched off the back foot, races to the rope, four more for RCB."),
            _rec("Worked away into the gap, and they pick up a boundary at the death."),
        ],
    )
    base = score_system(
        "base",
        [
            _rec("Smashed for SIX!", event="dot ball"),  # defect
            _rec("Bowled him!", event="dot ball"),  # defect
            _rec("A clean strike down the ground for four."),
        ],
    )
    passed, reason = gate_passed(finetuned, base, faithfulness_margin=0.05)
    assert passed, reason


def test_gate_fails_without_faithfulness_margin() -> None:
    same = [_rec("A crisp boundary through the covers for four.")]
    finetuned = score_system("ft", same)
    base = score_system("base", same)
    passed, reason = gate_passed(finetuned, base)
    assert not passed
    assert "margin" in reason


def test_gate_fails_on_diversity_collapse() -> None:
    repeated = [_rec("Four runs.") for _ in range(10)]  # identical lines -> collapse
    finetuned = score_system("ft", repeated)
    base = score_system("base", [_rec("Smashed for six!", event="dot ball")])
    passed, reason = gate_passed(finetuned, base)
    assert not passed
    assert "collapse" in reason or "repetition" in reason
