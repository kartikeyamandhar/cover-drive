"""Phase 5 comparison: score generated commentary for faithfulness + diversity.

The generation itself happens on a GPU (the fine-tuned student, the base model, and
optionally the teacher); this module is the GPU-free scoring layer that turns those
outputs into the gate's numbers. It reuses the Phase 2 heuristic faithfulness checker
and the diversity metrics, so the same definition of "faithful" applies end to end.

The gate (CLAUDE.md / phase-5): the fine-tuned model must beat the base model on
faithfulness and style by the margin in this phase file before any serving work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.distill.filters import DiversityResult, diversity_metrics, faithfulness_check


@dataclass(frozen=True)
class SystemScore:
    """Faithfulness + diversity for one system's outputs over the eval set."""

    name: str
    n: int
    faithful: int
    faithfulness_rate: float
    diversity: DiversityResult
    defects: list[tuple[str, list[str]]]  # (line, reasons) for the unfaithful lines


def score_system(name: str, records: list[dict[str, Any]]) -> SystemScore:
    """Score one system. ``records`` are ``{event, state, line}`` dicts (the generated
    line plus the ground-truth event/state it was generated from)."""
    faithful = 0
    defects: list[tuple[str, list[str]]] = []
    lines: list[str] = []
    for rec in records:
        line = rec["line"]
        lines.append(line)
        result = faithfulness_check(line, rec["event"], rec["state"])
        if result.ok:
            faithful += 1
        else:
            defects.append((line, result.reasons))
    n = len(records)
    return SystemScore(
        name=name,
        n=n,
        faithful=faithful,
        faithfulness_rate=faithful / n if n else 0.0,
        diversity=diversity_metrics(lines),
        defects=defects,
    )


def comparison_table(scores: list[SystemScore]) -> list[dict[str, Any]]:
    """A flat table (base vs fine-tuned vs teacher) of the headline gate metrics."""
    return [
        {
            "system": s.name,
            "n": s.n,
            "faithfulness": round(s.faithfulness_rate, 4),
            "distinct_2": s.diversity.distinct_2,
            "self_dup_rate": s.diversity.duplicate_rate,
        }
        for s in scores
    ]


def gate_passed(
    finetuned: SystemScore, base: SystemScore, *, faithfulness_margin: float = 0.05
) -> tuple[bool, str]:
    """The training gate: the fine-tune must beat base on faithfulness by a margin and
    not collapse (diversity must stay healthy). Returns (passed, reason)."""
    delta = finetuned.faithfulness_rate - base.faithfulness_rate
    if delta < faithfulness_margin:
        return False, (
            f"faithfulness gain {delta:+.1%} below the {faithfulness_margin:.0%} margin "
            f"(fine-tuned {finetuned.faithfulness_rate:.1%} vs base {base.faithfulness_rate:.1%})"
        )
    if finetuned.diversity.distinct_2 < 0.5:
        return False, f"diversity collapse: distinct-2 {finetuned.diversity.distinct_2:.2f} < 0.50"
    if finetuned.diversity.duplicate_rate > 0.1:
        return False, f"repetition: self-dup {finetuned.diversity.duplicate_rate:.2f} > 0.10"
    return True, (
        f"PASS: faithfulness {finetuned.faithfulness_rate:.1%} vs base "
        f"{base.faithfulness_rate:.1%} ({delta:+.1%}); distinct-2 "
        f"{finetuned.diversity.distinct_2:.2f}"
    )
