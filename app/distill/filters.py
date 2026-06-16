"""Post-generation quality filters: faithfulness, diversity, seed-dedup.

Faithfulness is the load-bearing one and foreshadows the Phase 5 checker: a
generated line may carry descriptive color, but if it states or contradicts a
HARD fact (a boundary/wicket outcome, the score, the chase equation) it is a
defect and is dropped. The check runs against the factual ``event`` and the
``state_string`` the line was generated from, so it needs no extra data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_SCORE_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
_NEED_RE = re.compile(r"(\d+)\s+(?:runs?\s+)?(?:needed|required|to\s+win\s+)?(?:off|from)\s+(\d+)")
_WORD_RE = re.compile(r"[a-z']+")

_WICKET_WORDS = frozenset({"bowled", "caught", "lbw", "stumped", "dismissed", "wicket", "castled"})


@dataclass(frozen=True)
class FaithResult:
    """Outcome of a faithfulness check."""

    ok: bool
    reasons: list[str] = field(default_factory=list)


def _score(text: str) -> tuple[int, int] | None:
    match = _SCORE_RE.search(text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _need(text: str) -> tuple[int, int] | None:
    match = _NEED_RE.search(text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _claims_boundary(line_lower: str, word: str) -> bool:
    """True if ``word`` is a boundary claim, not a wicket count ('six down' = wickets)."""
    return bool(re.search(rf"\b{word}\b(?!\s+(?:down|wickets?))", line_lower))


def faithfulness_check(line: str, event: str, state_string: str) -> FaithResult:
    """Flag a line that states or contradicts a hard fact not in event/state."""
    reasons: list[str] = []
    line_lower = line.lower()
    words = set(_WORD_RE.findall(line_lower))

    event_six = event.startswith("SIX")
    event_four = event.startswith("FOUR")
    event_wicket = event.startswith("WICKET")

    claims_six = bool(re.search(r"\b(?:maximum|sixer)\b", line_lower)) or _claims_boundary(
        line_lower, "six"
    )
    claims_four = _claims_boundary(line_lower, "four") or "boundary" in words

    if event_six and claims_four:
        reasons.append("calls a six a four")
    elif event_four and claims_six:
        reasons.append("calls a four a six")
    elif not event_six and not event_four:
        if claims_six:
            reasons.append("claims a six that did not happen")
        if claims_four:
            reasons.append("claims a boundary that did not happen")

    claims_wicket = bool(words & _WICKET_WORDS) or "run out" in line.lower()
    if claims_wicket and not event_wicket:
        reasons.append("claims a wicket that did not happen")

    line_score, state_score = _score(line), _score(state_string)
    if line_score is not None and state_score is not None and line_score != state_score:
        reasons.append(f"score {line_score} != state {state_score}")

    line_need, state_need = _need(line), _need(state_string)
    if line_need is not None and state_need is not None and line_need != state_need:
        reasons.append(f"equation {line_need} != state {state_need}")

    return FaithResult(ok=not reasons, reasons=reasons)


@dataclass(frozen=True)
class DiversityResult:
    """Lexical-diversity summary for a set of generated lines."""

    n_lines: int
    distinct_1: float
    distinct_2: float
    duplicate_rate: float


def _tokens(line: str) -> list[str]:
    return _WORD_RE.findall(line.lower())


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def diversity_metrics(lines: list[str]) -> DiversityResult:
    """distinct-1/2 and the near-duplicate rate over a set of lines."""
    if not lines:
        return DiversityResult(0, 0.0, 0.0, 0.0)
    uni: list[str] = []
    bi: list[tuple[str, ...]] = []
    for line in lines:
        toks = _tokens(line)
        uni.extend(toks)
        bi.extend(_ngrams(toks, 2))
    distinct_1 = len(set(uni)) / len(uni) if uni else 0.0
    distinct_2 = len(set(bi)) / len(bi) if bi else 0.0
    normalized = [" ".join(_tokens(line)) for line in lines]
    duplicate_rate = 1.0 - len(set(normalized)) / len(normalized)
    return DiversityResult(
        n_lines=len(lines),
        distinct_1=round(distinct_1, 4),
        distinct_2=round(distinct_2, 4),
        duplicate_rate=round(duplicate_rate, 4),
    )


def seed_overlap(line: str, seed_lines: list[str], *, n: int = 6) -> bool:
    """True if the line shares an n-gram with any seed line (a near-copy of the seed)."""
    line_grams = set(_ngrams(_tokens(line), n))
    if not line_grams:
        return False
    return any(line_grams & set(_ngrams(_tokens(seed), n)) for seed in seed_lines)
