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
# A chase equation "X off Y" - but only when a need/require/win/chase word is near
# (so a batter's "27 off 24" score is NOT mistaken for the equation). The connective
# tissue is permissive: "2 needed off the last 2", "5 more from 4 balls", "8 off 7
# now" all parse, because the teacher's off-by-one error hides behind that phrasing.
_EQ_RE = re.compile(
    r"(\d+)\s+(?:(?:runs?|more|still|needed|required|now|to\s+win|to\s+get)\s+)*"
    r"(?:off|from)\s+(?:the\s+)?(?:last\s+|final\s+|remaining\s+)?(\d+)"
)
_EQ_CONTEXT = re.compile(r"need|requir|to\s+win|to\s+get|chase")
_WORD_RE = re.compile(r"[a-z']+")
_NUMBER_WORDS = "one|two|three|four|five|six|seven|eight|nine|ten"
_FIGURES_BEFORE = re.compile(rf"(?:\d+|\b(?:{_NUMBER_WORDS}))\s+for\s+$")

_WICKET_WORDS = frozenset({"bowled", "caught", "lbw", "stumped", "dismissed", "castled"})

# Wicket-count claims, compared against the "/W" in the team score (which already
# includes this ball). Ordinals are unambiguous; the cardinal "N down" form excludes
# a shot played "down the ground/pitch" via the trailing direction lookahead.
_ORDINAL = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}
_CARDINAL = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_NTH_WICKET_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+"
    r"(?:wicket|scalp|to\s+fall)\b"
)
_LOSE_NTH_RE = re.compile(
    r"\blos(?:e|es|t|ing)\s+(?:their|its|his)\s+"
    r"(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b"
)
_DOWN_RE = re.compile(
    rf"\b(\d+|{_NUMBER_WORDS})\s+(?:wickets?\s+)?down\b"
    r"(?!\s+(?:the|to|towards|over|past|into|at|and\s+out))"
)


@dataclass(frozen=True)
class FaithResult:
    """Outcome of a faithfulness check."""

    ok: bool
    reasons: list[str] = field(default_factory=list)


def _score(text: str) -> tuple[int, int] | None:
    match = _SCORE_RE.search(text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _need(text: str) -> tuple[int, int] | None:
    """Extract the chase equation, ignoring batter scores phrased as 'X off Y'."""
    for match in _EQ_RE.finditer(text):
        window = text[max(0, match.start() - 22) : match.end() + 12].lower()
        if _EQ_CONTEXT.search(window):
            return (int(match.group(1)), int(match.group(2)))
    return None


def _claims_boundary(line_lower: str, word: str) -> bool:
    """True if ``word`` ("four"/"six") is a real boundary claim.

    Excludes cricket's other uses of the word: a wicket count ('six down', 'four
    wickets', '55 for four'), an over reference ('inside six', 'six overs'), bowling
    figures ('two for six'), and a score fragment ('162/4 ... four').
    """
    for match in re.finditer(rf"\b{word}\b", line_lower):
        after = line_lower[match.end() : match.end() + 9]
        before = line_lower[max(0, match.start() - 18) : match.start()]
        if re.match(r"\s+(?:down|wickets?|overs?|for\b)", after):
            continue  # 'six down' / 'four wickets' / 'six overs' / 'six for 24'
        if re.search(r"(?:inside|over)\s+$", before):
            continue  # 'inside six' / 'over six'
        if _FIGURES_BEFORE.search(before):
            continue  # '55 for four' / 'two for six'
        if re.search(r"\d\s*/\s*$", before):
            continue  # score 'X/four'
        return True
    return False


def _claimed_wicket_count(line_lower: str) -> int | None:
    """How many wickets the line claims have fallen, or ``None`` if it makes no claim."""
    match = _DOWN_RE.search(line_lower)
    if match:
        token = match.group(1)
        return int(token) if token.isdigit() else _CARDINAL.get(token)
    for pattern in (_NTH_WICKET_RE, _LOSE_NTH_RE):
        match = pattern.search(line_lower)
        if match:
            return _ORDINAL.get(match.group(1))
    return None


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
    claims_four = _claims_boundary(line_lower, "four")

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

    if state_score is not None:
        claimed_wickets = _claimed_wicket_count(line_lower)
        if claimed_wickets is not None and claimed_wickets != state_score[1]:
            reasons.append(f"wickets {claimed_wickets} != state {state_score[1]}")

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
