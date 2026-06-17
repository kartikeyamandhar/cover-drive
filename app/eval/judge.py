"""LLM-as-judge faithfulness auditor (Opus), built early for the boundary hunt.

The Phase 2 heuristic ``faithfulness_check`` is a cheap regex gate: it catches the
gross defects (a six called a four, a contradicted score) but it is blind to the
subtle ones that actually poison an SFT set -- an invented fielder name, a wicket
miscount ("their third" when the score shows two down), a fabricated head-to-head.
Those need a reader that understands the STATE the line was generated from.

This module is that reader: a strict auditor running on a STRONGER model than the
teacher (Opus judges Sonnet), so it does not share the generator's blind spots. It
is the seed of the Phase 5 faithfulness evaluation, pulled forward to validate and
harden the filter before the full distillation spend.

The judge sees exactly what the teacher saw -- the persona, the EVENT, and the
STATE string -- and nothing more, so "faithful" means "every hard fact in the line
is supported by that input". It returns a structured ``Verdict`` via a forced-shape
tool call. No live call happens in ``make check``; the client is injected and mocked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import anthropic
import structlog
from anthropic.types import MessageParam, TextBlockParam, ToolParam
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from pydantic import BaseModel, ConfigDict, ValidationError

from configs.distill import CACHE_READ_MULT, CACHE_WRITE_MULT_1H, PRICING

log = structlog.get_logger(__name__)

# Controlled failure-mode vocabulary. The judge tags every defect with one or more
# of these so the catalog aggregates cleanly; "other" carries anything novel in the
# explanation (which then becomes a candidate for a new tag / filter rule).
FAILURE_MODES: tuple[str, ...] = (
    "hallucinated_name",  # a fielder/player/team not present in STATE
    "invented_event",  # an outcome that did not happen (catch/boundary/dismissal)
    "contradicted_score",  # team runs/wickets disagree with STATE
    "wrong_wicket_count",  # "their third" when STATE shows a different tally
    "wrong_runs_on_ball",  # misstates how many runs the delivery scored
    "chase_equation_error",  # wrong runs-needed / balls-left / target
    "wrong_run_rate",  # CRR/RRR misstated
    "wrong_player_score",  # striker runs(balls) or bowler figures misstated
    "wrong_over_or_phase",  # wrong over number or match phase
    "unknowable_claim",  # a specific unverifiable fact asserted (record, crowd count)
    "internal_contradiction",  # the line contradicts itself
    "persona_bleed",  # wrong voice for the requested persona
    "other",
)

_SEVERITY = ("none", "minor", "major", "critical")
_CONFIDENCE = ("high", "medium", "low")


class Verdict(BaseModel):
    """A judge's structured ruling on one commentary line."""

    model_config = ConfigDict(frozen=True)

    faithful: bool
    severity: Literal["none", "minor", "major", "critical"]
    failure_modes: list[str]
    persona_match: bool
    confidence: Literal["high", "medium", "low"]
    explanation: str


@dataclass(frozen=True)
class JudgeResult:
    """A verdict (or ``None`` on a judge malfunction) plus token accounting."""

    verdict: Verdict | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    raw: str


# --- The judge's instruction: role, the facts-vs-color spine, the STATE legend,
# the taxonomy, and worked calibration examples (which also pad the prefix toward
# the cache floor). This block is byte-stable, so it is the cache breakpoint. ---

_RUBRIC = """\
You are a STRICT faithfulness auditor for AI-generated live cricket commentary.

A commentary model was given a single delivery (EVENT) and a compact match STATE,
and asked to narrate it in a PERSONA voice. The model is ALLOWED to add descriptive
COLOR, but it must NEVER state or contradict a HARD FACT it was not given. Your job
is to catch every hard-fact defect, and ONLY real defects.

HARD FACTS (must be supported by EVENT/STATE; a mismatch is a defect):
- the ball outcome: runs scored, boundary (four/six), wicket, or dot
- the team score and wickets, the chase equation (runs needed, balls left, target)
- run rates (CRR, RRR), the over/ball, the match phase
- player and team NAMES, the striker's runs(balls), the bowler's wickets/runs
- the partnership, and any milestone

COLOR (allowed, NOT a defect):
- the shot type and placement ("driven through covers", "pulled to deep square")
- mood, tension, generic pressure talk, the bowler's effort, ambient crowd energy
- anything that does not assert a checkable fact

A line is FAITHFUL only if EVERY hard fact in it is supported by EVENT/STATE. Color
is never a defect. When genuinely uncertain whether a stated fact is supported, lean
toward flagging -- but do not invent defects: plausible shot description is color.

HOW TO READ THE STATE STRING (pipe-separated, computed AFTER this ball):
  "<competition> | Inns N | O.B | TEAM runs/wkts | [need R off B] | CRR x [RRR y] |
   Striker NAME r(b) | Bowler NAME w/runs | P'ship r(b) | [Last ...] | phase | [nearing: ...]"
- "Inns 1" = first innings, "Inns 2" = the chase.
- "O.B" is standard over notation: O completed overs and B balls. "15.6" means the
  16th over is in progress (over number = O + 1). Naming the over O instead of O+1
  is a minor defect.
- "TEAM 106/5" = 106 runs for 5 wickets, INCLUDING this ball. On a WICKET ball the
  shown wicket count ALREADY counts this dismissal: if it reads /2, this is the 2nd
  wicket, so "their third" would be wrong_wicket_count. WICKETS IN HAND / STANDING /
  REMAINING = 10 minus the shown count: 146/6 means 6 down and 4 wickets in hand, so
  "four wickets standing" is CORRECT, not a defect. Only flag the FALLEN count.
- "need 7 off 5" = 7 runs required off 5 balls left (chase only).
- "Striker M Jansen 5(8)" = 5 runs off 8 balls, AFTER this ball. Names are
  abbreviated (initial + surname); using just the surname is fine, a DIFFERENT name
  is hallucinated.
- "Bowler M Pathirana 0/10" = 0 wickets for 10 runs in this spell.
- "Last 1 0 1 1 0 1" = the last six deliveries (digits = runs, W = wicket).
- FIELDER / CATCHER NAMES ARE NEVER IN THE STATE. On "WICKET, caught", naming the
  catcher is a hallucinated_name -- UNLESS the named person is the bowler (c&b) or
  the line credits "the bowler". Crediting the bowler by name (the Bowler in STATE)
  on a wicket is fine.

EVENT vocabulary: "WICKET, <kind>" (caught/bowled/lbw/stumped/run out/...),
"SIX off the bat", "FOUR off the bat", "wide", "no-ball", "dot ball", "N run(s)".
A run-out's EVENT is "WICKET, run out" and is NOT the bowler's wicket.

CRICKET DISAMBIGUATIONS (do NOT misread these as defects, and do not be fooled by
them either -- the word "four"/"six" has several unrelated cricket meanings):
- "six down" / "four wickets down" / "55 for four" = a WICKET tally or a score, NOT a
  boundary. A line saying "they are six down" on a non-boundary ball is fine.
- "inside six overs" / "in the sixth over" = an OVER reference, not a six.
- "two for six" / "figures of four for twenty" = BOWLING figures, not boundaries.
- byes and leg-byes are runs but credit NO batter; a wide/no-ball is an extra. The
  EVENT names the true outcome ("wide", "no-ball", "N runs"); judge the line against it.
- a four/six "off the bat" requires EVENT "FOUR/SIX off the bat". Four byes is not a
  batter boundary. If EVENT is "1 run" and the line claims a boundary, that is invented.
- a milestone the STATE only flags as "nearing" has NOT yet been reached; claiming it
  is reached (e.g. "brings up the fifty") when STATE shows "nearing: ..." is a defect.
- the chasing side's runs_required and balls_left already reflect THIS ball.

FAILURE-MODE TAGS (use one or more for any defect; "other" for anything novel):
  hallucinated_name, invented_event, contradicted_score, wrong_wicket_count,
  wrong_runs_on_ball, chase_equation_error, wrong_run_rate, wrong_player_score,
  wrong_over_or_phase, unknowable_claim, internal_contradiction, persona_bleed, other.

SEVERITY: critical = the ball outcome itself is wrong (a dot called a six, an
invented wicket); major = a wrong name/score/wicket-count/equation; minor = a soft
over-number slip or a fabricated non-load-bearing specific; none = faithful.

WORKED EXAMPLES (calibrate to these):

1) PERSONA broadcast | EVENT "FOUR off the bat" |
   STATE "... RCB 162/4 | CRR 8.1 | Striker V Kohli 54(38) | Bowler ..."
   LINE "Kohli leans into the drive, finds the gap at cover and races away for four."
   -> FAITHFUL. four matches EVENT, Kohli is in STATE, "cover" is color. severity none.

2) EVENT "WICKET, caught" |
   STATE "... 88/3 | Striker S Iyer 41(29) | Bowler R Jadeja 1/22 | ..."
   LINE "Caught at deep midwicket by Rohit Sharma, Iyer has to go!"
   -> UNFAITHFUL. Rohit Sharma is not in STATE; the catcher is invented.
   failure_modes [hallucinated_name], severity major.

3) EVENT "WICKET, caught" | STATE "... CSK 45/2 | Striker F du Plessis 33(17) | ..."
   LINE "du Plessis goes for 33 and that's CSK's third wicket down."
   -> UNFAITHFUL. STATE shows /2 (inclusive), so this is the 2nd wicket.
   failure_modes [wrong_wicket_count], severity major.

4) EVENT "1 run" | STATE "... death ..."
   LINE "Launched high over long-on, that has gone all the way for SIX!"
   -> UNFAITHFUL. The ball scored 1, not a six.
   failure_modes [invented_event, wrong_runs_on_ball], severity critical.

5) EVENT "dot ball" | STATE "... need 30 off 12 | RRR 15.0 | death ..."
   LINE "Full and quick, squeezed out to the man at point, no run, and the
   asking rate climbs again -- huge pressure here."
   -> FAITHFUL. dot is correct, "point" is color, the equation is not misstated.
   severity none.

6) EVENT "FOUR off the bat" | STATE "... Striker KL Rahul 30(20) ..."
   LINE "That's his second fifty in three games and the fourth boundary of the over."
   -> UNFAITHFUL. A cross-match record ("second fifty in three games") is unknowable
   from STATE; "fourth boundary of the over" is unsupported by Last.
   failure_modes [unknowable_claim], severity major.

7) EVENT "2 runs" | STATE "... Inns 2 | MI 150/6 | need 18 off 9 | RRR 12.0 | death ..."
   LINE "Pushed for two, and now it's 18 needed off the last nine balls -- squeaky."
   -> FAITHFUL. The equation (18 off 9) matches STATE; "squeaky" is color. severity none.

8) EVENT "1 run" | STATE "... Inns 2 | need 40 off 30 | RRR 8.0 | middle ..."
   LINE "Single taken, they need just six an over from here, cruising."
   -> UNFAITHFUL. RRR is 8.0 (eight an over), not six. failure_modes [chase_equation_error,
   wrong_run_rate], severity major.

9) EVENT "FOUR off the bat" | STATE "... Striker R Sharma 24(19) | Bowler ..."
   LINE "Rohit brings up a brilliant fifty with that boundary!"
   -> UNFAITHFUL. STATE shows 24 off 19; he is not on fifty. failure_modes
   [wrong_player_score], severity major.

10) PERSONA text (terse written wire) | EVENT "SIX off the bat" | STATE "... Striker ..."
    LINE "OH WHAT A SHOT!! That is ENORMOUS, the crowd is on its feet, unbelievable!!"
    -> Faithful on facts (a six is a six) BUT persona_match=false: the Wire is a clipped
    written note, not a frantic spoken shout. failure_modes [persona_bleed], severity minor.

11) EVENT "WICKET, run out" | STATE "... Bowler JJ Bumrah 1/20 | ..."
    LINE "Bumrah strikes again, that's his second wicket, run out at the striker's end!"
    -> UNFAITHFUL. A run out is NOT the bowler's wicket; Bumrah's figures (1/20) are
    unchanged. failure_modes [wrong_player_score, internal_contradiction], severity major.

12) EVENT "WICKET, caught and bowled" | STATE "... Bowler Y Chahal 2/28 | ..."
    LINE "Caught and bowled! Chahal takes a sharp return catch, two now for the leggie."
    -> FAITHFUL. Crediting the bowler Chahal (named in STATE) on a c&b is correct; "2/28"
    supports "two". severity none.

13) EVENT "wide" | STATE "... Bowler M Starc 0/14 | death ..."
    LINE "Sprayed down the leg side, called wide, an extra and the pressure tells on Starc."
    -> FAITHFUL. The EVENT is a wide; calling it a wide and an extra is correct, the rest
    is color. severity none.

14) EVENT "FOUR off the bat" | STATE "... 14.2 | RR 110/3 | P'ship 18(11) | middle ..."
    LINE "Driven for four, and this fourth-wicket stand is up to fifty in no time."
    -> UNFAITHFUL. STATE shows the partnership at 18, not fifty. failure_modes
    [wrong_player_score], severity major. (Partnership figure fabricated.)

15) EVENT "dot ball" | STATE "... 9.6 | RR 78/2 | middle ..."
    LINE "Defended back to the bowler to end the eleventh over, still 78 for two."
    -> FAITHFUL. "9.6" is the end of the 10th over, so "eleventh" is a small over-number
    slip -- but if you judge it unfaithful, use [wrong_over_or_phase], severity minor.
    The score 78/2 is correct. Prefer faithful here unless the over error is blatant.

Reason briefly through the hard facts in the line, then call record_verdict EXACTLY
once. Do not output prose after the tool call."""


_VERDICT_TOOL: ToolParam = {
    "name": "record_verdict",
    "description": "Record the faithfulness ruling for the commentary line.",
    "input_schema": {
        "type": "object",
        "properties": {
            "faithful": {
                "type": "boolean",
                "description": "True only if every hard fact in the line is supported.",
            },
            "severity": {"type": "string", "enum": list(_SEVERITY)},
            "failure_modes": {
                "type": "array",
                "items": {"type": "string", "enum": list(FAILURE_MODES)},
                "description": "Empty if faithful; else one or more tags.",
            },
            "persona_match": {
                "type": "boolean",
                "description": "True if the voice fits the requested persona.",
            },
            "confidence": {"type": "string", "enum": list(_CONFIDENCE)},
            "explanation": {
                "type": "string",
                "description": "One terse sentence (<=25 words): the exact defect, or 'faithful'.",
            },
        },
        "required": [
            "faithful",
            "severity",
            "failure_modes",
            "persona_match",
            "confidence",
            "explanation",
        ],
    },
}


def system_blocks() -> list[TextBlockParam]:
    """The stable auditor rubric, with the cache breakpoint on the last block."""
    return [{"type": "text", "text": _RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}]


def user_message(
    persona_key: str, persona_instruction: str, event: str, state: str, line: str
) -> MessageParam:
    """The volatile per-line turn the judge rules on."""
    text = (
        f"PERSONA: {persona_key} - {persona_instruction}\n"
        f"EVENT: {event}\n"
        f"STATE: {state}\n"
        f"LINE: {line}\n\n"
        "Audit the LINE. Call record_verdict."
    )
    return {"role": "user", "content": text}


def _parse_verdict(message: anthropic.types.Message) -> tuple[Verdict | None, str]:
    """Pull the verdict from the tool call, falling back to JSON in any text block."""
    for block in message.content:
        if block.type == "tool_use" and block.name == "record_verdict":
            try:
                return Verdict.model_validate(block.input), json.dumps(block.input)
            except ValidationError:
                return None, json.dumps(block.input)
    # Fallback: a JSON object in the text (e.g. if the model declined the tool).
    text = next((b.text for b in message.content if b.type == "text"), "")
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            return Verdict.model_validate_json(text[start : end + 1]), text
        except ValidationError:
            return None, text
    return None, text


def judge_one(
    client: anthropic.Anthropic,
    *,
    persona_key: str,
    persona_instruction: str,
    event: str,
    state: str,
    line: str,
    model: str = "claude-opus-4-8",
    use_thinking: bool = False,
    max_tokens: int = 1024,
) -> JudgeResult:
    """Audit one commentary line with the judge model and return a structured result.

    Two modes, both returning a tool-shaped ``Verdict``:
    - bulk (``use_thinking=False``, the default): the verdict tool is FORCED, so the
      output is exactly the structured ruling -- minimal tokens, cheap, deterministic
      shape. Opus with the worked-example rubric is a strong fact-checker unaided.
    - careful (``use_thinking=True``): adaptive thinking with ``tool_choice`` on auto
      (forcing a tool is incompatible with thinking on this family). For re-checking
      the subtlest flagged cases, where step-by-step reasoning on the numbers helps.
    """
    messages = [user_message(persona_key, persona_instruction, event, state, line)]
    if use_thinking:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_blocks(),
            messages=messages,
            tools=[_VERDICT_TOOL],
            thinking={"type": "adaptive"},
        )
    else:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_blocks(),
            messages=messages,
            tools=[_VERDICT_TOOL],
            tool_choice={"type": "tool", "name": "record_verdict"},
        )
    verdict, raw = _parse_verdict(message)
    usage = message.usage
    if verdict is None:
        log.warning("judge produced no verdict", raw=raw[:200])
    return JudgeResult(
        verdict=verdict,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
        cache_write_tokens=usage.cache_creation_input_tokens or 0,
        raw=raw,
    )


def build_judge_params(
    *,
    persona_key: str,
    persona_instruction: str,
    event: str,
    state: str,
    line: str,
    model: str = "claude-opus-4-8",
    max_tokens: int = 256,
) -> MessageCreateParamsNonStreaming:
    """Forced-tool judge params for the Batches API (no thinking, deterministic shape)."""
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_blocks(),
        "messages": [user_message(persona_key, persona_instruction, event, state, line)],
        "tools": [_VERDICT_TOOL],
        "tool_choice": {"type": "tool", "name": "record_verdict"},
    }


def parse_judge_message(message: anthropic.types.Message) -> tuple[Verdict | None, JudgeResult]:
    """Parse a (sync or batch) judge message into a verdict and its accounting."""
    verdict, raw = _parse_verdict(message)
    usage = message.usage
    return verdict, JudgeResult(
        verdict=verdict,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
        cache_write_tokens=usage.cache_creation_input_tokens or 0,
        raw=raw,
    )


def judge_cost(result: JudgeResult, model: str = "claude-opus-4-8") -> float:
    """USD cost of one judge call from its usage counters (no batch discount)."""
    in_price, out_price = PRICING[model]
    input_usd = (
        result.input_tokens * in_price
        + result.cache_read_tokens * in_price * CACHE_READ_MULT
        + result.cache_write_tokens * in_price * CACHE_WRITE_MULT_1H
    )
    return (input_usd + result.output_tokens * out_price) / 1e6
