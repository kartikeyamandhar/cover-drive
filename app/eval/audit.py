"""Failure-mode hunt: audit teacher output with the LLM judge, catalog the defects.

Two sources of lines to audit:
- ``pilot``: the lines already generated and KEPT by the Phase 2 heuristic filter
  (``data/distill/*.jsonl``). Judging these surfaces FALSE NEGATIVES -- defects the
  cheap regex let through. These are the dangerous ones (they would enter the SFT
  set), and they cost only judge calls, no generation.
- ``generated``: a fresh hard-weighted draw, generated live, with BOTH the kept and
  the rejected lines retained. Judging these also surfaces FALSE POSITIVES (good
  lines the filter wrongly dropped) so the filter can be tuned for recall too.

The judge is the arbiter; the heuristic is the thing on trial. Cross-tabulating the
two gives the filter's precision and recall against a strong reader, a per-mode
defect catalog, and a go/no-go faithfulness rate for the full run.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import anthropic
import structlog

from app.distill.filters import faithfulness_check
from app.distill.sampling import SampleCandidate, stratified_sample
from app.distill.teacher import generate_one, generation_cost
from app.eval.judge import Verdict, judge_cost, judge_one
from configs.distill import DistillConfig
from configs.personas import PERSONAS, Persona, persona_by_key

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class AuditItem:
    """One commentary line to audit, with the teacher input it was generated from."""

    persona_key: str
    persona_instruction: str
    event: str
    state: str
    line: str
    bucket: str
    match_id: str
    ball_id: str
    heuristic_ok: bool
    heuristic_reasons: list[str]
    source: str  # "pilot" | "generated"


@dataclass(frozen=True)
class JudgedItem:
    """An audit item with its judge verdict and the call cost."""

    item: AuditItem
    verdict: Verdict | None
    cost: float
    raw: str


@dataclass
class AuditOutcome:
    """The result of an audit run: judged items plus accounting."""

    judged: list[JudgedItem]
    judge_spend: float = 0.0
    gen_spend: float = 0.0
    n_judge_failed: int = 0


# --- Building items -------------------------------------------------------------


def _pair_to_item(pair: dict[str, str]) -> AuditItem:
    persona = persona_by_key(pair["persona"])
    event, state, line = pair["event"], pair["state"], pair["commentary"]
    faith = faithfulness_check(line, event, state)
    return AuditItem(
        persona_key=persona.key,
        persona_instruction=persona.instruction,
        event=event,
        state=state,
        line=line,
        bucket=pair.get("bucket", "?"),
        match_id=pair.get("match_id", "?"),
        ball_id=pair.get("ball_id", "?"),
        heuristic_ok=faith.ok,
        heuristic_reasons=list(faith.reasons),
        source="pilot",
    )


def build_items_from_pilot(config: DistillConfig) -> list[AuditItem]:
    """Read the kept pilot pairs from ``data/distill/`` into audit items."""
    items: list[AuditItem] = []
    for path in sorted(config.distill_dir.glob("*.jsonl")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                items.append(_pair_to_item(json.loads(raw)))
    log.info("pilot items loaded", n=len(items))
    return items


def generate_items(
    client: anthropic.Anthropic,
    config: DistillConfig,
    candidates: list[SampleCandidate],
    *,
    personas: tuple[Persona, ...] = PERSONAS,
) -> tuple[list[AuditItem], float]:
    """Generate a line per candidate (rotating personas) and build audit items.

    Retains BOTH heuristic-kept and heuristic-rejected lines so the audit can score
    the filter for precision AND recall. Returns the items and the generation spend.
    """
    items: list[AuditItem] = []
    spend = 0.0
    for i, cand in enumerate(candidates):
        persona = personas[i % len(personas)]
        gen = generate_one(client, cand.event, cand.state, persona, config)
        spend += generation_cost(gen, config, batch=False)
        faith = faithfulness_check(gen.text, cand.event, cand.state)
        items.append(
            AuditItem(
                persona_key=persona.key,
                persona_instruction=persona.instruction,
                event=cand.event,
                state=cand.state,
                line=gen.text,
                bucket=cand.bucket,
                match_id=cand.match_id,
                ball_id=cand.ball_id,
                heuristic_ok=faith.ok,
                heuristic_reasons=list(faith.reasons),
                source="generated",
            )
        )
    log.info("generated items", n=len(items), spend=round(spend, 4))
    return items, spend


def hard_candidates(config: DistillConfig, n: int) -> list[SampleCandidate]:
    """A small hard-weighted draw for the generation rounds (existing stratification
    already over-weights wickets/boundaries/chase pressure)."""
    return stratified_sample(config, set_size=n)


@dataclass(frozen=True)
class GenSpec:
    """A ball to regenerate, with the persona it was originally rendered in."""

    event: str
    state: str
    persona_key: str
    bucket: str
    match_id: str
    ball_id: str


def load_unfaithful_specs(path: Path) -> list[GenSpec]:
    """Read a prior ``*.judged.jsonl`` and return the balls the judge ruled unfaithful.

    The before/after lever: regenerate exactly these (previously 0%-faithful) balls with
    the corrected prompt and re-judge, so the repair rate is measured, not assumed.
    """
    specs: list[GenSpec] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        rec = json.loads(raw)
        verdict = rec.get("verdict")
        if verdict is not None and not verdict.get("faithful", True):
            specs.append(
                GenSpec(
                    event=rec["event"],
                    state=rec["state"],
                    persona_key=rec["persona"],
                    bucket=rec["bucket"],
                    match_id=rec["match_id"],
                    ball_id=rec["ball_id"],
                )
            )
    return specs


def generate_from_specs(
    client: anthropic.Anthropic, config: DistillConfig, specs: list[GenSpec]
) -> tuple[list[AuditItem], float]:
    """Regenerate each spec in its original persona with the current prompt."""
    items: list[AuditItem] = []
    spend = 0.0
    for spec in specs:
        persona = persona_by_key(spec.persona_key)
        gen = generate_one(client, spec.event, spec.state, persona, config)
        spend += generation_cost(gen, config, batch=False)
        faith = faithfulness_check(gen.text, spec.event, spec.state)
        items.append(
            AuditItem(
                persona_key=spec.persona_key,
                persona_instruction=persona.instruction,
                event=spec.event,
                state=spec.state,
                line=gen.text,
                bucket=spec.bucket,
                match_id=spec.match_id,
                ball_id=spec.ball_id,
                heuristic_ok=faith.ok,
                heuristic_reasons=list(faith.reasons),
                source="regen",
            )
        )
    log.info("regenerated defects", n=len(items), spend=round(spend, 4))
    return items, spend


# --- Judging --------------------------------------------------------------------


def _judge_item(
    client: anthropic.Anthropic, item: AuditItem, model: str, use_thinking: bool
) -> JudgedItem:
    try:
        res = judge_one(
            client,
            persona_key=item.persona_key,
            persona_instruction=item.persona_instruction,
            event=item.event,
            state=item.state,
            line=item.line,
            model=model,
            use_thinking=use_thinking,
        )
        return JudgedItem(item=item, verdict=res.verdict, cost=judge_cost(res, model), raw=res.raw)
    except anthropic.APIError as exc:  # one bad call must not kill the run
        log.warning("judge call failed", error=str(exc))
        return JudgedItem(item=item, verdict=None, cost=0.0, raw=f"ERROR: {exc}")


def _chunks(items: list[AuditItem], size: int) -> list[list[AuditItem]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def run_audit(
    client: anthropic.Anthropic,
    items: list[AuditItem],
    *,
    model: str = "claude-opus-4-8",
    use_thinking: bool = False,
    max_workers: int = 6,
    budget_usd: float = 6.0,
) -> AuditOutcome:
    """Judge every item, concurrently, stopping before ``budget_usd`` is exceeded.

    The first item is judged solo so the cached rubric prefix is written once; the
    rest fan out in waves of ``max_workers`` with a budget check between waves.
    """
    if not items:
        return AuditOutcome(judged=[])
    judged: list[JudgedItem] = [_judge_item(client, items[0], model, use_thinking)]
    spend = judged[0].cost
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for chunk in _chunks(items[1:], max_workers):
            if spend >= budget_usd:
                log.warning("audit budget reached", spend=round(spend, 4), budget=budget_usd)
                break
            results = list(pool.map(lambda it: _judge_item(client, it, model, use_thinking), chunk))
            judged.extend(results)
            spend += sum(r.cost for r in results)
            log.info("audit progress", done=len(judged), total=len(items), spend=round(spend, 4))
    n_failed = sum(1 for j in judged if j.verdict is None)
    return AuditOutcome(judged=judged, judge_spend=spend, n_judge_failed=n_failed)


# --- Reporting ------------------------------------------------------------------


@dataclass
class AuditReport:
    """Aggregate metrics and the heuristic-vs-judge cross-tab for one audit run."""

    n: int
    n_judged: int
    n_judge_failed: int
    faithful: int
    unfaithful: int
    faithfulness_rate: float
    severity: dict[str, int]
    failure_modes: dict[str, int]
    persona_match_rate: float
    false_negatives: int  # heuristic kept, judge says unfaithful (dangerous)
    false_positives: int  # heuristic rejected, judge says faithful (over-rejection)
    true_rejects: int  # heuristic rejected, judge agrees unfaithful
    by_bucket: dict[str, list[int]]  # bucket -> [faithful, total]
    by_persona: dict[str, list[int]]
    judge_spend: float
    gen_spend: float

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "n_judged": self.n_judged,
            "n_judge_failed": self.n_judge_failed,
            "faithful": self.faithful,
            "unfaithful": self.unfaithful,
            "faithfulness_rate": round(self.faithfulness_rate, 4),
            "severity": self.severity,
            "failure_modes": self.failure_modes,
            "persona_match_rate": round(self.persona_match_rate, 4),
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            "true_rejects": self.true_rejects,
            "by_bucket": self.by_bucket,
            "by_persona": self.by_persona,
            "judge_spend_usd": round(self.judge_spend, 4),
            "gen_spend_usd": round(self.gen_spend, 4),
        }


def summarize(outcome: AuditOutcome) -> AuditReport:
    """Aggregate verdicts into metrics, the failure-mode histogram, and the cross-tab."""
    judged = [j for j in outcome.judged if j.verdict is not None]
    severity: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    by_bucket: dict[str, list[int]] = {}
    by_persona: dict[str, list[int]] = {}
    faithful = persona_ok = fn = fp = tr = 0

    for j in judged:
        v = j.verdict
        assert v is not None
        severity[v.severity] += 1
        persona_ok += int(v.persona_match)
        for tag in v.failure_modes:
            modes[tag] += 1
        bkt = by_bucket.setdefault(j.item.bucket, [0, 0])
        per = by_persona.setdefault(j.item.persona_key, [0, 0])
        bkt[1] += 1
        per[1] += 1
        if v.faithful:
            faithful += 1
            bkt[0] += 1
            per[0] += 1
            if not j.item.heuristic_ok:
                fp += 1
        else:
            if j.item.heuristic_ok:
                fn += 1
            else:
                tr += 1

    n_judged = len(judged)
    return AuditReport(
        n=len(outcome.judged),
        n_judged=n_judged,
        n_judge_failed=outcome.n_judge_failed,
        faithful=faithful,
        unfaithful=n_judged - faithful,
        faithfulness_rate=faithful / n_judged if n_judged else 0.0,
        severity=dict(severity),
        failure_modes=dict(modes.most_common()),
        persona_match_rate=persona_ok / n_judged if n_judged else 0.0,
        false_negatives=fn,
        false_positives=fp,
        true_rejects=tr,
        by_bucket=by_bucket,
        by_persona=by_persona,
        judge_spend=outcome.judge_spend,
        gen_spend=outcome.gen_spend,
    )


def _defect_entries(outcome: AuditOutcome) -> list[JudgedItem]:
    """Judged items the JUDGE ruled unfaithful, worst severity first."""
    order = {"critical": 0, "major": 1, "minor": 2, "none": 3}
    defects = [j for j in outcome.judged if j.verdict is not None and not j.verdict.faithful]
    return sorted(defects, key=lambda j: order.get(j.verdict.severity, 9))  # type: ignore[union-attr]


def render_catalog(report: AuditReport, outcome: AuditOutcome, *, label: str) -> str:
    """A human-readable markdown catalog: metrics, then defects grouped by mode."""
    lines: list[str] = [
        f"# Failure-mode audit - {label}",
        "",
        f"- items judged: {report.n_judged} (judge failures: {report.n_judge_failed})",
        f"- judge faithfulness rate: {report.faithfulness_rate:.1%} "
        f"({report.faithful} faithful / {report.unfaithful} defects)",
        f"- persona-match rate: {report.persona_match_rate:.1%}",
        f"- heuristic vs judge: false-negatives (kept but bad) = {report.false_negatives}, "
        f"false-positives (dropped but fine) = {report.false_positives}, "
        f"true-rejects = {report.true_rejects}",
        f"- severity: {report.severity}",
        f"- spend: judge ${report.judge_spend:.4f} + generation ${report.gen_spend:.4f}",
        "",
        "## Failure modes (judge tags)",
        "",
    ]
    if report.failure_modes:
        for tag, count in report.failure_modes.items():
            lines.append(f"- {tag}: {count}")
    else:
        lines.append("- none")

    lines += ["", "## Defects (judge ruled unfaithful), worst first", ""]
    for j in _defect_entries(outcome):
        v = j.verdict
        assert v is not None
        kept = "KEPT by heuristic" if j.item.heuristic_ok else "dropped by heuristic"
        lines += [
            f"### [{v.severity}] {', '.join(v.failure_modes) or 'unspecified'}  ({kept})",
            f"- persona/bucket: {j.item.persona_key} / {j.item.bucket}  "
            f"(match {j.item.match_id} ball {j.item.ball_id})",
            f"- EVENT: {j.item.event}",
            f"- STATE: {j.item.state}",
            f"- LINE: {j.item.line}",
            f"- judge: {v.explanation}",
            "",
        ]
    return "\n".join(lines)


@dataclass(frozen=True)
class _WriteResult:
    report_path: Path
    catalog_path: Path
    judged_path: Path


def write_outputs(
    out_dir: Path, report: AuditReport, outcome: AuditOutcome, *, label: str
) -> _WriteResult:
    """Write report.json, catalog.md, and a judged.jsonl of every verdict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{label}.report.json"
    catalog_path = out_dir / f"{label}.catalog.md"
    judged_path = out_dir / f"{label}.judged.jsonl"
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    catalog_path.write_text(render_catalog(report, outcome, label=label), encoding="utf-8")
    with judged_path.open("w", encoding="utf-8") as fh:
        for j in outcome.judged:
            v = j.verdict
            fh.write(
                json.dumps(
                    {
                        "persona": j.item.persona_key,
                        "bucket": j.item.bucket,
                        "match_id": j.item.match_id,
                        "ball_id": j.item.ball_id,
                        "event": j.item.event,
                        "state": j.item.state,
                        "line": j.item.line,
                        "heuristic_ok": j.item.heuristic_ok,
                        "heuristic_reasons": j.item.heuristic_reasons,
                        "verdict": v.model_dump() if v is not None else None,
                        "source": j.item.source,
                    }
                )
                + "\n"
            )
    return _WriteResult(report_path=report_path, catalog_path=catalog_path, judged_path=judged_path)
