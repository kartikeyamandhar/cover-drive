"""Tests for the failure-mode audit orchestration. The SDK is mocked; no live calls."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import anthropic

from app.distill.sampling import SampleCandidate
from app.eval.audit import (
    AuditItem,
    build_items_from_pilot,
    generate_from_specs,
    generate_items,
    load_unfaithful_specs,
    run_audit,
    summarize,
    write_outputs,
)
from configs.distill import DistillConfig

_MARK = "HALLUCINATE"


def _verdict_for(line: str) -> dict[str, object]:
    faithful = _MARK not in line
    return {
        "faithful": faithful,
        "severity": "none" if faithful else "major",
        "failure_modes": [] if faithful else ["hallucinated_name"],
        "persona_match": True,
        "confidence": "high",
        "explanation": "ok" if faithful else "invented a name",
    }


def _line_from(kwargs: dict[str, object]) -> str:
    messages = cast(list[dict[str, str]], kwargs["messages"])
    content = messages[0]["content"]
    for piece in content.splitlines():
        if piece.startswith("LINE: "):
            return piece[len("LINE: ") :]
    return ""


def _usage() -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=900,
        output_tokens=80,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )


class _FakeJudgeMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        verdict = _verdict_for(_line_from(kwargs))
        return SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", name="record_verdict", input=verdict)],
            usage=_usage(),
        )


class _FakeJudgeClient:
    def __init__(self) -> None:
        self.messages = _FakeJudgeMessages()


def _judge_client() -> anthropic.Anthropic:
    return cast(anthropic.Anthropic, _FakeJudgeClient())


def _item(line: str, *, heuristic_ok: bool, bucket: str = "wicket") -> AuditItem:
    return AuditItem(
        persona_key="broadcast",
        persona_instruction="lead caller",
        event="WICKET, caught",
        state="T20 | Inns 1 | 5.4 | CSK 45/2 | CRR 7.9 | Striker F du Plessis 33(17) | "
        "Bowler CH Morris 1/1 | P'ship 20(11) | powerplay",
        line=line,
        bucket=bucket,
        match_id="m1",
        ball_id="0-5-4",
        heuristic_ok=heuristic_ok,
        heuristic_reasons=[] if heuristic_ok else ["claims a boundary that did not happen"],
        source="pilot",
    )


def test_cross_tab_classifies_all_four_cells() -> None:
    items = [
        _item(f"Caught by {_MARK} at slip!", heuristic_ok=True),  # false negative
        _item("Caught well, Morris gets the wicket.", heuristic_ok=True),  # agree faithful
        _item(f"{_MARK} and a six too", heuristic_ok=False),  # true reject
        _item("A fine catch, dropped by the filter.", heuristic_ok=False),  # false positive
    ]
    outcome = run_audit(_judge_client(), items, budget_usd=10.0, max_workers=3)
    report = summarize(outcome)
    assert report.n_judged == 4
    assert report.false_negatives == 1
    assert report.true_rejects == 1
    assert report.false_positives == 1
    assert report.faithful == 2
    assert abs(report.faithfulness_rate - 0.5) < 1e-9
    assert report.failure_modes["hallucinated_name"] == 2


def test_budget_guard_stops_after_first_item() -> None:
    items = [_item("line one", heuristic_ok=True) for _ in range(8)]
    # first item is judged solo at ~$0.0065; a tiny budget stops the waves.
    outcome = run_audit(_judge_client(), items, budget_usd=0.001, max_workers=4)
    assert len(outcome.judged) == 1


def test_summarize_tracks_per_bucket() -> None:
    items = [
        _item("clean", heuristic_ok=True, bucket="six"),
        _item(f"{_MARK} here", heuristic_ok=True, bucket="six"),
    ]
    report = summarize(run_audit(_judge_client(), items, budget_usd=10.0))
    assert report.by_bucket["six"] == [1, 2]  # [faithful, total]


def test_build_items_from_pilot_runs_heuristic(tmp_path: Path) -> None:
    config = DistillConfig(data_dir=tmp_path / "data")
    config.distill_dir.mkdir(parents=True)
    pairs = [
        {  # clean -> heuristic ok
            "persona": "broadcast",
            "event": "1 run",
            "state": "T20 | Inns 1 | 5.4 | CSK 45/2 | powerplay",
            "commentary": "Nudged into the leg side for a single.",
            "bucket": "powerplay_routine",
            "match_id": "m1",
            "ball_id": "0-5-4",
        },
        {  # claims a six on a 1-run ball -> heuristic flags it
            "persona": "broadcast",
            "event": "1 run",
            "state": "T20 | Inns 1 | 5.5 | CSK 46/2 | powerplay",
            "commentary": "That is a massive SIX into the stands!",
            "bucket": "powerplay_routine",
            "match_id": "m1",
            "ball_id": "0-5-5",
        },
    ]
    (config.distill_dir / "broadcast.jsonl").write_text(
        "\n".join(json.dumps(p) for p in pairs) + "\n", encoding="utf-8"
    )
    items = build_items_from_pilot(config)
    assert len(items) == 2
    by_ball = {it.ball_id: it for it in items}
    assert by_ball["0-5-4"].heuristic_ok is True
    assert by_ball["0-5-5"].heuristic_ok is False


class _FakeTeacherMessages:
    def create(self, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Driven through the covers for four.")],
            usage=_usage(),
        )


class _FakeTeacherClient:
    def __init__(self) -> None:
        self.messages = _FakeTeacherMessages()


def test_generate_items_builds_and_costs() -> None:
    client = cast(anthropic.Anthropic, _FakeTeacherClient())
    candidates = [
        SampleCandidate(
            match_id="m1",
            ball_id="0-5-4",
            bucket="four",
            event="FOUR off the bat",
            state="T20 | Inns 1 | 5.4 | CSK 45/2 | Striker X 10(8) | powerplay",
        )
    ]
    items, spend = generate_items(client, DistillConfig(), candidates)
    assert len(items) == 1
    assert items[0].source == "generated"
    assert items[0].line == "Driven through the covers for four."
    assert spend > 0


def test_load_unfaithful_specs_filters_to_defects(tmp_path: Path) -> None:
    path = tmp_path / "prior.judged.jsonl"
    rows = [
        {  # faithful -> excluded
            "persona": "broadcast",
            "bucket": "four",
            "match_id": "m1",
            "ball_id": "0-1-1",
            "event": "FOUR off the bat",
            "state": "S1",
            "line": "Four.",
            "verdict": {"faithful": True, "failure_modes": []},
        },
        {  # unfaithful -> kept
            "persona": "analyst",
            "bucket": "tight_finish",
            "match_id": "m2",
            "ball_id": "1-19-3",
            "event": "1 run",
            "state": "S2",
            "line": "11 off 7.",
            "verdict": {"faithful": False, "failure_modes": ["chase_equation_error"]},
        },
        {"verdict": None},  # judge failure -> excluded
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    specs = load_unfaithful_specs(path)
    assert len(specs) == 1
    assert specs[0].persona_key == "analyst"
    assert specs[0].ball_id == "1-19-3"


def test_generate_from_specs_preserves_persona(tmp_path: Path) -> None:
    from app.eval.audit import GenSpec

    client = cast(anthropic.Anthropic, _FakeTeacherClient())
    specs = [GenSpec("1 run", "S", "radio", "tight_finish", "m2", "1-19-3")]
    items, spend = generate_from_specs(client, DistillConfig(), specs)
    assert len(items) == 1
    assert items[0].persona_key == "radio"
    assert items[0].source == "regen"
    assert spend > 0


def test_write_outputs_emits_three_files(tmp_path: Path) -> None:
    items = [_item("clean", heuristic_ok=True), _item(f"{_MARK} bad", heuristic_ok=True)]
    outcome = run_audit(_judge_client(), items, budget_usd=10.0)
    report = summarize(outcome)
    written = write_outputs(tmp_path, report, outcome, label="round0")
    assert written.report_path.exists()
    assert written.catalog_path.exists()
    assert written.judged_path.exists()
    saved = json.loads(written.report_path.read_text(encoding="utf-8"))
    assert saved["n_judged"] == 2
    assert "Failure-mode audit" in written.catalog_path.read_text(encoding="utf-8")
    assert len(written.judged_path.read_text(encoding="utf-8").splitlines()) == 2
