"""CLI: the boundary failure-mode hunt -- audit teacher output with the LLM judge.

uv run python -m scripts.audit --source pilot --label round0          # judge kept pilot lines
uv run python -m scripts.audit --generate 200 --label round1         # fresh hard draw, live gen
uv run python -m scripts.audit --source pilot --thinking --budget 8  # careful re-check

Writes data/audit/<label>.report.json + .catalog.md + .judged.jsonl (git-ignored).
The judge runs on Opus by default (a stronger model than the Sonnet teacher).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anthropic

from app.distill.errors import DistillError
from app.eval.audit import (
    AuditOutcome,
    build_items_from_pilot,
    generate_from_specs,
    generate_items,
    hard_candidates,
    load_unfaithful_specs,
    run_audit,
    summarize,
    write_outputs,
)
from app.logging_config import configure_logging
from configs.distill import DistillConfig
from configs.settings import get_settings


def _make_client() -> anthropic.Anthropic:
    settings = get_settings()
    if settings.anthropic_api_key is None:
        raise DistillError("ANTHROPIC_API_KEY is not set; add it to .env before a live run")
    return anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit teacher output for failure modes.")
    parser.add_argument("--source", choices=("pilot",), default="pilot", help="lines to audit")
    parser.add_argument("--generate", type=int, default=None, help="generate N fresh hard lines")
    parser.add_argument(
        "--regen-defects",
        type=Path,
        default=None,
        help="regenerate unfaithful balls from a judged.jsonl",
    )
    parser.add_argument("--label", type=str, default="audit", help="output file label")
    parser.add_argument("--model", type=str, default="claude-opus-4-8", help="judge model id")
    parser.add_argument("--thinking", action="store_true", help="judge with adaptive thinking")
    parser.add_argument("--workers", type=int, default=6, help="concurrent judge calls")
    parser.add_argument("--budget", type=float, default=6.0, help="judge USD budget guard")
    parser.add_argument("--limit", type=int, default=None, help="cap items (for a live smoke test)")
    parser.add_argument("--data-dir", type=Path, default=None, help="root data directory")
    parser.add_argument("--json-logs", action="store_true", help="emit JSON logs")
    args = parser.parse_args()

    configure_logging(json_logs=args.json_logs)
    config = DistillConfig() if args.data_dir is None else DistillConfig(data_dir=args.data_dir)

    client = _make_client()
    gen_spend = 0.0
    if args.regen_defects is not None:
        specs = load_unfaithful_specs(args.regen_defects)
        items, gen_spend = generate_from_specs(client, config, specs)
    elif args.generate is not None:
        candidates = hard_candidates(config, args.generate)
        items, gen_spend = generate_items(client, config, candidates)
    else:
        items = build_items_from_pilot(config)

    if args.limit is not None:
        items = items[: args.limit]

    if not items:
        parser.error("no items to audit (is data/distill populated, or --generate N set?)")

    outcome: AuditOutcome = run_audit(
        client,
        items,
        model=args.model,
        use_thinking=args.thinking,
        max_workers=args.workers,
        budget_usd=args.budget,
    )
    outcome.gen_spend = gen_spend
    report = summarize(outcome)
    written = write_outputs(config.data_dir / "audit", report, outcome, label=args.label)

    print(
        f"judged={report.n_judged} faithful={report.faithfulness_rate:.1%} "
        f"false_neg={report.false_negatives} false_pos={report.false_positives} "
        f"judge_spend=${report.judge_spend:.4f} gen_spend=${report.gen_spend:.4f}"
    )
    print(f"top modes: {report.failure_modes}")
    print(f"catalog: {written.catalog_path}")


if __name__ == "__main__":
    main()
