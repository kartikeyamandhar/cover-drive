"""Phase 6 serving entrypoint.

  uv run python -m scripts.serve            # serve the real model (transformers + peft)
  uv run python -m scripts.serve --stub     # serve with the stub runtime (no model)
  uv run python -m scripts.serve --smoke    # headless: stream one bundled match to stdout

The ``--smoke`` path uses the stub runtime and the bundled data, so it proves the whole
serving pipeline (repo -> scoreboard -> engine -> validated line) with no GPU and no model.
"""

from __future__ import annotations

import argparse

from app.serve.app import create_app
from app.serve.engine import generate_commentary
from app.serve.matches import MatchRepository
from app.serve.runtime import RuntimeAdapter, StubRuntime
from app.serve.scoreboard import scoreboard
from configs.personas import persona_by_key
from configs.serve import ServeConfig

_STUB_LINE = "A good ball, well played."


def _build_runtime(stub: bool, config: ServeConfig) -> RuntimeAdapter:
    if stub:
        return StubRuntime(_STUB_LINE)
    from app.serve.transformers_runtime import TransformersPeftRuntime

    return TransformersPeftRuntime(config)


def _smoke(config: ServeConfig, persona_key: str, limit: int) -> None:
    repo = MatchRepository(config.processed_dir, config.demo_match_ids)
    summaries = repo.list_matches()
    if not summaries:
        raise SystemExit(f"no matches under {config.processed_dir}")
    match = summaries[0]
    persona = persona_by_key(persona_key)
    runtime = StubRuntime(_STUB_LINE)
    print(f"== {match.teams[0]} vs {match.teams[1]} ({match.match_id}) | persona={persona.key} ==")
    for ball in repo.load_balls(match.match_id)[:limit]:
        sb = scoreboard(ball)
        result = generate_commentary(runtime, ball, persona, retries=config.faithfulness_retries)
        print(f"[{sb.over:>5}] {sb.score:>6} {sb.batting_team}  |  {sb.event}")
        print(f"         {result.line}   ({result.source})")


def main() -> None:
    parser = argparse.ArgumentParser(description="cricket commentary serving")
    parser.add_argument("--stub", action="store_true", help="serve with the stub runtime")
    parser.add_argument("--smoke", action="store_true", help="stream one match to stdout and exit")
    parser.add_argument("--persona", default="broadcast")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    config = ServeConfig()
    if args.smoke:
        _smoke(config, args.persona, args.limit)
        return

    import uvicorn

    runtime = _build_runtime(args.stub, config)
    repo = MatchRepository(config.processed_dir, config.demo_match_ids)
    app = create_app(runtime=runtime, repository=repo, config=config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
