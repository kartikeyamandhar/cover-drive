"""The FastAPI serving app: list matches, then replay one ball by ball over SSE.

The app depends only on the ``RuntimeAdapter`` protocol and the match repository, so it is
built by a factory (``create_app``) that the real entrypoint wires with the transformers
runtime and the bundled data, and that the tests wire with a ``StubRuntime`` and a tmp repo.

The stream contract per delivery: a ``state`` event (the deterministic scoreboard), then
``token`` events (the validated commentary, replayed chunk by chunk), then a ``ball`` event
(the line plus its provenance). The commentary is validated before any token is sent, so the
client only ever renders faithful text.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.serve.engine import generate_commentary
from app.serve.matches import MatchRepository, MatchSummary, UnknownMatchError
from app.serve.runtime import RuntimeAdapter, token_chunks
from app.serve.scoreboard import scoreboard
from configs.personas import PERSONAS, Persona, persona_by_key
from configs.serve import ServeConfig


def _sse(event: str, data: str) -> str:
    """Frame one Server-Sent Event."""
    return f"event: {event}\ndata: {data}\n\n"


def _resolve_persona(key: str) -> Persona:
    try:
        return persona_by_key(key)
    except KeyError as exc:
        known = ", ".join(p.key for p in PERSONAS)
        raise HTTPException(
            status_code=404, detail=f"unknown persona {key!r}; try: {known}"
        ) from exc


def create_app(
    *, runtime: RuntimeAdapter, repository: MatchRepository, config: ServeConfig | None = None
) -> FastAPI:
    """Build the serving app around a runtime backend and a match repository."""
    cfg = config or ServeConfig()
    app = FastAPI(title="cricket-commentary", version="0.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cfg.cors_origins),
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/personas")
    def personas() -> list[dict[str, str]]:
        return [
            {"key": p.key, "display_name": p.display_name, "instruction": p.instruction}
            for p in PERSONAS
        ]

    @app.get("/matches")
    def matches() -> list[MatchSummary]:
        return repository.list_matches()

    @app.get("/matches/{match_id}")
    def match(match_id: str) -> MatchSummary:
        for summary in repository.list_matches():
            if summary.match_id == match_id:
                return summary
        raise HTTPException(status_code=404, detail=f"unknown match {match_id!r}")

    @app.get("/matches/{match_id}/stream")
    def stream(match_id: str, persona: str = "broadcast", from_ball: int = 0) -> StreamingResponse:
        voice = _resolve_persona(persona)
        try:
            balls = repository.load_balls(match_id)
        except UnknownMatchError as exc:
            raise HTTPException(status_code=404, detail=f"unknown match {match_id!r}") from exc

        def events() -> Iterator[str]:
            for ball in balls[max(from_ball, 0) :]:
                yield _sse("state", scoreboard(ball).model_dump_json())
                result = generate_commentary(runtime, ball, voice, retries=cfg.faithfulness_retries)
                for chunk in token_chunks(result.line):
                    yield _sse("token", json.dumps({"t": chunk}))
                yield _sse("ball", result.model_dump_json())
                if cfg.pacing_seconds > 0:
                    time.sleep(cfg.pacing_seconds)
            yield _sse("done", json.dumps({"match_id": match_id, "persona": voice.key}))

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
