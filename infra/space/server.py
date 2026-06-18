"""Hugging Face Space entrypoint: serve the GGUF commentary API on CPU.

Downloads the ball-data dataset once at startup (so every catalogued match is replayable
without baking it into the image), builds the FastAPI serving app with the llama.cpp runtime,
and exposes ``app`` for uvicorn. The Dockerfile runs:

    uvicorn server:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

from app.serve.app import create_app
from app.serve.llama_runtime import LlamaCppRuntime
from app.serve.matches import MatchRepository
from configs.serve import ServeConfig

BALL_DATA_REPO = os.environ.get("BALL_DATA_REPO", "kattymandy/cricket-commentary-ball-data")
CORS_ORIGIN = os.environ.get("CORS_ORIGIN", "*")

_data_dir = Path(snapshot_download(BALL_DATA_REPO, repo_type="dataset"))
_config = ServeConfig(processed_dir=_data_dir, cors_origins=(CORS_ORIGIN,))
app = create_app(
    runtime=LlamaCppRuntime(_config),
    repository=MatchRepository(_data_dir),
    config=_config,
)
