"""Cricket commentary voice engine: library code.

This package holds all library logic for the project. Each sub-package owns one
stage of the pipeline and is filled in by a later phase:

- ``cricsheet``: acquisition and parsing of structured ball-by-ball data (Phase 1)
- ``features``:  per-ball match-state featurization and serialization (Phase 1)
- ``distill``:   teacher client, style-seed, prompt caching, sampling (Phase 2)
- ``dataset``:   instruction formatting, chat template, split-by-match (Phase 3)
- ``train``:     QLoRA / Unsloth training entrypoint and config (Phase 4)
- ``eval``:      faithfulness, style judge, diversity, comparison (Phase 5)
- ``serve``:     FastAPI app, runtime adapter, deterministic fact-fill, SSE (Phase 6)

The governing invariant is facts-versus-voice separation: the model owns voice
only; every fact is supplied deterministically from structured data.
"""

from __future__ import annotations

__version__ = "0.0.0"
