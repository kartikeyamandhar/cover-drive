"""Structured logging setup (structlog).

Wired as configuration only in Phase 0 and used from Phase 1 onward. Centralizing
it here means every entrypoint gets the same structured, typed log pipeline and
no module reaches for the stdlib ``logging`` root logger directly.

Standard (per CLAUDE.md Section 7): structured logging, no secrets in logs.
"""

from __future__ import annotations

import logging
from typing import cast

import structlog
from structlog.typing import FilteringBoundLogger, Processor


def configure_logging(*, json_logs: bool = False, level: int = logging.INFO) -> None:
    """Configure structlog process-wide.

    Args:
        json_logs: emit one JSON object per line (production) instead of the
            colorized console renderer (local development).
        level: the minimum stdlib log level to pass through.

    Idempotent: safe to call more than once; the last call wins.
    """
    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]
    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=True)
    )
    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Return a bound structlog logger, optionally named for the calling module."""
    return cast(FilteringBoundLogger, structlog.get_logger(name))
