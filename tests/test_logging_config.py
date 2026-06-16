"""Tests for the structlog wiring (config-only in Phase 0)."""

from __future__ import annotations

import logging

from app.logging_config import configure_logging, get_logger


def test_configure_console_is_idempotent() -> None:
    configure_logging(json_logs=False, level=logging.INFO)
    configure_logging(json_logs=False, level=logging.INFO)
    logger = get_logger("test.console")
    logger.info("smoke", phase=0)  # must not raise


def test_configure_json_renderer() -> None:
    configure_logging(json_logs=True, level=logging.WARNING)
    logger = get_logger()
    logger.warning("json-smoke", ok=True)  # must not raise
