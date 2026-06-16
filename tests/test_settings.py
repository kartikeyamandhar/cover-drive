"""Tests for typed settings: defaults, secret masking, and caching."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from configs.settings import Settings, get_settings

_ENV_VARS = [
    "APP_ENV",
    "LOG_JSON",
    "HF_TOKEN",
    "ANTHROPIC_API_KEY",
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
    "RUNPOD_API_KEY",
    "WANDB_API_KEY",
]


def test_defaults_with_empty_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Run from a directory with no .env and with the relevant vars cleared,
    # so the assertions exercise the in-code defaults deterministically.
    monkeypatch.chdir(tmp_path)
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    settings = Settings()

    assert settings.app_env == "dev"
    assert settings.log_json is False
    assert settings.hf_token is None
    assert settings.anthropic_api_key is None


def test_secret_value_never_leaks_in_repr() -> None:
    settings = Settings(anthropic_api_key=SecretStr("sk-do-not-leak"))

    assert "sk-do-not-leak" not in repr(settings)
    assert "sk-do-not-leak" not in str(settings)
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-do-not-leak"


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
