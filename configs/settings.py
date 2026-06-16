"""Application settings, read from the environment / ``.env`` via pydantic-settings.

This is the single configuration entrypoint for the project. Per CLAUDE.md
Section 7: configuration is read only from ``.env``, no secrets live in code, and
no secret is ever logged. Secrets are typed as ``SecretStr`` so their values do
not leak through ``repr``/``str`` or structured log rendering.

Phase 0 requires no secrets: every field is optional with a safe default, so the
process boots with an empty environment. Later phases consume these fields.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly typed runtime configuration.

    Field names map case-insensitively to environment variables (for example
    ``hf_token`` reads ``HF_TOKEN``). Unknown environment variables are ignored
    so the process is not coupled to the exact contents of a shared ``.env``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Non-secret runtime knobs.
    app_env: str = "dev"
    log_json: bool = False

    # Secrets, wired in later phases (Phase 2 teacher, Phase 4 RunPod/HF, etc.).
    # Optional so Phase 0 needs none; never printed, never committed.
    hf_token: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    kaggle_username: str | None = None
    kaggle_key: SecretStr | None = None
    runpod_api_key: SecretStr | None = None
    wandb_api_key: SecretStr | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton, loaded once and cached."""
    return Settings()
