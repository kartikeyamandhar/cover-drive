"""Phase 1 run configuration: data paths, acquisition limits, feature parameters.

A plain typed config model (not settings): defaults are sensible for the IPL data
spine and every value is overridable. Cricket-rule constants (phase boundaries,
milestone windows) live here so they are visible and testable, not buried in code.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class FeatureConfig(BaseModel):
    """Tunable constants for match-state featurization."""

    model_config = ConfigDict(frozen=True)

    balls_per_over_default: int = 6
    powerplay_overs: int = 6
    death_overs_from_end: int = 5
    last_n_deliveries: int = 6
    batter_milestone_window: int = 10
    team_milestone_step: int = 50
    team_milestone_window: int = 10


class DataConfig(BaseModel):
    """Acquisition and dataset-build configuration for the IPL data spine."""

    model_config = ConfigDict(frozen=True)

    archive_url: str = "https://cricsheet.org/downloads/ipl_json.zip"
    allowed_hosts: frozenset[str] = Field(default_factory=lambda: frozenset({"cricsheet.org"}))
    data_dir: Path = Path("data")

    # Acquisition safety caps. IPL archive is ~5 MB compressed, ~150 MB extracted,
    # ~1.3k files; caps are generous multiples that still stop a runaway archive.
    max_download_bytes: int = 200 * 1024 * 1024
    max_uncompressed_bytes: int = 2 * 1024 * 1024 * 1024
    max_entries: int = 50_000

    # Selection (applied at build time). None/empty means all extracted matches.
    match_ids: tuple[str, ...] = ()
    season: str | None = None
    limit: int | None = None

    features: FeatureConfig = Field(default_factory=FeatureConfig)

    @property
    def raw_dir(self) -> Path:
        """Archives and extracted match JSON live here (git-ignored scratch)."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Per-match JSONL output lives here."""
        return self.data_dir / "processed"

    @property
    def manifest_path(self) -> Path:
        """Persisted job state for resumability."""
        return self.data_dir / "manifest.json"
