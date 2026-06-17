"""Phase 2 distillation configuration: teacher, sampling mix, budget, paths.

A plain typed config (not settings); the secret (``ANTHROPIC_API_KEY``) is read
separately via ``configs.settings``. Cricket-rule constants for bucketing live in
``configs.data``; this file owns the generation-side knobs.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

# Per-1M-token USD pricing by model (input, output). Source: Anthropic pricing.
PRICING: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0),
}

# Cost multipliers relative to base input price.
CACHE_READ_MULT = 0.1
CACHE_WRITE_MULT_1H = 2.0
BATCH_MULT = 0.5

# Minimum cacheable prefix (tokens) by model. A shorter prefix silently won't cache.
CACHE_MIN_TOKENS: dict[str, int] = {
    "claude-haiku-4-5": 4096,
    "claude-sonnet-4-6": 2048,
    "claude-opus-4-8": 4096,
}

# Measured size of the cached few-shot prefix (system + seed), from a live
# cache-write (usage.cache_creation_input_tokens). Far more accurate than a
# chars/4 approximation. Re-measure and update when the seed changes.
# 7917 after the boundary prompt fix (state-is-after-the-ball rules); was 7541.
MEASURED_SEED_TOKENS = 7917


class StratBucket(BaseModel):
    """One situation bucket and its target share of the stratified set."""

    model_config = ConfigDict(frozen=True)

    name: str
    target_fraction: float


# First-match priority order: a ball is assigned to the FIRST bucket it matches,
# so the fractions form a real partition. Rare high-information buckets (wicket,
# hat_trick) come first so they are actually captured rather than claimed by a
# commoner bucket. Definitions are implemented in app/distill/sampling.py and must
# stay in sync with this order. Fractions sum to 1.0.
STRATIFICATION: tuple[StratBucket, ...] = (
    StratBucket(name="wicket", target_fraction=0.18),
    StratBucket(name="hat_trick", target_fraction=0.02),
    StratBucket(name="four", target_fraction=0.15),
    StratBucket(name="death_chase_pressure", target_fraction=0.12),
    StratBucket(name="six", target_fraction=0.12),
    StratBucket(name="batter_milestone", target_fraction=0.08),
    StratBucket(name="dot_under_pressure", target_fraction=0.07),
    StratBucket(name="powerplay_routine", target_fraction=0.07),
    StratBucket(name="tight_finish", target_fraction=0.06),
    StratBucket(name="team_milestone", target_fraction=0.05),
    StratBucket(name="middle_routine", target_fraction=0.05),
    StratBucket(name="death_routine", target_fraction=0.03),
)


class DistillConfig(BaseModel):
    """Generation-side configuration for Phase 2."""

    model_config = ConfigDict(frozen=True)

    teacher_model: str = "claude-sonnet-4-6"
    max_output_tokens: int = 90
    temperature: float = 0.95  # only sent to models that accept sampling params

    # Coverage: the primary persona gets the full stratified set; each secondary
    # persona re-renders a subset of the SAME balls (for the per-ball A/B demo).
    # Sized so the full plan (~8.7k calls) completes within budget in batch mode
    # at the measured per-call cost; the budget guard is the hard backstop.
    primary_set_size: int = 5000
    secondary_subset_fraction: float = 0.25

    # Budget guard (USD). The run stops before exceeding this.
    budget_usd_cap: float = 18.0

    # Match split (shared with Phase 3): held-out matches are never narrated here.
    val_fraction: float = 0.1
    test_fraction: float = 0.1

    sampling_seed: int = 17
    cache_ttl: str = "1h"

    data_dir: Path = Path("data")

    @property
    def processed_dir(self) -> Path:
        """Where the Phase 1 per-ball JSONL lives."""
        return self.data_dir / "processed"

    @property
    def distill_dir(self) -> Path:
        """Where the synthetic SFT pairs are written (git-ignored)."""
        return self.data_dir / "distill"

    @property
    def dataset_dir(self) -> Path:
        """Where the assembled train/val/test SFT JSONL is written (git-ignored)."""
        return self.data_dir / "dataset"

    @property
    def manifest_path(self) -> Path:
        """Resumable generation state."""
        return self.distill_dir / "manifest.json"

    def stratification(self) -> tuple[StratBucket, ...]:
        """The situation mix used for sampling."""
        return STRATIFICATION
