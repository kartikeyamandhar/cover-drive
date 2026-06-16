"""Anthropic teacher client: generate stylized commentary from a structured prompt.

The client is injected (not constructed here) so tests mock it and ``make check``
never makes a live call. Prompt caching on the shared few-shot prefix is the lever
that makes the budget feasible; ``count_tokens`` gives an a-priori cost estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import anthropic
import structlog
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

from app.distill.errors import TeacherError
from app.distill.prompt import system_blocks, user_message
from configs.distill import (
    BATCH_MULT,
    CACHE_READ_MULT,
    CACHE_WRITE_MULT_1H,
    PRICING,
    DistillConfig,
)
from configs.personas import Persona, primary_persona

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class Generation:
    """One teacher completion plus its token accounting."""

    text: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int


@dataclass(frozen=True)
class CostEstimate:
    """An a-priori cost projection for a run."""

    n_calls: int
    seed_tokens: int
    per_call_uncached_input: int
    est_output_tokens: int
    est_usd: float
    assumes_cache_hits: bool
    assumes_batch: bool


def _uses_sampling_params(model: str) -> bool:
    """Sonnet/Haiku accept ``temperature``; Opus 4.7+/Fable reject it."""
    return model.startswith(("claude-sonnet", "claude-haiku"))


def _create_params(
    event: str,
    state: str,
    persona: Persona,
    config: DistillConfig,
    *,
    max_tokens: int | None = None,
) -> MessageCreateParamsNonStreaming:
    """Build the typed Messages params for one ball, shared by sync and batch."""
    params: MessageCreateParamsNonStreaming = {
        "model": config.teacher_model,
        "max_tokens": max_tokens if max_tokens is not None else config.max_output_tokens,
        "system": system_blocks(),
        "messages": [user_message(event, state, persona)],
    }
    if _uses_sampling_params(config.teacher_model):
        params["temperature"] = config.temperature
        params["thinking"] = {"type": "disabled"}
    return params


def build_message_params(
    event: str, state: str, persona: Persona, config: DistillConfig
) -> MessageCreateParamsNonStreaming:
    """Public builder for the Messages params (used by the batch path)."""
    return _create_params(event, state, persona, config)


def approx_seed_tokens() -> int:
    """Offline approximation of the cached prefix size (~chars/4), for dry-run only."""
    return sum(len(block["text"]) for block in system_blocks()) // 4


def _extract(message: anthropic.types.Message) -> Generation:
    """Pull the line and token accounting from a Message."""
    text = next((b.text for b in message.content if b.type == "text"), "").strip()
    usage = message.usage
    return Generation(
        text=text,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
        cache_write_tokens=usage.cache_creation_input_tokens or 0,
    )


def generate_one(
    client: anthropic.Anthropic, event: str, state: str, persona: Persona, config: DistillConfig
) -> Generation:
    """Generate a single commentary line synchronously."""
    params = _create_params(event, state, persona, config)
    try:
        message = client.messages.create(**params)
    except anthropic.APIError as exc:  # typed, surfaced
        raise TeacherError(f"teacher generation failed: {exc}") from exc
    return _extract(message)


def warm_cache(client: anthropic.Anthropic, config: DistillConfig) -> Generation:
    """Write the shared few-shot prefix to cache before a batch reads it.

    Sends one cheap request (1 output token) whose stable prefix is the same block
    every real request uses, so concurrent batch requests read rather than re-write.
    """
    primer = "T20 | Inns 1 | 0.1 | A 0/0 | CRR 0.0 | Striker X 0(0) | Bowler Y 0/0 | "
    primer += "P'ship 0(0) | Last 0 | powerplay"
    generation = generate_one_capped(client, "dot ball", primer, primary_persona(), config, 1)
    log.info("cache warmed", write_tokens=generation.cache_write_tokens)
    return generation


def generate_one_capped(
    client: anthropic.Anthropic,
    event: str,
    state: str,
    persona: Persona,
    config: DistillConfig,
    max_tokens: int,
) -> Generation:
    """Like ``generate_one`` but with an explicit ``max_tokens`` (used for warming)."""
    params = _create_params(event, state, persona, config, max_tokens=max_tokens)
    try:
        message = client.messages.create(**params)
    except anthropic.APIError as exc:
        raise TeacherError(f"teacher warm/generation failed: {exc}") from exc
    return _extract(message)


def count_seed_tokens(client: anthropic.Anthropic, config: DistillConfig) -> int:
    """Count the cached prefix size; warns the caller if it is below the cache floor."""
    params = _create_params("dot ball", "STATE", primary_persona(), config)
    result = client.messages.count_tokens(
        model=config.teacher_model,
        system=params["system"],
        messages=params["messages"],
    )
    return result.input_tokens


def generation_cost(gen: Generation, config: DistillConfig, *, batch: bool) -> float:
    """Actual USD cost of one completed generation, from its usage counters."""
    if config.teacher_model not in PRICING:
        raise TeacherError(f"no pricing for model {config.teacher_model!r}")
    in_price, out_price = PRICING[config.teacher_model]
    mult = BATCH_MULT if batch else 1.0
    input_usd = (
        gen.input_tokens * in_price
        + gen.cache_read_tokens * in_price * CACHE_READ_MULT
        + gen.cache_write_tokens * in_price * CACHE_WRITE_MULT_1H
    )
    output_usd = gen.output_tokens * out_price
    return (input_usd + output_usd) / 1e6 * mult


def estimate_cost(
    n_calls: int,
    seed_tokens: int,
    *,
    config: DistillConfig,
    avg_volatile_input: int = 80,
    avg_output: int = 60,
    assume_cache_hits: bool = True,
    assume_batch: bool = True,
) -> CostEstimate:
    """Project the USD cost of ``n_calls`` generations.

    Pure arithmetic over the pricing table, the cache multipliers, and the Batches
    discount; no API call. ``seed_tokens`` is the cached prefix size.
    """
    if config.teacher_model not in PRICING:
        raise TeacherError(f"no pricing for model {config.teacher_model!r}")
    in_price, out_price = PRICING[config.teacher_model]
    batch = BATCH_MULT if assume_batch else 1.0

    if assume_cache_hits:
        # One cache write, then every call reads the seed at ~0.1x.
        write_usd = seed_tokens * CACHE_WRITE_MULT_1H * in_price / 1e6
        seed_read_usd = n_calls * seed_tokens * CACHE_READ_MULT * in_price / 1e6 * batch
    else:
        # Worst case: every call pays the full uncached seed.
        write_usd = 0.0
        seed_read_usd = n_calls * seed_tokens * in_price / 1e6 * batch

    volatile_in_usd = n_calls * avg_volatile_input * in_price / 1e6 * batch
    out_usd = n_calls * avg_output * out_price / 1e6 * batch
    total = write_usd + seed_read_usd + volatile_in_usd + out_usd
    return CostEstimate(
        n_calls=n_calls,
        seed_tokens=seed_tokens,
        per_call_uncached_input=seed_tokens + avg_volatile_input,
        est_output_tokens=avg_output,
        est_usd=round(total, 2),
        assumes_cache_hits=assume_cache_hits,
        assumes_batch=assume_batch,
    )
