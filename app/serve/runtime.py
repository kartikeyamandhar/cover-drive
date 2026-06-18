"""The serving runtime seam: a protocol the API depends on, not a model.

The API and the commentary engine talk to a ``RuntimeAdapter``; they never import torch
or transformers. Two backends implement it: ``TransformersPeftRuntime`` (the real model,
base + LoRA adapter, added in T6.6 with torch imported lazily so this module stays
GPU-free) and ``StubRuntime`` (deterministic, no model) so the whole service is testable
headless. An Ollama backend can be added behind the same protocol without touching the API.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class RuntimeAdapter(Protocol):
    """Streams a model's reply to a ``(system, user)`` prompt, chunk by chunk."""

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield decoded text chunks whose concatenation is the model's full reply."""
        ...


def token_chunks(line: str) -> list[str]:
    """Split a line into chunks that re-join to the original.

    Used by ``StubRuntime`` to fake a token stream, and by the SSE layer to replay a
    validated line to the client token by token (the client only ever sees checked text).
    """
    words = line.split(" ")
    return [word if i == 0 else " " + word for i, word in enumerate(words)]


class StubRuntime:
    """A deterministic ``RuntimeAdapter`` for tests: no model, no GPU, no network.

    Configured with either a fixed line or a callable ``(system, user) -> line`` so a test
    can simulate a faithful or an unfaithful draw per prompt. Emits the line as a fake token
    stream (chunks that concatenate back to the line), exercising the streaming code path.
    """

    def __init__(self, line: str | Callable[[str, str], str]) -> None:
        self._line = line

    def stream(self, system: str, user: str) -> Iterator[str]:
        line = self._line(system, user) if callable(self._line) else self._line
        yield from token_chunks(line)
