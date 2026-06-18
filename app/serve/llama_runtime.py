"""The CPU serving runtime: a quantized GGUF of the fine-tune via llama-cpp-python.

This is what the free Hugging Face CPU Space runs. Excluded from CI and coverage (it needs
the GGUF and a compiled llama.cpp); ``llama_cpp`` is imported lazily so importing this module
on the control plane is cheap. It implements the ``RuntimeAdapter`` protocol, so the FastAPI
app is unchanged. Qwen2.5 speaks ChatML, hence ``chat_format="chatml"``.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

from configs.serve import ServeConfig


class LlamaCppRuntime:
    """A ``RuntimeAdapter`` backed by the quantized GGUF, served on CPU.

    A single ``Llama``/``llama_context`` is NOT thread-safe: concurrent inference on it
    corrupts state and aborts the process (a GGML_ASSERT). The engine consumes each draw
    fully, so a lock held across the generation serializes requests (they queue rather than
    collide) at no extra cost on a single-CPU box.
    """

    def __init__(self, config: ServeConfig) -> None:
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        model_path = hf_hub_download(repo_id=config.gguf_repo, filename=config.gguf_file)
        llama_cls: Any = Llama  # llama_cpp is untyped under strict mypy
        self._cfg = config
        self._lock = threading.Lock()
        self._llm = llama_cls(
            model_path=model_path,
            n_ctx=config.llama_ctx,
            n_threads=config.llama_threads,
            chat_format="chatml",
            verbose=False,
        )

    def stream(self, system: str, user: str) -> Iterator[str]:
        with self._lock:
            chunks = self._llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=self._cfg.max_new_tokens,
                temperature=self._cfg.temperature,
                top_p=self._cfg.top_p,
                stream=True,
            )
            for chunk in chunks:
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
