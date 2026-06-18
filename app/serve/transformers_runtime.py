"""The real serving runtime: the base model + the LoRA adapter via transformers + peft.

Runs only on a serving host (opt-in); excluded from CI and coverage because it needs torch
and the model weights. torch/transformers/peft are imported lazily inside the methods, so
importing this module on a CPU-only control plane is cheap and never the reason a test fails.

Device branch (the Phase 6 red-team's assumption-buster): bitsandbytes 4-bit is CUDA-only,
so CUDA may load the 4-bit base, while MPS/CPU load the canonical fp16 (or fp32) base and
apply the adapter. Real tokens are streamed via ``TextIteratorStreamer`` so the API can
replay them to the client.
"""

from __future__ import annotations

from collections.abc import Iterator
from threading import Thread
from typing import Any

from configs.serve import ServeConfig


def _resolve_device(pref: str) -> str:
    import torch

    if pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class TransformersPeftRuntime:
    """A ``RuntimeAdapter`` backed by the fine-tuned Qwen2.5 LoRA adapter."""

    def __init__(self, config: ServeConfig) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # transformers/peft are untyped under strict mypy; treat the callables as Any
        # (the same pattern as app/train/dataset.py).
        auto_model: Any = AutoModelForCausalLM
        auto_tok: Any = AutoTokenizer
        peft_model: Any = PeftModel

        self._cfg = config
        self._device = _resolve_device(config.device)
        used_device_map = self._device == "cuda" and config.load_in_4bit_on_cuda

        load_kwargs: dict[str, Any] = {}
        if used_device_map:
            from transformers import BitsAndBytesConfig

            bnb_config: Any = BitsAndBytesConfig
            load_kwargs["quantization_config"] = bnb_config(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            )
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float16 if self._device == "mps" else torch.float32

        base = auto_model.from_pretrained(config.base_model, **load_kwargs)
        model = peft_model.from_pretrained(base, config.adapter_repo)
        model.eval()
        if not used_device_map:
            model.to(self._device)

        tok = auto_tok.from_pretrained(config.tokenizer_id)
        tok.eos_token = config.eos_token
        tok.pad_token = config.pad_token

        self._model = model
        self._tok = tok

    def stream(self, system: str, user: str) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        streamer_cls: Any = TextIteratorStreamer
        enc = self._tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self._device)
        streamer = streamer_cls(self._tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs: dict[str, Any] = dict(
            **enc,
            max_new_tokens=self._cfg.max_new_tokens,
            do_sample=self._cfg.do_sample,
            temperature=self._cfg.temperature,
            top_p=self._cfg.top_p,
            pad_token_id=self._tok.pad_token_id,
            eos_token_id=self._tok.eos_token_id,
            streamer=streamer,
        )
        thread = Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()
        try:
            yield from streamer
        finally:
            thread.join()
