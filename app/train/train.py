"""QLoRA fine-tune of Qwen2.5 with Unsloth + TRL (the pod-side entrypoint).

GPU libraries (unsloth, torch, trl, datasets) are imported LAZILY inside ``run_train``
so the Mac control plane and ``make check`` never need CUDA. ``run_check`` exercises
only the GPU-free data path (chat template + loss-mask span) for pre-spend validation.

Loss is masked to the assistant turn via ``train_on_responses_only`` -- the model is
graded on producing the commentary line, not on echoing the state.
"""

from __future__ import annotations

from typing import Any

import structlog

from app.train.dataset import CheckSummary, check, load_examples, split_for_monitoring
from configs.settings import get_settings
from configs.train import (
    QWEN_EOS_TOKEN,
    QWEN_INSTRUCTION_PART,
    QWEN_PAD_TOKEN,
    QWEN_RESPONSE_PART,
    TrainConfig,
)

log = structlog.get_logger(__name__)


def run_check(config: TrainConfig) -> CheckSummary:
    """No-GPU validation of the data path (chat template, loss-mask span, token lengths)."""
    return check(config)


def _accepted_kwarg(target: Any, preferred: str, fallback: str) -> str:
    """Return whichever kwarg name ``target`` accepts. Prefers the new name; uses the old
    one only if it is named and the new one is not (Unsloth-shimmed TRL). Lets the trainer
    run on both the current TRL API and the older/shimmed one."""
    import inspect

    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return preferred
    if fallback in params and preferred not in params:
        return fallback
    return preferred


def _assert_loss_mask(trainer: Any) -> None:
    """Fail fast if completion-only masking did not take: a real run must have BOTH
    masked (-100) and trained labels. Defensive -- if TRL internals differ and the
    introspection itself fails, warn and continue (the data --check already verified the
    marker at the string level)."""
    try:
        batch = trainer.data_collator([trainer.train_dataset[0]])
        labels = batch["labels"][0]
        trained = int((labels != -100).sum())
        masked = int((labels == -100).sum())
    except Exception as exc:  # introspection is best-effort, never blocks a good run
        log.warning("loss-mask self-check skipped", error=str(exc))
        return
    log.info("loss-mask check", trained_tokens=trained, masked_tokens=masked)
    if trained == 0 or masked == 0:
        raise SystemExit(
            f"loss mask wrong: trained={trained} masked={masked} (expected both > 0; "
            "the response marker did not match the chat template)"
        )


def run_train(config: TrainConfig) -> None:
    """Rent-side QLoRA training. Imports GPU libraries lazily; saves + pushes the adapter
    and (optionally) a merged GGUF. Idempotent enough to resume from the saved adapter."""
    import torch  # noqa: F401  (Unsloth needs torch initialized)
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    settings = get_settings()
    hf_token = settings.hf_token.get_secret_value() if settings.hf_token else None
    # An unset WANDB_API_KEY arrives as SecretStr('') (truthy object, empty value); check
    # the INNER value, or a blank key resolves to report_to="wandb" and hangs a billed pod
    # on an interactive login prompt.
    wandb_key = settings.wandb_api_key.get_secret_value() if settings.wandb_api_key else None
    report_to = "wandb" if wandb_key else "none"

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model,
        max_seq_length=config.max_seq_len,
        load_in_4bit=True,
        dtype=None,
    )
    # The 4-bit repo ships an incomplete special-token config (no pad token; TRL's SFTConfig
    # then defaults eos to a '<EOS_TOKEN>' sentinel and rejects it). The Qwen2.5 chat template
    # ends every turn with <|im_end|>, so that IS the eos; pin the real tokens.
    tokenizer.eos_token = QWEN_EOS_TOKEN
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = QWEN_PAD_TOKEN
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_targets),
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )

    examples = load_examples(config, source="hub")
    train_ex, eval_ex = split_for_monitoring(examples, config.eval_match_fraction)
    log.info("dataset loaded", total=len(examples), train=len(train_ex), eval=len(eval_ex))

    def to_text(example: dict[str, object]) -> dict[str, str]:
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    train_ds = Dataset.from_list(train_ex).map(to_text)
    eval_ds = Dataset.from_list(eval_ex).map(to_text) if eval_ex else None

    # TRL renamed kwargs across versions (max_seq_length->max_length, tokenizer->
    # processing_class) and Unsloth shims the OLD names on new TRL. Reviewers disagreed on
    # which is correct, so pick whichever the INSTALLED version actually accepts -- this
    # runs on either API without a first-run TypeError that would waste GPU minutes.
    seq_kw = _accepted_kwarg(SFTConfig, "max_length", "max_seq_length")
    tok_kw = _accepted_kwarg(SFTTrainer, "processing_class", "tokenizer")
    sft_kwargs: dict[str, object] = {
        "dataset_text_field": "text",
        "per_device_train_batch_size": config.per_device_batch,
        "gradient_accumulation_steps": config.grad_accum,
        "warmup_ratio": config.warmup_ratio,
        "num_train_epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "logging_steps": 5,
        "eval_strategy": "epoch" if eval_ds is not None else "no",
        # Keep each epoch's checkpoint and restore the best one, so an over-trained final
        # epoch (a real risk on 4k narrow-domain examples) is recoverable.
        "save_strategy": "epoch" if eval_ds is not None else "no",
        "load_best_model_at_end": eval_ds is not None,
        "metric_for_best_model": "eval_loss",
        "save_total_limit": 2,
        "optim": "adamw_8bit",
        "weight_decay": config.weight_decay,
        "lr_scheduler_type": config.lr_scheduler,
        "seed": config.seed,
        "output_dir": str(config.output_dir),
        "report_to": report_to,
        seq_kw: config.max_seq_len,
    }
    # TRL/Unsloth turns a None eos_token into a '<EOS_TOKEN>' placeholder it then rejects.
    # Passing the real Qwen eos into the CONSTRUCTOR sticks (a post-hoc setattr does not
    # survive SFTTrainer's arg processing). Guard for an older TRL that lacks the arg.
    try:
        sft_args = SFTConfig(eos_token=QWEN_EOS_TOKEN, **sft_kwargs)
    except TypeError:
        sft_args = SFTConfig(**sft_kwargs)

    # ROOT CAUSE of the '<EOS_TOKEN>' error: transformers TrainingArguments.to_dict()
    # REDACTS any field whose name contains "token" (a guard for hub_token secrets),
    # turning eos_token='<|im_end|>' into the literal '<EOS_TOKEN>'. Unsloth's patched
    # SFTTrainer round-trips args through to_dict(), so the redacted value reaches TRL's
    # eos validator (tokenizer.convert_tokens_to_ids('<EOS_TOKEN>') is None -> ValueError).
    # Patch to_dict to restore the real, non-secret token fields so they survive the trip.
    _orig_to_dict = type(sft_args).to_dict

    def _to_dict_keep_tokens(self: Any) -> dict[str, Any]:
        out: dict[str, Any] = _orig_to_dict(self)
        for key in ("eos_token", "pad_token", "bos_token"):
            value = getattr(self, key, None)
            if value is not None:
                out[key] = value
        return out

    type(sft_args).to_dict = _to_dict_keep_tokens  # type: ignore[method-assign]

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_args,
        **{tok_kw: tokenizer},
    )
    # Mask the loss to the assistant turn only.
    trainer = train_on_responses_only(
        trainer,
        instruction_part=QWEN_INSTRUCTION_PART,
        response_part=QWEN_RESPONSE_PART,
    )
    _assert_loss_mask(trainer)

    trainer.train()

    adapter_dir = config.output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    log.info("adapter saved", path=str(adapter_dir))

    if hf_token is not None:
        # Push the adapter FIRST: it is the irreplaceable training output. The GGUF is a
        # convenience export that builds llama.cpp on the fly and can fail on a thin image;
        # if it does, the adapter is already safe and the GGUF can be rebuilt from it.
        model.push_to_hub(config.adapter_repo, token=hf_token)
        tokenizer.push_to_hub(config.adapter_repo, token=hf_token)
        log.info("adapter pushed", repo=config.adapter_repo)
        if config.push_gguf:
            try:
                model.push_to_hub_gguf(
                    config.gguf_repo,
                    tokenizer,
                    quantization_method=config.gguf_quant,
                    token=hf_token,
                )
                log.info("gguf pushed", repo=config.gguf_repo)
            except Exception as exc:  # GGUF is optional; never lose the adapter over it
                log.warning("gguf export failed; adapter is safe", error=str(exc))
    else:
        log.warning("HF_TOKEN not set; adapter saved locally but not pushed")
