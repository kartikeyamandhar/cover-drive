#!/usr/bin/env bash
# Run the QLoRA fine-tune on the pod. Run from the repo root AFTER setup.sh.
# Watch this terminal (loss every 5 steps) and/or the W&B run URL it prints.
set -euo pipefail

: "${HF_TOKEN:?set HF_TOKEN (WRITE-scoped) before training -- needed to push the adapter}"
# W&B is OPT-IN. Do NOT force-export an empty key: an empty WANDB_API_KEY would still
# select the wandb logger and hang this billed pod on a login prompt. Only export when set.
if [ -z "${WANDB_API_KEY:-}" ]; then export WANDB_DISABLED=true; fi
# Put the HF cache on the network volume so the base model is not re-downloaded next pod.
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo ">> training (data path was validated locally with: python -m scripts.train --check)"
python -m scripts.train --train --json-logs

echo ">> done. Verify the adapter + GGUF on the Hub, then TEAR DOWN THE POD (see LAUNCH.txt)."
