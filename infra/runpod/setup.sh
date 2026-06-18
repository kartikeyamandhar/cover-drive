#!/usr/bin/env bash
# One-time dependency install on the RunPod pod. Run from the repo root.
# Assumes a RunPod PyTorch CUDA image (torch + CUDA already present).
set -euo pipefail

echo ">> GPU:"
nvidia-smi -L

echo ">> disk (need ~>=15GB free on / for weights + any GGUF merge):"
df -h / /workspace 2>/dev/null || df -h /

echo ">> build tools (only needed if you later enable the GGUF export; best-effort):"
(apt-get update -qq && apt-get install -y -qq build-essential cmake git) || \
  echo "   (apt unavailable; fine unless you enable push_gguf, which builds llama.cpp)"

echo ">> clearing any preview torch/unsloth so the torch-matched build installs clean..."
pip uninstall -y -q torch torchvision torchaudio unsloth unsloth_zoo trl xformers 2>/dev/null || true

echo ">> installing GPU + repo deps (unsloth pinned to the image's torch 2.4.1+cu124)..."
pip install -q --upgrade pip
pip install -q -r infra/runpod/requirements-pod.txt

echo ">> sanity: imports + CUDA visible to torch"
python - <<'PY'
import torch, unsloth, trl  # noqa: F401
assert torch.cuda.is_available(), "CUDA not available to torch"
print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0))
PY

echo ">> setup done."
