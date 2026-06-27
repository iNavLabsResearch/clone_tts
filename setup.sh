#!/usr/bin/env bash
# Environment setup for Kaggle 2x T4 (Turing) finetuning of Indic-Mio.
#
# Secrets are read from the environment (or secrets.env / Kaggle Secrets) — never hardcoded.
# Required: HF_TOKEN (with accepted access to ARTPARK-IISc/Vaani), WANDB_API_KEY (optional).
#
# Usage:
#   export HF_TOKEN=hf_xxx   WANDB_API_KEY=xxx
#   bash setup.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# Load secrets.env if present (gitignored).
if [[ -f secrets.env ]]; then
  echo "[setup] sourcing secrets.env"
  set -a; # shellcheck disable=SC1091
  source secrets.env; set +a
fi

echo "[setup] python: $(python --version 2>&1)"

# Torch/torchaudio are preinstalled on Kaggle; only install if missing.
python - <<'PY'
import importlib.util, sys
missing = [m for m in ("torch", "torchaudio") if importlib.util.find_spec(m) is None]
print("MISSING_TORCH=" + ",".join(missing))
PY

echo "[setup] installing python deps ..."
pip install -q -r requirements.txt
pip install -q "git+https://github.com/Aratako/MioCodec"

# IndicConformer remote code may need NeMo. Best-effort (heavy; safe to skip if it fails).
echo "[setup] installing nemo_toolkit[asr] (best-effort for IndicConformer) ..."
pip install -q "nemo_toolkit[asr]" || echo "[setup] WARN: nemo install failed; IndicConformer may still load via its HF remote code."

# Hugging Face login (token from env). Vaani is gated -> token must have accepted access.
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "[setup] logging into Hugging Face Hub"
  python - <<'PY'
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("[setup] HF login OK")
PY
else
  echo "[setup] WARN: HF_TOKEN not set — Vaani (gated) and pushes will fail."
fi

# W&B login (optional).
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "[setup] W&B key detected (will be used by train.py)."
else
  echo "[setup] NOTE: WANDB_API_KEY not set — training will run with W&B disabled."
fi

echo "[setup] verifying imports ..."
python - <<'PY'
import torch, transformers, peft, accelerate, datasets
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("transformers", transformers.__version__)
try:
    import miocodec; print("miocodec OK")
except Exception as e:
    print("miocodec import FAILED:", e)
PY

echo "[setup] done."
