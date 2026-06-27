#!/usr/bin/env bash
# Kaggle one-shot: install deps + run Gujarati Vaani db_prep on 2x T4.
#
# Secrets come from config.json (huggingface.token, wandb.api_key); optional
# secrets.env overrides. Vaani is gated — HF token must have accepted access.
#
# Usage (from repo root on Kaggle):
#   bash scripts/kaggle_setup_and_db_prep.sh
#
# Dry-run one config (no Hub push):
#   bash scripts/kaggle_setup_and_db_prep.sh --config Gujarat_Valsad --limit 8 --no-push
#
# Any extra args are forwarded to db_prep.py (both workers get them).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

echo "=============================================="
echo " Kaggle 2x T4 — setup + db_prep"
echo " repo: $HERE"
echo " gpus: $(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ') visible"
echo "=============================================="

echo ""
echo "========== [1/2] SETUP =========="
bash setup.sh

echo ""
echo "========== [2/2] DB PREP (worker 0 -> GPU0, worker 1 -> GPU1) =========="
bash scripts/run_db_prep_2gpu.sh "$@"

echo ""
echo "[done] setup + db_prep finished."
