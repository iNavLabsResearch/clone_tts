#!/usr/bin/env bash
# Run dataset prep across both T4s: one worker per GPU, splitting rows by index.
# Each worker pushes its own shard configs, so there is no merge step.
#
# Usage: bash scripts/run_db_prep_2gpu.sh [--config Gujarat_Valsad] [extra db_prep args...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
[[ -f secrets.env ]] && { set -a; source secrets.env; set +a; }

echo "[db_prep] worker 0 -> GPU0, worker 1 -> GPU1"
CUDA_VISIBLE_DEVICES=0 python db_prep.py --worker-id 0 --num-workers 2 --gpu cuda:0 "$@" &
PID0=$!
CUDA_VISIBLE_DEVICES=1 python db_prep.py --worker-id 1 --num-workers 2 --gpu cuda:0 "$@" &
PID1=$!

wait $PID0; wait $PID1
echo "[db_prep] both workers finished."
