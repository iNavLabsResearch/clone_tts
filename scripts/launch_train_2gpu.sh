#!/usr/bin/env bash
# Launch LoRA finetuning across 2x T4 with Accelerate DDP.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
[[ -f secrets.env ]] && { set -a; source secrets.env; set +a; }

accelerate launch \
  --config_file accelerate_config_2xt4.yaml \
  --num_processes 2 \
  train.py "$@"
