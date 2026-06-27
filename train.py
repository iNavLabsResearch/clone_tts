#!/usr/bin/env python
"""Finetune Indic-Mio (LoRA) on the processed Gujarati dataset.

Launch on 2x T4 via Accelerate DDP:
    accelerate launch --config_file accelerate_config_2xt4.yaml --num_processes 2 train.py

W&B logs train/eval loss, audio_token_accuracy, eval_perplexity, learning_rate, and
train_samples_per_second (steps/sec) automatically.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from milli_tts.config import load_config, resolve_dataset_repo_id, resolve_secrets  # noqa: E402
from milli_tts.dataset import load_processed_dataset, tokenize_dataset  # noqa: E402
from milli_tts.model_loader import load_model_for_training, load_tokenizer  # noqa: E402
from milli_tts.trainer import build_trainer  # noqa: E402


def is_main_process() -> bool:
    return os.environ.get("RANK", "0") in ("0", "") and os.environ.get("LOCAL_RANK", "0") in ("0", "")


def setup_wandb(cfg, secrets) -> str:
    wb = cfg["wandb"]
    if not wb.get("enabled", False) or not secrets.wandb_api_key:
        if wb.get("enabled", False) and not secrets.wandb_api_key:
            print("[wandb] enabled but no WANDB_API_KEY found -> disabling W&B logging.")
        return ""
    os.environ.setdefault("WANDB_API_KEY", secrets.wandb_api_key)
    os.environ["WANDB_PROJECT"] = wb["project"]
    run_name = wb["run_name"].replace("{timestamp}", time.strftime("%Y%m%d-%H%M%S"))
    os.environ["WANDB_RUN_NAME"] = run_name
    os.environ["WANDB_NAME"] = run_name
    if not wb.get("log_model", False):
        os.environ["WANDB_LOG_MODEL"] = "false"
    return "wandb"


def main() -> int:
    cfg = load_config()
    secrets = resolve_secrets(cfg)
    train_cfg = dict(cfg["train"])
    train_cfg["seed"] = cfg.get("seed", 42)

    from transformers import set_seed

    set_seed(train_cfg["seed"])

    report_to = setup_wandb(cfg, secrets)

    # Resolve which processed dataset repo to read.
    dataset_repo = (
        train_cfg.get("dataset_repo")
        or os.environ.get("DATASET_REPO")
        or resolve_dataset_repo_id(cfg, secrets.hf_token)
    )
    print(f"[data] loading processed dataset: {dataset_repo}")

    tokenizer = load_tokenizer(cfg["model"]["base_model_id"], token=secrets.hf_token)
    raw = load_processed_dataset(dataset_repo, token=secrets.hf_token)
    print(f"[data] {len(raw)} processed rows")

    tokenized = tokenize_dataset(
        raw, tokenizer,
        max_seq_len=cfg["model"]["max_seq_len"],
        speech_template=cfg["codec"]["speech_token_template"],
        num_proc=train_cfg.get("dataloader_num_workers", 1),
    )
    split = tokenized.train_test_split(test_size=cfg["train"]["eval_ratio"], seed=train_cfg["seed"])
    train_ds, eval_ds = split["train"], split["test"]
    print(f"[data] train={len(train_ds)} eval={len(eval_ds)}")

    model = load_model_for_training(cfg["model"], train_cfg, token=secrets.hf_token)

    trainer = build_trainer(model, tokenizer, train_ds, eval_ds, train_cfg, report_to=report_to)
    trainer.train()

    out_dir = train_cfg["output_dir"]
    trainer.save_model(out_dir)            # saves LoRA adapter
    tokenizer.save_pretrained(out_dir)
    print(f"[done] adapter + tokenizer saved to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
