"""Build HF TrainingArguments + Trainer for LoRA finetuning on 2x T4 (DDP via Accelerate).

Loss:        built-in causal-LM cross-entropy over assistant (audio) tokens only
             (labels are -100 elsewhere; see data_format.encode_example).
Optimizer:   paged_adamw_8bit (bitsandbytes) — memory friendly on T4.
Scheduler:   cosine with warmup.
Regularize:  weight_decay, lora_dropout, gradient clipping (max_grad_norm).
Precision:   fp16 (T4 has no bf16).
Reporting:   W&B — train/eval loss, audio_token_accuracy, eval_perplexity, lr, steps/sec.
"""
from __future__ import annotations

import os

from transformers import Trainer, TrainingArguments

from .callbacks import PerplexityCallback
from .dataset import SpeechCollator
from .metrics import compute_metrics, preprocess_logits_for_metrics


def build_training_args(train_cfg: dict, report_to: str) -> TrainingArguments:
    # ddp_find_unused_parameters must be False for grad-checkpointing + DDP.
    return TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        optim=train_cfg["optim"],
        fp16=train_cfg.get("fp16", True),
        bf16=train_cfg.get("bf16", False),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        logging_steps=train_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 2),
        report_to=[report_to] if report_to else [],
        run_name=os.environ.get("WANDB_RUN_NAME"),
        seed=train_cfg.get("seed", 42),
        remove_unused_columns=False,
    )


def build_trainer(model, tokenizer, train_ds, eval_ds, train_cfg: dict, report_to: str = "wandb"):
    args = build_training_args(train_cfg, report_to)
    collator = SpeechCollator(pad_token_id=tokenizer.pad_token_id)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[PerplexityCallback()],
    )
    return trainer
