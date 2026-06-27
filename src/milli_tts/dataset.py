"""Load the processed HF dataset (milli_guj_dataset_artpark) and turn it into a
tokenized torch dataset + a padding collator with label masking.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .data_format import encode_example


def load_processed_dataset(repo: str, token: str | None, split: str = "train"):
    """Load all configs/shards of the processed dataset and concatenate into one split."""
    from datasets import concatenate_datasets, get_dataset_config_names, load_dataset

    try:
        config_names = get_dataset_config_names(repo, token=token)
    except Exception:
        config_names = None

    if config_names:
        parts = []
        for cfg in config_names:
            try:
                parts.append(load_dataset(repo, cfg, split=split, token=token))
            except Exception:
                continue
        if not parts:
            raise RuntimeError(f"Could not load any config of {repo}")
        return concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    return load_dataset(repo, split=split, token=token)


def tokenize_dataset(ds, tokenizer, max_seq_len: int, speech_template: str, num_proc: int = 1):
    """Map raw rows (text + speech_tokens) -> input_ids/labels, dropping ones that don't fit."""

    def _map_fn(batch):
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for text, tokens in zip(batch["text"], batch["speech_tokens"]):
            enc = encode_example(tokenizer, text, tokens, max_seq_len, speech_template)
            if enc is None:
                continue
            out["input_ids"].append(enc["input_ids"])
            out["attention_mask"].append(enc["attention_mask"])
            out["labels"].append(enc["labels"])
        return out

    keep_cols = {"text", "speech_tokens"}
    remove_cols = [c for c in ds.column_names if c not in keep_cols]
    tokenized = ds.map(
        _map_fn,
        batched=True,
        batch_size=64,
        num_proc=num_proc,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )
    # Filter empties produced by skipped rows.
    tokenized = tokenized.filter(lambda ex: len(ex["input_ids"]) > 0)
    return tokenized


@dataclass
class SpeechCollator:
    """Pad input_ids/labels to the longest sequence in the batch."""

    pad_token_id: int
    label_pad_id: int = -100

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attn, labels = [], [], []
        for f in features:
            ids = f["input_ids"]
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [self.label_pad_id] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
