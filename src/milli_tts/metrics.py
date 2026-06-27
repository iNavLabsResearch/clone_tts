"""Evaluation metrics for the TTS-as-LM task: audio-token accuracy + perplexity.

We use ``preprocess_logits_for_metrics`` to reduce logits to argmax predictions on-GPU
(the vocab is 164k, so keeping full logits would OOM a T4). ``compute_metrics`` then
compares predictions to labels over the assistant (non -100) positions only.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    preds, labels = eval_pred  # preds: (B, T) argmax ids; labels: (B, T)
    preds = np.asarray(preds)
    labels = np.asarray(labels)

    # Causal LM shift: prediction at position t targets label at t+1.
    shift_preds = preds[:, :-1]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100

    correct = (shift_preds == shift_labels) & mask
    total = mask.sum()
    accuracy = float(correct.sum()) / float(total) if total > 0 else 0.0
    return {"audio_token_accuracy": accuracy}


def perplexity_from_loss(loss: float) -> float:
    try:
        return float(math.exp(loss))
    except OverflowError:
        return float("inf")
