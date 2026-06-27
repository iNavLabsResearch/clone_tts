"""Custom W&B-friendly callbacks: log eval perplexity alongside the metrics the
HF Trainer already reports (train/eval loss, learning_rate, train_samples_per_second,
global_step, epoch)."""
from __future__ import annotations

from transformers import TrainerCallback

from .metrics import perplexity_from_loss


class PerplexityCallback(TrainerCallback):
    """Adds eval_perplexity to the logged metrics after each evaluation."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        loss = metrics.get("eval_loss")
        if loss is not None:
            metrics["eval_perplexity"] = perplexity_from_loss(loss)
