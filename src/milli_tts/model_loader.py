"""Load Indic-Mio (Qwen3ForCausalLM) + tokenizer, and attach a PEFT LoRA adapter.

Tuned for Kaggle 2x T4 (Turing): fp16 (no bf16), SDPA attention (no FlashAttention-2),
gradient checkpointing on.
"""
from __future__ import annotations

import torch


def load_tokenizer(model_id: str, token: str | None = None):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_base_model(model_cfg: dict, token: str | None = None):
    from transformers import AutoModelForCausalLM

    dtype = torch.float16 if model_cfg.get("dtype", "float16") == "float16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_id"],
        torch_dtype=dtype,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        token=token,
    )
    if model_cfg.get("gradient_checkpointing", True):
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def attach_lora(model, train_cfg: dict):
    from peft import LoraConfig, get_peft_model

    if train_cfg.get("gradient_checkpointing", True):
        # Required so gradients flow to LoRA params when inputs are non-leaf under ckpt.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora = LoraConfig(
        r=train_cfg["lora_r"],
        lora_alpha=train_cfg["lora_alpha"],
        lora_dropout=train_cfg["lora_dropout"],
        target_modules=train_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def load_model_for_training(model_cfg: dict, train_cfg: dict, token: str | None = None):
    base = load_base_model(model_cfg, token=token)
    # Propagate gradient-checkpointing flag for attach_lora's input-grad fix.
    train_cfg = {**train_cfg, "gradient_checkpointing": model_cfg.get("gradient_checkpointing", True)}
    return attach_lora(base, train_cfg)


def load_model_for_inference(model_cfg: dict, adapter_dir: str | None = None, token: str | None = None):
    """Load base (optionally + LoRA adapter) for generation."""
    from transformers import AutoModelForCausalLM

    dtype = torch.float16 if model_cfg.get("dtype", "float16") == "float16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_id"],
        torch_dtype=dtype,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        token=token,
    )
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
    model.eval()
    return model
