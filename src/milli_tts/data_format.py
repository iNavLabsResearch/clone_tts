"""Turn a processed dataset row into a tokenized training example.

Indic-Mio uses plain ChatML:
    <|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n{<|s_..|> tokens}<|im_end|>\n

We compute the loss ONLY over the assistant (audio-token) span. We do this by tokenizing
the prompt (user turn + generation prompt) and the full conversation separately, then
masking the prompt portion of the labels to -100. This is robust to whatever the model's
chat_template emits, because the full sequence always starts with the prompt sequence.
"""
from __future__ import annotations

from .codec_utils import tokens_to_str


def build_messages(text: str, speech_token_str: str) -> list[dict]:
    return [
        {"role": "user", "content": text},
        {"role": "assistant", "content": speech_token_str},
    ]


def encode_example(tokenizer, text: str, speech_tokens: list[int], max_seq_len: int, speech_template: str = "<|s_{}|>"):
    """Return dict(input_ids, attention_mask, labels) or None if it doesn't fit / is empty."""
    if not text or not speech_tokens:
        return None

    speech_str = tokens_to_str(speech_tokens, template=speech_template)
    user_msg = [{"role": "user", "content": text}]
    full_msgs = build_messages(text, speech_str)

    prompt_ids = tokenizer.apply_chat_template(user_msg, tokenize=True, add_generation_prompt=True)
    full_ids = tokenizer.apply_chat_template(full_msgs, tokenize=True, add_generation_prompt=False)

    # Sanity: full must extend prompt. If template differs, fall back to a contains-check.
    prompt_len = len(prompt_ids)
    if full_ids[:prompt_len] != list(prompt_ids):
        # Conservative fallback: find the first divergence.
        prompt_len = 0
        for a, b in zip(prompt_ids, full_ids):
            if a != b:
                break
            prompt_len += 1

    if len(full_ids) > max_seq_len:
        return None  # skip over-long examples rather than truncate audio mid-utterance
    if prompt_len >= len(full_ids):
        return None  # no assistant tokens survived

    labels = [-100] * prompt_len + list(full_ids[prompt_len:])
    attention_mask = [1] * len(full_ids)
    return {
        "input_ids": list(full_ids),
        "attention_mask": attention_mask,
        "labels": labels,
    }
