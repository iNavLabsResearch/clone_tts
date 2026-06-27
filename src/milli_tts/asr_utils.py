"""Gujarati ASR using AI4Bharat IndicConformer.

Model: ``ai4bharat/indic-conformer-600m-multilingual`` (loaded with trust_remote_code).
Usage from the model card:
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    text = model(waveform_16k, "gu", "ctc")   # also supports "rnnt"
IndicConformer expects 16 kHz mono audio.
"""
from __future__ import annotations

import re

import numpy as np
import torch

ASR_SAMPLE_RATE = 16000

# Keep Gujarati script, ASCII, common punctuation and the explicit pause/tag markers
# that the TTS model understands; strip other junk.
_KEEP_RE = re.compile(r"[^઀-૿0-9A-Za-z\s\.,!\?<>\[\]ઽ।॥ः]")
_WS_RE = re.compile(r"\s+")


class GujaratiASR:
    def __init__(self, model_id: str, device: str, language: str = "gu", decoding: str = "ctc"):
        self.model_id = model_id
        self.device = device
        self.language = language
        self.decoding = decoding
        self._model = None

    def load(self) -> None:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(self.model_id, trust_remote_code=True)
        try:
            model = model.to(self.device)
        except Exception:
            pass  # some remote-code models manage device internally
        model.eval()
        self._model = model

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("ASR model not loaded. Call .load() first.")
        return self._model

    @torch.no_grad()
    def transcribe(self, waveform_16k: torch.Tensor) -> str:
        """waveform_16k: 1-D float32 tensor sampled at 16 kHz."""
        wav = waveform_16k
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # (1, T) — IndicConformer expects a batch/channel dim
        wav = wav.to(self.device, dtype=torch.float32)
        out = self.model(wav, self.language, self.decoding)
        text = out[0] if isinstance(out, (list, tuple)) else out
        return clean_transcript(str(text))


def to_asr_waveform(array: np.ndarray, src_sr: int) -> torch.Tensor:
    """Convert raw audio array -> mono 16 kHz float32 tensor for ASR."""
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    wav = torch.from_numpy(arr)
    if src_sr != ASR_SAMPLE_RATE:
        import torchaudio.functional as AF

        wav = AF.resample(wav, orig_freq=src_sr, new_freq=ASR_SAMPLE_RATE)
    return wav.contiguous().float()


def clean_transcript(text: str) -> str:
    text = text.strip()
    text = _KEEP_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def is_valid_transcript(text: str, min_chars: int, max_chars: int) -> bool:
    if not text:
        return False
    n = len(text.strip())
    if n < min_chars or n > max_chars:
        return False
    # Require at least one Gujarati codepoint so we don't train on garbage/empty ASR.
    return any("઀" <= ch <= "૿" for ch in text)
