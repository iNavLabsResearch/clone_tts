"""MioCodec helpers: load the codec, encode audio -> content tokens + global embedding,
decode tokens -> waveform, and convert between codec integer ids and the ``<|s_N|>``
string form the Qwen3 LLM consumes/produces.

MioCodec API (github.com/Aratako/MioCodec):
    m = MioCodecModel.from_pretrained(model_id).eval().to(device)
    feat = m.encode(wav, return_content=True, return_global=True)
        feat.content_token_indices -> LongTensor (seq_len,)  values in [0, 12800)
        feat.global_embedding      -> FloatTensor (dim,)     speaker/acoustic vector
    wav = m.decode(global_embedding=..., content_token_indices=...)  # (num_samples,)
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import torch

# Matches the audio tokens emitted by the LLM, e.g. "<|s_1234|>".
SPEECH_TOKEN_PATTERN = re.compile(r"<\|s_(\d+)\|>")


def load_codec(model_id: str, device: str):
    """Load a MioCodec model onto ``device`` (e.g. 'cuda', 'cuda:0', 'cpu')."""
    from miocodec import MioCodecModel  # imported lazily so the package imports without CUDA deps

    codec = MioCodecModel.from_pretrained(model_id)
    codec = codec.eval().to(device)
    return codec


def codec_sample_rate(codec) -> int:
    return int(codec.config.sample_rate)


def tokens_to_str(tokens: Iterable[int], template: str = "<|s_{}|>") -> str:
    """[12, 5, 900] -> '<|s_12|><|s_5|><|s_900|>'."""
    return "".join(template.format(int(t)) for t in tokens)


def parse_speech_tokens(text: str) -> list[int]:
    """Extract the integer codec ids from an LLM output string."""
    return [int(v) for v in SPEECH_TOKEN_PATTERN.findall(text)]


@torch.no_grad()
def encode_audio(codec, waveform: torch.Tensor, device: str):
    """Encode a mono waveform tensor (already at the codec sample rate) into
    (content_token_indices: list[int], global_embedding: np.ndarray[float32])."""
    wav = waveform.to(device=device, dtype=torch.float32)
    if wav.dim() > 1:
        wav = wav.squeeze()
    feat = codec.encode(wav, return_content=True, return_global=True)
    tokens = feat.content_token_indices.detach().to("cpu").long().tolist()
    gemb = feat.global_embedding.detach().to("cpu").float().numpy().reshape(-1)
    return tokens, gemb


@torch.no_grad()
def decode_tokens(codec, tokens: list[int] | torch.Tensor, global_embedding, device: str) -> torch.Tensor:
    """Decode codec ids + a global embedding back into a 1-D waveform tensor (on CPU)."""
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.to(device=device, dtype=torch.long)

    if isinstance(global_embedding, np.ndarray):
        global_embedding = torch.from_numpy(global_embedding)
    elif not isinstance(global_embedding, torch.Tensor):
        global_embedding = torch.tensor(global_embedding)
    global_embedding = global_embedding.squeeze().to(device=device, dtype=torch.float32)

    wav = codec.decode(global_embedding=global_embedding, content_token_indices=tokens)
    return wav.detach().to("cpu").reshape(-1)


def load_audio_for_codec(codec, path: str) -> torch.Tensor:
    """Load + resample an audio file to the codec's sample rate using MioCodec's loader."""
    from miocodec import load_audio

    return load_audio(path, sample_rate=codec_sample_rate(codec))


def waveform_from_row(audio_field, target_sr: int) -> torch.Tensor:
    """Convert a HF datasets ``audio`` field ({'array','sampling_rate'}) into a mono
    torch waveform resampled to ``target_sr``. Returns a 1-D float32 tensor."""
    array = np.asarray(audio_field["array"], dtype=np.float32)
    src_sr = int(audio_field["sampling_rate"])
    if array.ndim > 1:  # stereo -> mono
        array = array.mean(axis=-1)
    wav = torch.from_numpy(array)
    if src_sr != target_sr:
        import torchaudio.functional as AF

        wav = AF.resample(wav, orig_freq=src_sr, new_freq=target_sr)
    return wav.contiguous().float()
