"""Build per-speaker voice presets from the processed dataset.

In MioTTS the speaker timbre is the codec ``global_embedding`` (a vector), applied at
decode time. To "speak as" a Vaani speaker we average that speaker's per-utterance
global_embeddings into one preset and save it as ``presets/<speakerID>.pt`` (a 1-D
float tensor), which is exactly the format MioTTS-Inference presets use.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def build_speaker_presets(ds, presets_dir: str, speaker_col: str = "speakerID", emb_col: str = "global_embedding") -> dict[str, int]:
    """Aggregate global_embeddings per speaker -> L2-normalized mean -> presets/<sid>.pt.

    Returns a dict {speakerID: num_utterances_used}.
    """
    out_dir = Path(presets_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = defaultdict(int)

    for row in ds:
        sid = row.get(speaker_col)
        emb = row.get(emb_col)
        if not sid or emb is None:
            continue
        vec = np.asarray(emb, dtype=np.float32).reshape(-1)
        if sid not in sums:
            sums[sid] = np.zeros_like(vec)
        sums[sid] += vec
        counts[sid] += 1

    written: dict[str, int] = {}
    for sid, total in sums.items():
        n = counts[sid]
        if n == 0:
            continue
        mean = total / n
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        torch.save(torch.from_numpy(mean.astype(np.float32)), out_dir / f"{sid}.pt")
        written[sid] = n
    return written
