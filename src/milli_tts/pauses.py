"""Insert explicit ``<pause>`` tags into text based on silences detected in the audio.

IndicConformer produces plain text without pause/breath markers. To teach the model
"more pauses" (per the user's goal) we detect long silent gaps with librosa's energy
split and insert a ``<pause>`` tag proportional to how many gaps were found, distributed
across word boundaries. This is a heuristic alignment-free approximation: it gives the
model a correlation between long silences in the target audio and the pause marker.
"""
from __future__ import annotations

import numpy as np


def count_silence_gaps(array: np.ndarray, sr: int, top_db: float = 30.0, min_silence_ms: int = 350) -> int:
    """Return the number of silent gaps (longer than min_silence_ms) between non-silent
    intervals in the waveform."""
    try:
        import librosa
    except Exception:
        return 0
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=-1)
    if arr.size == 0:
        return 0
    intervals = librosa.effects.split(arr, top_db=top_db)
    if len(intervals) <= 1:
        return 0
    min_gap = int(sr * min_silence_ms / 1000)
    gaps = 0
    for (prev_end, next_start) in zip(intervals[:-1, 1], intervals[1:, 0]):
        if next_start - prev_end >= min_gap:
            gaps += 1
    return gaps


def insert_pause_tags(text: str, num_gaps: int, tag: str = "<pause>") -> str:
    """Insert ``num_gaps`` pause tags at evenly spaced word boundaries in ``text``."""
    if num_gaps <= 0 or not text.strip():
        return text
    words = text.split()
    if len(words) < 2:
        return text
    # Cap the number of inserted tags to the number of internal boundaries available.
    num_gaps = min(num_gaps, len(words) - 1)
    # Evenly spaced boundary positions (1..len(words)-1).
    positions = {round(i * len(words) / (num_gaps + 1)) for i in range(1, num_gaps + 1)}
    positions = {p for p in positions if 0 < p < len(words)}
    out: list[str] = []
    for idx, word in enumerate(words):
        if idx in positions:
            out.append(tag)
        out.append(word)
    return " ".join(out)


def add_pauses_from_audio(text: str, array: np.ndarray, sr: int, *, top_db: float, min_silence_ms: int, tag: str) -> str:
    gaps = count_silence_gaps(array, sr, top_db=top_db, min_silence_ms=min_silence_ms)
    return insert_pause_tags(text, gaps, tag=tag)
