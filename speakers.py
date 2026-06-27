#!/usr/bin/env python
"""Build per-speaker voice presets (presets/<speakerID>.pt) from the processed dataset.

Run after db_prep.py has populated the dataset:
    python speakers.py
Each preset is the L2-normalized mean of that speaker's MioCodec global_embeddings,
usable by inference.py via --speaker <SID>.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from milli_tts.config import load_config, resolve_dataset_repo_id, resolve_secrets  # noqa: E402
from milli_tts.dataset import load_processed_dataset  # noqa: E402
from milli_tts.speakers import build_speaker_presets  # noqa: E402


def main() -> int:
    cfg = load_config()
    secrets = resolve_secrets(cfg)

    dataset_repo = os.environ.get("DATASET_REPO") or resolve_dataset_repo_id(cfg, secrets.hf_token)
    print(f"[data] loading {dataset_repo}")
    ds = load_processed_dataset(dataset_repo, token=secrets.hf_token)

    presets_dir = cfg["paths"]["presets_dir"]
    written = build_speaker_presets(ds, presets_dir)
    print(f"[done] wrote {len(written)} speaker presets to {presets_dir}/")
    for sid, n in sorted(written.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {sid}: {n} utterances")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
