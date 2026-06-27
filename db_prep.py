#!/usr/bin/env python
"""Prepare the Gujarati finetuning dataset from ARTPARK-IISc/Vaani.

For one Vaani config at a time it:
  1. STREAMS the source (gated -> needs HF token with accepted access),
  2. keeps ONLY Gujarati rows (language == "Gujarati", state == "Gujarat"),
  3. filters by duration,
  4. transcribes with IndicConformer (Gujarati has no transcripts in Vaani),
  5. optionally inserts <pause> tags from detected silences,
  6. encodes audio with MioCodec -> speech_tokens + global_embedding,
  7. pushes each processed SHARD to the new HF dataset (milli_guj_dataset_artpark)
     under its own config name, so a crash never loses earlier shards.

Resumable: a JSON state file per (config, worker) records completed shards so reruns
continue where they left off. Splits work across 2x T4 with --worker-id / --num-workers
(see scripts/run_db_prep_2gpu.sh).

Examples:
  # dry run, no push, 8 Gujarati rows from one config
  python db_prep.py --config Gujarat_Valsad --limit 8 --no-push
  # one worker, push shards
  python db_prep.py --config Gujarat_Valsad --worker-id 0 --num-workers 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from milli_tts import asr_utils, codec_utils, pauses  # noqa: E402
from milli_tts.config import (  # noqa: E402
    load_config,
    require_hf_token,
    resolve_dataset_repo_id,
    resolve_secrets,
)


def pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


def state_path(state_dir: Path, repo_config: str, worker_id: int) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f"{repo_config}__w{worker_id}.json"


def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"completed_shards": 0, "streamed_consumed": 0}


def save_state(path: Path, completed_shards: int, streamed_consumed: int) -> None:
    path.write_text(json.dumps({"completed_shards": completed_shards, "streamed_consumed": streamed_consumed}))


def build_row(cfg, src, text, tokens, gemb, dur, keep_audio, audio_24k_array, sr):
    ds_src = cfg["dataset"]["source"]
    row = {
        "id": str(src.get("UtteranceSequenceID", "")) + "_" + str(src.get(ds_src["speaker_column"], "")),
        "speakerID": src.get(ds_src["speaker_column"]),
        "gender": src.get(ds_src["gender_column"]),
        "district": src.get(ds_src["district_column"]),
        "language": src.get(ds_src["language_column"]),
        "duration": float(dur) if dur is not None else None,
        "text": text,
        "transcript": text,
        "transcript_source": "asr_indicconformer",
        "speech_tokens": tokens,
        "num_speech_tokens": len(tokens),
        "global_embedding": gemb.astype(np.float32).tolist(),
    }
    if keep_audio:
        row["audio"] = {"array": audio_24k_array, "sampling_rate": sr}
    return row


def flush_shard(rows, repo, config_name, token, no_push):
    from datasets import Dataset

    ds = Dataset.from_list(rows)
    if no_push:
        print(f"[dry-run] would push {len(rows)} rows as config '{config_name}' to {repo}")
        return
    ds.push_to_hub(repo, config_name=config_name, split="train", token=token, private=True)
    print(f"[push] {len(rows)} rows -> {repo} (config={config_name})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="one Vaani config (default: all from config.json)")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--gpu", default=None, help="device, e.g. cuda:0 / cpu")
    ap.add_argument("--limit", type=int, default=None, help="max accepted (Gujarati) rows to process")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--no-push", action="store_true", help="dry run, do not push to the Hub")
    ap.add_argument("--config-file", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config_file)
    token = require_hf_token(cfg)
    device = pick_device(args.gpu)
    print(f"[device] {device}")

    src = cfg["dataset"]["source"]
    out = cfg["dataset"]["output"]
    repo = resolve_dataset_repo_id(cfg, token)
    configs = [args.config] if args.config else src["configs"]
    shard_rows = int(out.get("shard_rows", 500))
    keep_audio = bool(out.get("keep_audio", False))
    pause_cfg = cfg["pauses"]
    asr_cfg = cfg["asr"]
    codec_cfg = cfg["codec"]
    state_dir = Path(cfg["paths"]["state_dir"])

    # Load models once.
    print("[load] MioCodec ...")
    codec = codec_utils.load_codec(codec_cfg["model_id"], device)
    target_sr = codec_utils.codec_sample_rate(codec)
    print(f"[codec] sample_rate={target_sr}")

    print("[load] IndicConformer ASR ...")
    asr = asr_utils.GujaratiASR(asr_cfg["model_id"], device, asr_cfg["language"], asr_cfg["decoding"])
    asr.load()

    from datasets import load_dataset

    accepted_total = 0
    for vaani_config in configs:
        print(f"\n=== config: {vaani_config} (worker {args.worker_id}/{args.num_workers}) ===")
        sp = state_path(state_dir, vaani_config, args.worker_id)
        st = load_state(sp)
        shard_idx = st["completed_shards"]
        abs_idx = st["streamed_consumed"]

        ds = load_dataset(src["repo"], vaani_config, split=src["split"], streaming=True, token=token)
        if abs_idx > 0:
            ds = ds.skip(abs_idx)
            print(f"[resume] skipping {abs_idx} source rows, starting at shard {shard_idx}")

        buffer: list[dict] = []
        for row in ds:
            cur_idx = abs_idx
            abs_idx += 1
            # Worker row-split.
            if cur_idx % args.num_workers != args.worker_id:
                continue
            # Gujarati-only filter (cheap; before heavy work).
            if str(row.get(src["language_column"])) != src["language_filter"]:
                continue
            if src.get("state_filter") and str(row.get(src["state_column"])) != src["state_filter"]:
                continue
            audio = row.get(src["audio_column"])
            if not audio or "array" not in audio:
                continue
            dur = row.get(src["duration_column"])
            if dur is not None and not (src["min_dur_s"] <= float(dur) <= src["max_dur_s"]):
                continue

            arr = np.asarray(audio["array"], dtype=np.float32)
            ssr = int(audio["sampling_rate"])

            # ASR transcription.
            try:
                wav16 = asr_utils.to_asr_waveform(arr, ssr)
                text = asr.transcribe(wav16)
            except Exception as exc:  # noqa: BLE001
                print(f"[asr-skip] {exc}")
                continue
            if asr_cfg.get("drop_empty", True) and not asr_utils.is_valid_transcript(
                text, asr_cfg["min_chars"], asr_cfg["max_chars"]
            ):
                continue

            # Optional pause tags from silence.
            if pause_cfg.get("enabled", True) and out.get("pause_tags", True):
                text = pauses.add_pauses_from_audio(
                    text, arr, ssr,
                    top_db=pause_cfg["top_db"],
                    min_silence_ms=pause_cfg["min_silence_ms"],
                    tag=pause_cfg["tag"],
                )

            # MioCodec encode.
            try:
                wav24 = codec_utils.waveform_from_row(audio, target_sr)
                tokens, gemb = codec_utils.encode_audio(codec, wav24, device)
            except Exception as exc:  # noqa: BLE001
                print(f"[codec-skip] {exc}")
                continue
            if not tokens:
                continue

            buffer.append(build_row(cfg, row, text, tokens, gemb, dur, keep_audio, wav24.numpy(), target_sr))
            accepted_total += 1

            if args.no_push and accepted_total <= 3:
                print(f"[sample] sid={row.get(src['speaker_column'])} chars={len(text)} "
                      f"tokens={len(tokens)} emb_dim={gemb.shape[0]} text='{text[:60]}...'")

            if len(buffer) >= shard_rows:
                cfg_name = f"{vaani_config}_w{args.worker_id}_s{shard_idx}"
                flush_shard(buffer, repo, cfg_name, token, args.no_push)
                shard_idx += 1
                if not args.no_push:
                    save_state(sp, shard_idx, abs_idx)
                buffer = []
                if args.max_shards and shard_idx >= args.max_shards:
                    break
            if args.limit and accepted_total >= args.limit:
                break

        # Flush remainder.
        if buffer:
            cfg_name = f"{vaani_config}_w{args.worker_id}_s{shard_idx}"
            flush_shard(buffer, repo, cfg_name, token, args.no_push)
            shard_idx += 1
            if not args.no_push:
                save_state(sp, shard_idx, abs_idx)

        if args.limit and accepted_total >= args.limit:
            break

    print(f"\n[done] accepted {accepted_total} Gujarati rows total.")
    if not resolve_secrets(cfg).wandb_api_key:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
