#!/usr/bin/env python
"""Generate Gujarati speech with the finetuned Indic-Mio model.

Pipeline: text -> ChatML user turn -> LLM generates <|s_N|> tokens -> parse ids ->
MioCodec.decode(global_embedding, ids) -> 24 kHz wav.

Speaker voice comes from a preset (.pt built by speakers.py) OR a reference wav.

Examples:
  python inference.py --text "કેમ છો, તમે કેવા છો?" --speaker SID_187999 --out out.wav
  python inference.py --text "..." --ref-audio ref.wav --adapter outputs/indicmio-guj --out out.wav
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from milli_tts import codec_utils  # noqa: E402
from milli_tts.config import load_config, resolve_secrets  # noqa: E402
from milli_tts.model_loader import load_model_for_inference, load_tokenizer  # noqa: E402


def load_speaker_embedding(presets_dir: Path, speaker: str | None, ref_audio: str | None, codec, device: str):
    if speaker:
        path = presets_dir / f"{speaker}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Preset not found: {path}. Build it with speakers.py.")
        return torch.load(path, map_location="cpu")
    if ref_audio:
        wav = codec_utils.load_audio_for_codec(codec, ref_audio)
        _, gemb = codec_utils.encode_audio(codec, wav, device)
        return torch.from_numpy(gemb)
    raise ValueError("Provide --speaker <SID> or --ref-audio <file>.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--speaker", default=None, help="speaker preset id (SID_xxx)")
    ap.add_argument("--ref-audio", default=None, help="reference wav for zero-shot cloning")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir (default: base model)")
    ap.add_argument("--out", default="out.wav")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config()
    secrets = resolve_secrets(cfg)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(cfg["model"]["base_model_id"], token=secrets.hf_token)
    model = load_model_for_inference(cfg["model"], adapter_dir=args.adapter, token=secrets.hf_token).to(device)

    codec = codec_utils.load_codec(cfg["codec"]["model_id"], device)
    sr = codec_utils.codec_sample_rate(codec)
    gemb = load_speaker_embedding(Path(cfg["paths"]["presets_dir"]), args.speaker, args.ref_audio, codec, device)

    messages = [{"role": "user", "content": args.text.strip()}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        gen = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_text = tokenizer.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=False)
    tokens = codec_utils.parse_speech_tokens(new_text)
    if not tokens:
        print("[error] model produced no <|s_N|> audio tokens.")
        return 1
    print(f"[gen] {len(tokens)} audio tokens (~{len(tokens) / cfg['codec']['token_rate_hz']:.1f}s)")

    wav = codec_utils.decode_tokens(codec, tokens, gemb, device)

    import soundfile as sf

    sf.write(args.out, wav.numpy(), sr)
    print(f"[done] wrote {args.out} ({sr} Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
