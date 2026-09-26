#!/usr/bin/env python3
"""Transcribe local audio with the proposed Whisper configuration."""

from __future__ import annotations

import argparse
import json

from log_asr.audio import load_audio
from log_asr.model import load_whisper, transcribe_batch
from log_asr.silence import package_silence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="+")
    parser.add_argument("--mode", choices=["fp16", "w4a4"], default="w4a4")
    parser.add_argument("--model-id", default="openai/whisper-large-v3")
    parser.add_argument("--silence-pack", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=224)
    parser.add_argument("--frame-ms", type=float, default=80.0)
    parser.add_argument("--relative-db", type=float, default=-35.0)
    parser.add_argument("--min-silence-frames", type=int, default=2)
    parser.add_argument("--hold-frames", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    processor, model, replaced = load_whisper(args.model_id, mode=args.mode)
    print(f"model={args.model_id} mode={args.mode} quantized_linears={replaced}")
    for start in range(0, len(args.audio), args.batch_size):
        paths = args.audio[start : start + args.batch_size]
        waves = [load_audio(path) for path in paths]
        stats = [None] * len(waves)
        if args.silence_pack:
            packed = [
                package_silence(
                    wave,
                    frame_ms=args.frame_ms,
                    relative_db=args.relative_db,
                    min_silence_frames=args.min_silence_frames,
                    hold_frames=args.hold_frames,
                )
                for wave in waves
            ]
            waves = [item[0] for item in packed]
            stats = [item[1].to_dict() for item in packed]
        texts = transcribe_batch(processor, model, waves, args.max_new_tokens)
        for path, text, packaging in zip(paths, texts, stats):
            print(json.dumps({"audio": path, "text": text, "packaging": packaging}))


if __name__ == "__main__":
    main()

