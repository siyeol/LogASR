#!/usr/bin/env python3
"""Transcribe audio with Qwen3-ASR and the paper's Log-ASR configuration."""

from __future__ import annotations

import argparse
import json

from log_asr.audio import load_audio
from log_asr.qwen import load_qwen, transcribe_qwen_batch
from log_asr.silence import package_silence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="+")
    parser.add_argument("--mode", choices=["fp16", "w4a4"], default="w4a4")
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--decoder-checkpoint")
    parser.add_argument("--silence-pack", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    asr, replaced, source = load_qwen(
        model_id=args.model_id,
        mode=args.mode,
        decoder_checkpoint=args.decoder_checkpoint,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"model={source} mode={args.mode} encoder_quantized_linears={replaced}")
    for start in range(0, len(args.audio), args.batch_size):
        paths = args.audio[start : start + args.batch_size]
        waves = [load_audio(path) for path in paths]
        stats = [None] * len(waves)
        if args.silence_pack:
            packed = [package_silence(wave) for wave in waves]
            waves = [value[0] for value in packed]
            stats = [value[1].to_dict() for value in packed]
        texts = transcribe_qwen_batch(asr, waves)
        for path, text, packaging in zip(paths, texts, stats):
            print(json.dumps({"audio": path, "text": text, "packaging": packaging}))


if __name__ == "__main__":
    main()
