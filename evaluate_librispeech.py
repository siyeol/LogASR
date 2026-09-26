#!/usr/bin/env python3
"""Evaluate the proposed Whisper configuration on a local LibriSpeech split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from jiwer import wer
from whisper.normalizers import EnglishTextNormalizer

from log_asr.audio import SAMPLE_RATE, load_audio, load_librispeech
from log_asr.model import load_whisper, transcribe_batch
from log_asr.silence import package_silence
from log_asr.variable_length import encoder_frames


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", choices=["test-clean", "test-other"], default="test-clean")
    parser.add_argument("--mode", choices=["fp16", "w4a4"], default="w4a4")
    parser.add_argument("--model-id", default="openai/whisper-large-v3")
    parser.add_argument("--silence-pack", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=224)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument("--frame-ms", type=float, default=80.0)
    parser.add_argument("--relative-db", type=float, default=-35.0)
    parser.add_argument("--min-silence-frames", type=int, default=2)
    parser.add_argument("--hold-frames", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    items = load_librispeech(args.data_root, args.split)
    if args.limit:
        items = items[: args.limit]
    processor, model, replaced = load_whisper(args.model_id, mode=args.mode)
    normalizer = EnglishTextNormalizer()
    references: list[str] = []
    hypotheses: list[str] = []
    audio_seconds = 0.0
    packaged_seconds = 0.0
    frames_in = 0
    frames_out = 0
    peak_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()

    for start in range(0, len(items), args.batch_size):
        batch = items[start : start + args.batch_size]
        waves = [load_audio(item["path"]) for item in batch]
        audio_seconds += sum(wave.size for wave in waves) / SAMPLE_RATE
        frames_in += sum(encoder_frames(wave.size) for wave in waves)
        if args.silence_pack:
            packed = [
                package_silence(
                    wave,
                    frame_ms=args.frame_ms,
                    relative_db=args.relative_db,
                    min_silence_frames=args.min_silence_frames,
                    hold_frames=args.hold_frames,
                )[0]
                for wave in waves
            ]
        else:
            packed = waves
        packaged_seconds += sum(wave.size for wave in packed) / SAMPLE_RATE
        frames_out += sum(encoder_frames(wave.size) for wave in packed)
        texts = transcribe_batch(processor, model, packed, args.max_new_tokens)
        references.extend(normalizer(item["text"]) for item in batch)
        hypotheses.extend(normalizer(text) for text in texts)
        done = start + len(batch)
        if done % (10 * args.batch_size) == 0 or done == len(items):
            elapsed = time.perf_counter() - started
            print(f"{done}/{len(items)} RTF={elapsed / max(audio_seconds, 1e-8):.4f}", flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "model": args.model_id,
        "dataset": "LibriSpeech",
        "split": args.split,
        "n_utts": len(items),
        "mode": args.mode,
        "quantization": "fake" if args.mode == "w4a4" else None,
        "weight_quantizer": "uniform_symmetric_int4_group64" if args.mode == "w4a4" else None,
        "activation_quantizer": "grouping_free_per_token_log_int4_mu127" if args.mode == "w4a4" else None,
        "quantized_linears": replaced,
        "silence_packaging": args.silence_pack,
        "frame_ms": args.frame_ms if args.silence_pack else None,
        "relative_db": args.relative_db if args.silence_pack else None,
        "min_silence_frames": args.min_silence_frames if args.silence_pack else None,
        "hold_frames": args.hold_frames if args.silence_pack else None,
        "batch_size": args.batch_size,
        "wer": round(100.0 * wer(references, hypotheses), 4),
        "rtf": round(elapsed / max(audio_seconds, 1e-8), 6),
        "elapsed_sec": round(elapsed, 3),
        "audio_sec_in": round(audio_seconds, 3),
        "audio_sec_out": round(packaged_seconds, 3),
        "encoder_frames_in": frames_in,
        "encoder_frames_out": frames_out,
        "encoder_frames_saved_pct": round(100.0 * (1.0 - frames_out / max(frames_in, 1)), 2),
        "gpu_peak_mib": round(max(peak_memory, torch.cuda.max_memory_allocated()) / 2**20, 1),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {output}")


if __name__ == "__main__":
    main()

