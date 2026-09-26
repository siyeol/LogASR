#!/usr/bin/env python3
"""Evaluate Qwen3-ASR Log-ASR + GPTQ W4A16 on LibriSpeech."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from jiwer import wer
from whisper.normalizers import EnglishTextNormalizer

from log_asr.audio import SAMPLE_RATE, load_audio, load_librispeech
from log_asr.qwen import audio_token_lengths, load_qwen, transcribe_qwen_batch
from log_asr.silence import package_silence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", choices=["test-clean", "test-other"], default="test-clean")
    parser.add_argument("--mode", choices=["fp16", "w4a4"], default="w4a4")
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--decoder-checkpoint")
    parser.add_argument("--silence-pack", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
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
    asr, replaced, source = load_qwen(
        model_id=args.model_id,
        mode=args.mode,
        decoder_checkpoint=args.decoder_checkpoint,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    normalizer = EnglishTextNormalizer()
    references, hypotheses = [], []
    audio_seconds = packaged_seconds = 0.0
    tokens_in = tokens_out = 0
    torch.cuda.reset_peak_memory_stats()
    memory_after_load = torch.cuda.memory_allocated()
    torch.cuda.synchronize()
    started = time.perf_counter()

    for start in range(0, len(items), args.batch_size):
        batch = items[start : start + args.batch_size]
        waves = [load_audio(item["path"]) for item in batch]
        audio_seconds += sum(wave.size for wave in waves) / SAMPLE_RATE
        tokens_in += int(np.sum(audio_token_lengths(asr, waves)))
        if args.silence_pack:
            waves = [
                package_silence(
                    wave,
                    frame_ms=args.frame_ms,
                    relative_db=args.relative_db,
                    min_silence_frames=args.min_silence_frames,
                    hold_frames=args.hold_frames,
                )[0]
                for wave in waves
            ]
        packaged_seconds += sum(wave.size for wave in waves) / SAMPLE_RATE
        tokens_out += int(np.sum(audio_token_lengths(asr, waves)))
        texts = transcribe_qwen_batch(asr, waves)
        references.extend(normalizer(item["text"]) for item in batch)
        hypotheses.extend(normalizer(text) for text in texts)
        done = start + len(batch)
        if done % (5 * args.batch_size) == 0 or done == len(items):
            elapsed = time.perf_counter() - started
            print(f"{done}/{len(items)} RTF={elapsed / max(audio_seconds, 1e-8):.4f}", flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "model": args.model_id,
        "loaded_checkpoint": source,
        "dataset": "LibriSpeech",
        "split": args.split,
        "n_utts": len(items),
        "mode": args.mode,
        "encoder_quantization": "uniform_W4_G64+Log-ASR_A4_per-token" if args.mode == "w4a4" else None,
        "decoder_quantization": "GPTQ_W4A16_G128" if args.mode == "w4a4" else None,
        "encoder_quantized_linears": replaced,
        "silence_packaging": args.silence_pack,
        "batch_size": args.batch_size,
        "wer": round(100.0 * wer(references, hypotheses), 4),
        "rtf": round(elapsed / max(audio_seconds, 1e-8), 6),
        "elapsed_sec": round(elapsed, 3),
        "audio_sec_in": round(audio_seconds, 3),
        "audio_sec_out": round(packaged_seconds, 3),
        "audio_tokens_in": tokens_in,
        "audio_tokens_out": tokens_out,
        "audio_tokens_saved_pct": round(100.0 * (1.0 - tokens_out / max(tokens_in, 1)), 2),
        "gpu_mem_after_load_mib": round(memory_after_load / 2**20, 1),
        "gpu_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "sfp": {
            "frame_ms": args.frame_ms,
            "relative_db": args.relative_db,
            "min_silence_frames": args.min_silence_frames,
            "hold_frames": args.hold_frames,
        } if args.silence_pack else None,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
