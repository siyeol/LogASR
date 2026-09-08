"""Compare cross-KV storage on LibriSpeech and probe GPU peak at a given batch size."""

from __future__ import annotations

import json
from pathlib import Path

from data import SR, load_split
from kv_template import MAX_ENC_FRAMES, cross_kv_bytes, encoder_frames_from_nsamples

RESULTS = Path("/workspace/SpeechPTQ/results")


def corpus_kv_stats(split: str = "test-clean") -> dict:
    items = load_split(split)
    n_frames = []
    durs = []
    for it in items:
        import soundfile as sf

        info = sf.info(it["path"])
        ns = int(info.frames)
        n_frames.append(encoder_frames_from_nsamples(ns))
        durs.append(ns / SR)

    full_each = cross_kv_bytes(MAX_ENC_FRAMES) / (1024**2)
    speech = [cross_kv_bytes(n) / (1024**2) for n in n_frames]
    mean_n = sum(n_frames) / len(n_frames)
    mean_dur = sum(durs) / len(durs)
    return {
        "split": split,
        "n_utts": len(items),
        "mean_duration_sec": round(mean_dur, 3),
        "mean_encoder_frames": round(mean_n, 1),
        "full_cross_kv_mib_per_utt": round(full_each, 2),
        "speech_cross_kv_mib_mean": round(sum(speech) / len(speech), 2),
        "speech_cross_kv_mib_total": round(sum(speech), 1),
        "full_cross_kv_mib_total": round(full_each * len(items), 1),
        "shared_template_mib": round(full_each, 2),
        "saved_pct_vs_full": round(100.0 * (1.0 - (sum(speech) / len(speech)) / full_each), 2),
        "formula": "32 layers * 2 (K,V) * T * 1280 * 2 bytes (fp16)",
    }


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    stats = corpus_kv_stats("test-clean")
    path = RESULTS / "kv_memory_test-clean.json"
    path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"wrote {path}")
