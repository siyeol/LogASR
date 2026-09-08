"""LibriSpeech loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
DATA_ROOT = Path("/workspace/SpeechPTQ/data/LibriSpeech")


def load_split(split: str = "test-clean") -> list[dict]:
    root = DATA_ROOT / split
    if not root.is_dir():
        raise FileNotFoundError(f"Missing {root}")
    items: list[dict] = []
    for trans in sorted(root.rglob("*.trans.txt")):
        with trans.open() as f:
            for line in f:
                utt_id, text = line.strip().split(" ", 1)
                flac = trans.parent / f"{utt_id}.flac"
                items.append(
                    {
                        "id": utt_id,
                        "path": str(flac),
                        "text": text,
                    }
                )
    return items


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        raise ValueError(f"Expected {SR} Hz, got {sr} for {path}")
    return audio
