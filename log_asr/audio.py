"""Audio and LibriSpeech file utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000


def load_audio(path: str | Path) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz, got {sample_rate} Hz: {path}")
    return np.asarray(waveform, dtype=np.float32)


def load_librispeech(data_root: str | Path, split: str) -> list[dict[str, str]]:
    root = Path(data_root) / split
    if not root.is_dir():
        raise FileNotFoundError(f"LibriSpeech split not found: {root}")
    items: list[dict[str, str]] = []
    for transcript in sorted(root.rglob("*.trans.txt")):
        with transcript.open(encoding="utf-8") as handle:
            for line in handle:
                utterance_id, text = line.rstrip().split(" ", 1)
                audio_path = transcript.parent / f"{utterance_id}.flac"
                if not audio_path.is_file():
                    raise FileNotFoundError(audio_path)
                items.append({"id": utterance_id, "path": str(audio_path), "text": text})
    return items

