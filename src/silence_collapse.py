"""Collapse consecutive non-speech frames to a short shared hold.

Token-aligned to Qwen3-ASR (~12.5 Hz after 8x mel downsample, 80 ms/token).
Used to shrink pause audio tokens without dropping speech frames.
"""

from __future__ import annotations

import numpy as np

SR = 16000


def frame_rms(wave: np.ndarray, frame_samples: int) -> np.ndarray:
    n = int(wave.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    pad = (-n) % frame_samples
    if pad:
        wave = np.pad(wave, (0, pad))
    frames = wave.reshape(-1, frame_samples)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12).astype(np.float32)


def speech_mask_from_rms(rms: np.ndarray, rel_db: float = -35.0, abs_floor: float = 1e-4) -> np.ndarray:
    """Frame is speech if RMS is above max(peak * 10^(rel_db/20), abs_floor)."""
    if rms.size == 0:
        return np.zeros((0,), dtype=bool)
    peak = float(np.max(rms))
    thresh = max(peak * (10.0 ** (rel_db / 20.0)), abs_floor)
    return rms >= thresh


def collapse_silence(
    wave: np.ndarray,
    sr: int = SR,
    frame_ms: float = 80.0,
    hold_frames: int = 1,
    rel_db: float = -35.0,
    min_sil_frames: int = 2,
) -> tuple[np.ndarray, dict]:
    """Keep speech frames; replace each silence run of >= min_sil_frames with hold_frames.

    Returns the collapsed waveform and stats (frames in/out, audio sec in/out).
    """
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    frame_samples = max(int(round(sr * frame_ms / 1000.0)), 1)
    rms = frame_rms(wave, frame_samples)
    speech = speech_mask_from_rms(rms, rel_db=rel_db)
    n_in = int(rms.shape[0])
    keep = np.zeros(n_in, dtype=bool)

    i = 0
    n_runs = 0
    n_sil_in = 0
    n_sil_out = 0
    while i < n_in:
        if speech[i]:
            keep[i] = True
            i += 1
            continue
        j = i
        while j < n_in and not speech[j]:
            j += 1
        run = j - i
        n_sil_in += run
        n_runs += 1
        if run < min_sil_frames:
            keep[i:j] = True
            n_sil_out += run
        else:
            n_keep = min(hold_frames, run)
            keep[i : i + n_keep] = True
            n_sil_out += n_keep
        i = j

    if not np.any(keep):
        keep[: min(hold_frames, n_in)] = True

    n_out = int(keep.sum())
    parts = []
    padded = wave
    pad = (-wave.shape[0]) % frame_samples
    if pad:
        padded = np.pad(wave, (0, pad))
    frames = padded.reshape(-1, frame_samples)
    out = frames[keep].reshape(-1)
    # drop the padding we added if the last kept frame is the pad frame only — keep simple: trim to n_out * frame
    out = out[: n_out * frame_samples]
    if out.size == 0:
        out = np.zeros((frame_samples,), dtype=np.float32)

    stats = {
        "frames_in": n_in,
        "frames_out": n_out,
        "audio_sec_in": float(wave.shape[0]) / sr,
        "audio_sec_out": float(out.shape[0]) / sr,
        "silence_runs": n_runs,
        "sil_frames_in": n_sil_in,
        "sil_frames_out": n_sil_out,
        "frame_ms": frame_ms,
        "hold_frames": hold_frames,
        "rel_db": rel_db,
    }
    return out.astype(np.float32), stats
