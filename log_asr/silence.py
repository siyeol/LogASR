"""RMS-based silence frame packaging."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class PackagingStats:
    frames_in: int
    frames_out: int
    audio_sec_in: float
    audio_sec_out: float
    silence_runs: int
    silence_frames_in: int
    silence_frames_out: int
    frame_ms: float
    hold_frames: int
    relative_db: float
    min_silence_frames: int

    def to_dict(self) -> dict:
        return asdict(self)


def frame_rms(waveform: np.ndarray, frame_samples: int) -> np.ndarray:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        return np.zeros(0, dtype=np.float32)
    padding = (-waveform.size) % frame_samples
    padded = np.pad(waveform, (0, padding)) if padding else waveform
    frames = padded.reshape(-1, frame_samples)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12).astype(np.float32)


def speech_mask_from_rms(
    rms: np.ndarray, relative_db: float = -35.0, absolute_floor: float = 1e-4
) -> np.ndarray:
    if rms.size == 0:
        return np.zeros(0, dtype=bool)
    threshold = max(float(rms.max()) * 10.0 ** (relative_db / 20.0), absolute_floor)
    return rms >= threshold


def package_silence(
    waveform: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    frame_ms: float = 80.0,
    hold_frames: int = 1,
    relative_db: float = -35.0,
    min_silence_frames: int = 2,
) -> tuple[np.ndarray, PackagingStats]:
    """Replace each long silence run with its first ``hold_frames`` frames."""
    if hold_frames < 1:
        raise ValueError("hold_frames must be at least 1")
    if min_silence_frames < 1:
        raise ValueError("min_silence_frames must be at least 1")

    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    frame_samples = max(int(round(sample_rate * frame_ms / 1000.0)), 1)
    rms = frame_rms(waveform, frame_samples)
    speech = speech_mask_from_rms(rms, relative_db=relative_db)
    keep = np.zeros(rms.size, dtype=bool)
    silence_runs = 0
    silence_frames_in = 0
    silence_frames_out = 0

    index = 0
    while index < rms.size:
        if speech[index]:
            keep[index] = True
            index += 1
            continue
        end = index
        while end < rms.size and not speech[end]:
            end += 1
        run_length = end - index
        silence_runs += 1
        silence_frames_in += run_length
        retained = run_length if run_length < min_silence_frames else min(hold_frames, run_length)
        keep[index : index + retained] = True
        silence_frames_out += retained
        index = end

    if keep.size and not keep.any():
        retained = min(hold_frames, keep.size)
        keep[:retained] = True
        silence_frames_out = retained

    padding = (-waveform.size) % frame_samples
    padded = np.pad(waveform, (0, padding)) if padding else waveform
    frames = padded.reshape(-1, frame_samples) if padded.size else np.empty((0, frame_samples))
    output = frames[keep].reshape(-1).astype(np.float32) if keep.size else waveform.copy()
    if output.size == 0:
        output = np.zeros(frame_samples, dtype=np.float32)

    stats = PackagingStats(
        frames_in=int(rms.size),
        frames_out=int(keep.sum()),
        audio_sec_in=float(waveform.size) / sample_rate,
        audio_sec_out=float(output.size) / sample_rate,
        silence_runs=silence_runs,
        silence_frames_in=silence_frames_in,
        silence_frames_out=silence_frames_out,
        frame_ms=frame_ms,
        hold_frames=hold_frames,
        relative_db=relative_db,
        min_silence_frames=min_silence_frames,
    )
    return output, stats

