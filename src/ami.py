"""AMI IHM test loader (OpenASR-style edinburghcstr/ami chunks) plus pause-aware stitch."""

from __future__ import annotations

from pathlib import Path

import numpy as np

SR = 16000
AMI_CACHE = Path("/workspace/SpeechPTQ/data/ami_ihm_test")
AMI_STITCH30 = Path("/workspace/SpeechPTQ/data/ami_ihm_stitch30")


def load_ami_stitch30() -> list[dict]:
    """Load cached 30s pause-stitched AMI windows written by prepare_ami_windows.py."""
    if not AMI_STITCH30.is_dir():
        raise FileNotFoundError(AMI_STITCH30)
    items: list[dict] = []
    for trans in sorted(AMI_STITCH30.rglob("*.trans.txt")):
        with trans.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt_id, text = line.split(" ", 1)
                flac = trans.parent / f"{utt_id}.flac"
                items.append(
                    {
                        "id": utt_id,
                        "meeting_id": trans.parent.name,
                        "path": str(flac),
                        "text": text,
                    }
                )
    return items


def decode_hf_audio(aud) -> tuple[np.ndarray, int]:
    """Decode datasets AudioDecoder / dict / ndarray to (mono float32, sr)."""
    if isinstance(aud, dict) and "array" in aud:
        return np.asarray(aud["array"], dtype=np.float32), int(aud.get("sampling_rate", SR))
    if hasattr(aud, "get_all_samples"):
        samples = aud.get_all_samples()
        x = samples.data
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)
        return np.asarray(x, dtype=np.float32), int(samples.sample_rate)
    return np.asarray(aud, dtype=np.float32), SR


def _to_mono16k(audio, sr: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim > 1:
        # torchcodec is [C, T]; some loaders are [T, C]
        if x.shape[0] <= 8 and x.shape[-1] > x.shape[0]:
            x = x.mean(axis=0)
        else:
            x = x.mean(axis=-1)
    if sr != SR:
        try:
            import torchaudio

            t = torchaudio.functional.resample(
                __import__("torch").from_numpy(x).unsqueeze(0), sr, SR
            )
            x = t.squeeze(0).numpy().astype(np.float32)
        except Exception:
            n_out = int(round(x.shape[0] * SR / sr))
            xp = np.linspace(0, 1, num=x.shape[0], endpoint=False)
            xq = np.linspace(0, 1, num=n_out, endpoint=False)
            x = np.interp(xq, xp, x).astype(np.float32)
    return x


def load_ami_ihm_test(limit: int = 0) -> list[dict]:
    """Download/cache edinburghcstr/ami ihm test chunks."""
    from datasets import Audio, load_dataset

    ds = load_dataset("edinburghcstr/ami", "ihm", split="test")
    ds = ds.cast_column("audio", Audio(sampling_rate=SR))
    if limit and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    items: list[dict] = []
    for i, row in enumerate(ds):
        aud = row["audio"]
        arr, sr = decode_hf_audio(aud)
        wave = _to_mono16k(arr, sr)
        items.append(
            {
                "id": str(row.get("audio_id", i)),
                "meeting_id": str(row.get("meeting_id", "")),
                "speaker_id": str(row.get("speaker_id", "")),
                "begin_time": float(row.get("begin_time", 0.0) or 0.0),
                "end_time": float(row.get("end_time", 0.0) or 0.0),
                "text": str(row.get("text", "")),
                "audio": wave,
            }
        )
    return items


def stitch_pause_windows(
    items: list[dict],
    target_sec: float = 30.0,
    max_gap_sec: float = 4.0,
    sr: int = SR,
) -> list[dict]:
    """Concatenate same-meeting chunks with real inter-utt zeros (capped gaps).

    Kaldi AMI chunks are already VAD-cut, so intra-clip pauses are small.
    Stitching restores meeting pauses, which is the Qwen3 analog of Whisper 30s pad.
    """
    from collections import defaultdict

    by_m: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_m[it["meeting_id"]].append(it)

    windows: list[dict] = []
    for meeting, rows in by_m.items():
        rows = sorted(rows, key=lambda r: (r["begin_time"], r["id"]))
        cur_parts: list[np.ndarray] = []
        cur_texts: list[str] = []
        cur_ids: list[str] = []
        last_end: float | None = None
        win_i = 0

        def flush():
            nonlocal cur_parts, cur_texts, cur_ids, last_end, win_i
            if not cur_parts:
                return
            wave = np.concatenate(cur_parts).astype(np.float32)
            windows.append(
                {
                    "id": f"{meeting}_{win_i:04d}",
                    "meeting_id": meeting,
                    "text": " ".join(t for t in cur_texts if t),
                    "audio": wave,
                    "n_chunks": len(cur_ids),
                    "chunk_ids": cur_ids,
                }
            )
            cur_parts, cur_texts, cur_ids = [], [], []
            last_end = None
            win_i += 1

        for r in rows:
            gap = 0.0
            if last_end is not None:
                gap = max(0.0, r["begin_time"] - last_end)
                gap = min(gap, max_gap_sec)
            trial = (sum(p.shape[0] for p in cur_parts) / sr) + gap + (r["audio"].shape[0] / sr)
            if cur_parts and trial > target_sec:
                flush()
                gap = 0.0
            if gap > 0:
                n_gap = int(round(gap * sr))
                if n_gap > 0:
                    cur_parts.append(np.zeros((n_gap,), dtype=np.float32))
            cur_parts.append(r["audio"])
            cur_texts.append(r["text"])
            cur_ids.append(r["id"])
            last_end = r["end_time"] if r["end_time"] else (
                (last_end or 0.0) + r["audio"].shape[0] / sr + gap
            )
        flush()
    return windows
