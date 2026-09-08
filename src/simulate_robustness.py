"""Simulate LS test-clean noise-robustness sets with pyroomacoustics.

1) test-clean-reverb: REVERB-like RT60 [0.2, 1.2] s + stationary noise SNR [5, 20] dB
2) test-clean-dns: RT60 [0.2, 0.4] s, 1-3 babble sources (total SNR [-5, 20]),
   1-3 directional interferer talkers from LS test-other (total SNR [10, 20])

Room L,W ~ U[5, 10] m, H ~ U[3, 4] m. Mono mic, random geometry.
Official REVERB noise is LDC-only; stationary noise is NOISEX-style
white/pink plus a small MS-SNSD air-conditioner/vacuum/copy set.
Babble is the MS-SNSD Babble_* files (not full DNS5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import soundfile as sf
from scipy.signal import resample_poly

from data import DATA_ROOT, SR, load_audio, load_split

MARGIN = 0.5
MIN_SRC_DIST = 0.5
REVERB_RT = (0.2, 1.2)
REVERB_SNR = (5.0, 20.0)
DNS_RT = (0.2, 0.4)
DNS_BABBLE_SNR = (-5.0, 20.0)
DNS_TALKER_SNR = (10.0, 20.0)
DNS_N_RANGE = (1, 3)
ROOM_L = (5.0, 10.0)
ROOM_W = (5.0, 10.0)
ROOM_H = (3.0, 4.0)
SRC_Z = (1.2, 1.8)
MIC_Z = (1.0, 1.6)
TAIL_SEC = 0.25
NOISE_ROOT = Path("/workspace/SpeechPTQ/data/noise")


def _to_mono16k(x: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr == SR:
        return x
    g = math.gcd(int(sr), SR)
    y = resample_poly(x, SR // g, int(sr) // g)
    return np.asarray(y, dtype=np.float32)


def _load_wav_bank(paths: list[Path]) -> list[np.ndarray]:
    bank = []
    for p in paths:
        audio, sr = sf.read(str(p), dtype="float32")
        y = _to_mono16k(audio, sr)
        if y.size >= SR // 2:
            bank.append(y)
    if not bank:
        raise RuntimeError(f"empty noise bank from {paths[:3]}")
    return bank


def _fit_len(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if x.size >= n:
        start = int(rng.integers(0, x.size - n + 1))
        return x[start : start + n].copy()
    reps = int(math.ceil(n / x.size))
    return np.tile(x, reps)[:n]


def _peak_protect(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m > peak:
        x = x * (peak / m)
    return x.astype(np.float32)


def _scale_to_snr(ref: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    e_r = float(np.mean(ref.astype(np.float64) ** 2))
    e_n = float(np.mean(noise.astype(np.float64) ** 2))
    if e_r < 1e-12 or e_n < 1e-12:
        return np.zeros_like(noise)
    scale = math.sqrt(e_r / (e_n * (10.0 ** (snr_db / 10.0))))
    return (noise * scale).astype(np.float32)


def _rand_xyz(rng: np.random.Generator, dims, z_lo: float, z_hi: float) -> np.ndarray:
    L, W, H = dims
    z_hi = min(z_hi, H - 0.2)
    z_lo = min(z_lo, z_hi - 0.05)
    return np.array(
        [
            rng.uniform(MARGIN, L - MARGIN),
            rng.uniform(MARGIN, W - MARGIN),
            rng.uniform(z_lo, z_hi),
        ],
        dtype=np.float64,
    )


def _far_enough(p: np.ndarray, others: list[np.ndarray]) -> bool:
    return all(float(np.linalg.norm(p - q)) >= MIN_SRC_DIST for q in others)


def _place(rng, dims, z_lo, z_hi, occupied: list[np.ndarray]) -> np.ndarray:
    for _ in range(80):
        p = _rand_xyz(rng, dims, z_lo, z_hi)
        if _far_enough(p, occupied):
            return p
    return _rand_xyz(rng, dims, z_lo, z_hi)


def _absorption(rt60: float, dims) -> tuple[float, int]:
    try:
        e_abs, max_order = pra.inverse_sabine(rt60, dims)
    except Exception:
        vol = float(np.prod(dims))
        area = 2.0 * (dims[0] * dims[1] + dims[0] * dims[2] + dims[1] * dims[2])
        e_abs = max(0.05, min(0.95, 0.161 * vol / (rt60 * area)))
        max_order = 12
    e_abs = float(np.clip(e_abs, 0.05, 0.95))
    max_order = int(np.clip(max_order, 3, 12))
    return e_abs, max_order


def render_stems(
    signals_pos: list[tuple[np.ndarray, np.ndarray]],
    mic_pos: np.ndarray,
    dims,
    rt60: float,
    n_out: int,
) -> list[np.ndarray]:
    """Render each (signal, xyz) as its own mic stem in the same room."""
    e_abs, max_order = _absorption(rt60, dims)
    stems = []
    for sig, pos in signals_pos:
        room = pra.ShoeBox(
            dims,
            fs=SR,
            materials=pra.Material(e_abs),
            max_order=max_order,
            air_absorption=False,
            ray_tracing=False,
        )
        room.add_microphone(mic_pos.tolist())
        room.add_source(pos.tolist(), signal=np.asarray(sig, dtype=np.float32))
        room.simulate()
        mic = np.asarray(room.mic_array.signals[0], dtype=np.float32)
        if mic.size < n_out:
            mic = np.pad(mic, (0, n_out - mic.size))
        stems.append(mic[:n_out])
    return stems


def _rel_flac(item: dict) -> Path:
    p = Path(item["path"])
    return p.relative_to(DATA_ROOT / "test-clean")


_G = {}


def _init_worker(stationary: list[str], babble: list[str], talkers: list[dict]):
    _G["stationary"] = _load_wav_bank([Path(p) for p in stationary])
    _G["babble"] = _load_wav_bank([Path(p) for p in babble])
    _G["talkers"] = talkers


def process_item(item: dict) -> dict:
    utt = item["id"]
    spk = utt.split("-")[0]
    seed = int(hashlib.md5(utt.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    target = load_audio(item["path"])
    n_t = int(target.size)
    n_out = n_t + int(TAIL_SEC * SR)
    rel = _rel_flac(item)

    # --- REVERB-like ---
    dims_r = np.array(
        [rng.uniform(*ROOM_L), rng.uniform(*ROOM_W), rng.uniform(*ROOM_H)],
        dtype=np.float64,
    )
    rt_r = float(rng.uniform(*REVERB_RT))
    snr_r = float(rng.uniform(*REVERB_SNR))
    mic_r = _place(rng, dims_r, *MIC_Z, [])
    src_r = _place(rng, dims_r, *SRC_Z, [mic_r])
    npos_r = _place(rng, dims_r, 0.4, dims_r[2] - 0.3, [mic_r, src_r])
    noise = _fit_len(_G["stationary"][int(rng.integers(0, len(_G["stationary"])))], n_t, rng)
    t_stem, n_stem = render_stems(
        [(target, src_r), (noise, npos_r)], mic_r, dims_r, rt_r, n_out
    )
    mix_r = _peak_protect(t_stem + _scale_to_snr(t_stem, n_stem, snr_r))
    out_r = DATA_ROOT / "test-clean-reverb" / rel
    out_r.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_r), mix_r, SR, format="FLAC")

    # --- DNS-like ---
    dims_d = np.array(
        [rng.uniform(*ROOM_L), rng.uniform(*ROOM_W), rng.uniform(*ROOM_H)],
        dtype=np.float64,
    )
    rt_d = float(rng.uniform(*DNS_RT))
    n_b = int(rng.integers(DNS_N_RANGE[0], DNS_N_RANGE[1] + 1))
    n_k = int(rng.integers(DNS_N_RANGE[0], DNS_N_RANGE[1] + 1))
    snr_b = float(rng.uniform(*DNS_BABBLE_SNR))
    snr_k = float(rng.uniform(*DNS_TALKER_SNR))
    mic_d = _place(rng, dims_d, *MIC_Z, [])
    src_d = _place(rng, dims_d, *SRC_Z, [mic_d])
    occupied = [mic_d, src_d]
    babble_src = []
    for _ in range(n_b):
        pos = _place(rng, dims_d, 0.4, dims_d[2] - 0.3, occupied)
        occupied.append(pos)
        wav = _G["babble"][int(rng.integers(0, len(_G["babble"])))]
        babble_src.append((_fit_len(wav, n_t, rng), pos))
    candidates = [t for t in _G["talkers"] if t["id"].split("-")[0] != spk and t["id"] != utt]
    pick = rng.choice(len(candidates), size=n_k, replace=False)
    talker_src = []
    talker_ids = []
    for idx in np.atleast_1d(pick):
        tok = candidates[int(idx)]
        pos = _place(rng, dims_d, *SRC_Z, occupied)
        occupied.append(pos)
        wav = _fit_len(load_audio(tok["path"]), n_t, rng)
        talker_src.append((wav, pos))
        talker_ids.append(tok["id"])

    stems = render_stems(
        [(target, src_d), *babble_src, *talker_src], mic_d, dims_d, rt_d, n_out
    )
    t_d = stems[0]
    b_d = np.sum(stems[1 : 1 + n_b], axis=0)
    k_d = np.sum(stems[1 + n_b :], axis=0)
    mix_d = _peak_protect(t_d + _scale_to_snr(t_d, b_d, snr_b) + _scale_to_snr(t_d, k_d, snr_k))
    out_d = DATA_ROOT / "test-clean-dns" / rel
    out_d.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_d), mix_d, SR, format="FLAC")

    meta = {
        "id": utt,
        "reverb": {
            "room_m": dims_r.tolist(),
            "rt60": rt_r,
            "snr_db": snr_r,
            "mic": mic_r.tolist(),
            "src": src_r.tolist(),
            "noise_pos": npos_r.tolist(),
        },
        "dns": {
            "room_m": dims_d.tolist(),
            "rt60": rt_d,
            "n_babble": n_b,
            "n_talker": n_k,
            "snr_babble_db": snr_b,
            "snr_talker_db": snr_k,
            "mic": mic_d.tolist(),
            "src": src_d.tolist(),
            "talker_ids": talker_ids,
        },
    }
    return meta


def copy_transcripts(dest: Path) -> None:
    src = DATA_ROOT / "test-clean"
    for trans in src.rglob("*.trans.txt"):
        rel = trans.relative_to(src)
        d = dest / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        if not d.exists():
            shutil.copy2(trans, d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 2))
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    stationary = sorted((NOISE_ROOT / "synthetic").glob("*.wav")) + sorted(
        (NOISE_ROOT / "ms_snsd").glob("AirConditioner_*.wav")
    ) + sorted((NOISE_ROOT / "ms_snsd").glob("VacuumCleaner_*.wav")) + sorted(
        (NOISE_ROOT / "ms_snsd").glob("CopyMachine_*.wav")
    )
    babble = sorted((NOISE_ROOT / "ms_snsd").glob("Babble_*.wav"))
    if not stationary or not babble:
        raise SystemExit("noise banks missing under data/noise")

    items = load_split("test-clean")
    talkers = load_split("test-other")
    if args.limit:
        items = items[: args.limit]

    for dest in (DATA_ROOT / "test-clean-reverb", DATA_ROOT / "test-clean-dns"):
        dest.mkdir(parents=True, exist_ok=True)
        copy_transcripts(dest)

    meta_path = Path("/workspace/SpeechPTQ/data/robustness_meta.jsonl")
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[robust] n={len(items)} workers={args.workers} "
        f"stationary={len(stationary)} babble={len(babble)} talkers={len(talkers)}",
        flush=True,
    )
    done = 0
    with meta_path.open("w") as fout, ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=([str(x) for x in stationary], [str(x) for x in babble], talkers),
    ) as ex:
        futs = [ex.submit(process_item, it) for it in items]
        for fut in as_completed(futs):
            rec = fut.result()
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            done += 1
            if done % 50 == 0 or done == len(items):
                print(f"[robust] {done}/{len(items)}", flush=True)
    print(f"[robust] wrote {meta_path}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
