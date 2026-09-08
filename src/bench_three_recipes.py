"""H2D / iso-4GB throughput / RTF(bs=1) for FP16, grouped W4A8 baseline, proposed."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from data import SR
from kv_template import pad_to_30s
from quant import replace_linears
from run_eval import generate_from_features, load_model
from run_eval_qwen3 import compressed_tensors_load_kwargs

DEVICE = torch.device("cuda")
OUT = Path("/workspace/SpeechPTQ/results/diagnostics/three_recipes_iso.json")
BITS = json.loads(Path("/workspace/SpeechPTQ/results/diagnostics/param_bits.json").read_text())
AUDIO_SEC = 8.0
MAX_NEW = 24
BUDGET_MIB = 4096


def mib() -> float:
    return torch.cuda.memory_allocated() / (1024**2)


def peak_mib() -> float:
    return torch.cuda.max_memory_allocated() / (1024**2)


def reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def dummy_waves(n: int) -> list[np.ndarray]:
    t = int(AUDIO_SEC * SR)
    rng = np.random.default_rng(0)
    return [rng.standard_normal(t, dtype=np.float32) * 0.01 for _ in range(n)]


def pcie_gbps(cal_mib: int = 512) -> tuple[float, float]:
    nbytes = cal_mib * 1024 * 1024
    n = nbytes // 2
    host = torch.empty(n, dtype=torch.float16, pin_memory=True)
    dev = torch.empty(n, dtype=torch.float16, device=DEVICE)
    for _ in range(2):
        dev.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
    iters = 8
    t0 = time.perf_counter()
    for _ in range(iters):
        dev.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    gbps = (nbytes / 1e9) / (ms / 1e3)
    del host, dev
    return gbps, ms


@torch.inference_mode()
def time_whisper(model, processor, bs: int, n_iters: int = 3) -> dict:
    waves = [pad_to_30s(w) for w in dummy_waves(bs)]
    feats = processor(waves, sampling_rate=SR, return_tensors="pt").input_features
    feats = feats.to(DEVICE, dtype=torch.float16)

    def run():
        generate_from_features(model, processor, feats, MAX_NEW)

    run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n = bs * n_iters
    return {
        "batch": bs,
        "utt_s": round(n / elapsed, 4),
        "ms_per_utt": round(elapsed / n * 1e3, 2),
        "peak_mib": round(peak_mib(), 1),
        "rtf_vs_8s": round((elapsed / n) / AUDIO_SEC, 6),
        "rtf_vs_30s_pad": round((elapsed / n) / 30.0, 6),
    }


@torch.inference_mode()
def time_qwen(model, processor, bs: int, n_iters: int = 2) -> dict:
    waves = dummy_waves(bs)

    def run():
        inputs = processor.apply_transcription_request(audio=waves, language="English")
        inputs = inputs.to(DEVICE, model.dtype)
        model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False, num_beams=1, use_cache=True)

    run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    n = bs * n_iters
    return {
        "batch": bs,
        "utt_s": round(n / elapsed, 4),
        "ms_per_utt": round(elapsed / n * 1e3, 2),
        "peak_mib": round(peak_mib(), 1),
        "rtf_vs_8s": round((elapsed / n) / AUDIO_SEC, 6),
    }


def sweep(fn, batches) -> list[dict]:
    rows = []
    for bs in batches:
        reset()
        try:
            row = fn(bs)
            rows.append(row)
            print(f"  bs={bs} utt/s={row['utt_s']} peak={row['peak_mib']} RTF8={row['rtf_vs_8s']}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  bs={bs} OOM", flush=True)
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"  bs={bs} fail {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
            break
    return rows


def iso4(rows: list[dict], load_mib: float, packed_mib: float) -> dict | None:
    """Pick max utt/s among batches whose packed-corrected peak fits in 4GB."""
    cand = []
    for r in rows:
        corr = r["peak_mib"] - load_mib + packed_mib
        r = {**r, "peak_packed_mib": round(corr, 1)}
        if corr <= BUDGET_MIB:
            cand.append(r)
    if not cand:
        return None
    return max(cand, key=lambda x: x["utt_s"])


def load_qwen(model_id: str, dequant_gptq: bool):
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    proc = AutoProcessor.from_pretrained(model_id)
    kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if dequant_gptq:
        kw.update(compressed_tensors_load_kwargs(model_id))
    model = AutoModelForMultimodalLM.from_pretrained(model_id, **kw)
    model.to(DEVICE).eval()
    return proc, model


def pack_row(name, family, kind, packed_mib, load_mib, gbps, rows):
    h2d_ms = (packed_mib / 1024) / gbps * 1e3  # MiB -> GB
    bs1 = next((r for r in rows if r["batch"] == 1), None)
    iso = iso4(rows, load_mib, packed_mib)
    return {
        "name": name,
        "family": family,
        "kind": kind,
        "h2d_mib": round(packed_mib, 1),
        "h2d_ms": round(h2d_ms, 1),
        "load_mib_fake": round(load_mib, 1),
        "rtf_bs1": None if bs1 is None else bs1["rtf_vs_8s"],
        "rtf_bs1_vs_30s_pad": None if bs1 is None else bs1.get("rtf_vs_30s_pad"),
        "ms_per_utt_bs1": None if bs1 is None else bs1["ms_per_utt"],
        "utt_s_4gb": None if iso is None else iso["utt_s"],
        "iso_4gb": iso,
        "sweep": rows,
    }


def main():
    gbps, cal_ms = pcie_gbps()
    print(f"PCIe {gbps:.2f} GB/s (cal 512MiB {cal_ms:.1f}ms)", flush=True)
    w_fp = BITS["whisper"]["recipes"]["fp16"]["total_mib"]
    w_w4 = BITS["whisper"]["recipes"]["paper"]["total_mib"]
    q_fp = BITS["qwen3"]["recipes"]["fp16"]["total_mib"]
    q_w4 = BITS["qwen3"]["recipes"]["paper"]["total_mib"]

    out = {"pcie_GBps": round(gbps, 2), "audio_sec": AUDIO_SEC, "max_new_tokens": MAX_NEW, "rows": []}

    # --- Whisper FP16 ---
    print("== whisper fp16 ==", flush=True)
    reset()
    proc, model = load_model("openai/whisper-large-v3")
    load = mib()
    rows = sweep(lambda bs: time_whisper(model, proc, bs), (1, 2, 4, 8, 12, 16))
    out["rows"].append(pack_row("whisper_fp16", "whisper", "fp16", w_fp, load, gbps, rows))
    del model
    reset()

    # --- Whisper baseline: grouped uniform W4A8 ---
    print("== whisper baseline g64 W4A8 ==", flush=True)
    proc, model = load_model("openai/whisper-large-v3")
    n = replace_linears(model, {}, "uniform", "uniform_a8_dyn", "nolm", backend="fake", group_size=64)
    print(" replaced", n, flush=True)
    load = mib()
    rows = sweep(lambda bs: time_whisper(model, proc, bs), (1, 2, 4, 8, 12, 16))
    out["rows"].append(pack_row("whisper_baseline_g64_w4a8", "whisper", "baseline", w_w4, load, gbps, rows))
    del model
    reset()

    # --- Whisper proposed ---
    print("== whisper proposed log FFN A4 + A8 rest ==", flush=True)
    proc, model = load_model("openai/whisper-large-v3")
    n = replace_linears(model, {}, "log", "mixed_log_encffn_a8", "nolm", backend="fake", group_size=0)
    print(" replaced", n, flush=True)
    load = mib()
    rows = sweep(lambda bs: time_whisper(model, proc, bs), (1, 2, 4, 8, 12, 16))
    out["rows"].append(pack_row("whisper_proposed", "whisper", "proposed", w_w4, load, gbps, rows))
    del model
    reset()

    # --- Qwen BF16 ---
    print("== qwen bf16 ==", flush=True)
    proc, model = load_qwen("Qwen/Qwen3-ASR-1.7B-hf", False)
    load = mib()
    rows = sweep(lambda bs: time_qwen(model, proc, bs), (1, 2, 4, 8, 12))
    out["rows"].append(pack_row("qwen_bf16", "qwen", "fp16", q_fp, load, gbps, rows))
    del model
    reset()

    # --- Qwen baseline g64 enc W4A8 / dec W4A16 ---
    print("== qwen baseline g64 enc W4A8 ==", flush=True)
    proc, model = load_qwen("Qwen/Qwen3-ASR-1.7B-hf", False)
    n = replace_linears(model, {}, "uniform", "encoder_a8", "nolm", backend="fake", group_size=64)
    print(" replaced", n, flush=True)
    load = mib()
    rows = sweep(lambda bs: time_qwen(model, proc, bs), (1, 2, 4, 8, 12))
    out["rows"].append(pack_row("qwen_baseline_g64_encw4a8", "qwen", "baseline", q_w4, load, gbps, rows))
    del model
    reset()

    # --- Qwen proposed: GPTQ LLM + log enc FFN A4 / attn A8 ---
    print("== qwen proposed gptq+log ==", flush=True)
    mid = "/workspace/SpeechPTQ/models/qwen3-asr-1.7b-W4A16-G128-llm"
    proc, model = load_qwen(mid, True)
    n = replace_linears(model, {}, "log", "mixed_log_encffn_enc_a8", "encoder", backend="fake", group_size=0)
    print(" replaced", n, flush=True)
    load = mib()
    rows = sweep(lambda bs: time_qwen(model, proc, bs), (1, 2, 4, 8, 12))
    out["rows"].append(pack_row("qwen_proposed", "qwen", "proposed", q_w4, load, gbps, rows))
    del model
    reset()

    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
