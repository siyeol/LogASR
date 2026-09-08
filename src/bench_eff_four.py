"""Four efficiency metrics: H2D MiB, GEMM input bytes, utt/s@4GB, RTF@bs=1.

Recipes: FP16 / W4A8 baseline / W4A4 ours.
Runtime uses packed W4 (bitsandbytes) for both W4 columns — activations
stay FP16 in that kernel, so utt/s and RTF match; GEMM bytes still use
A8 vs A4 operand widths from a hooked generate.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_iso_resource import (  # noqa: E402
    DEVICE,
    dummy_waves,
    is_lm_head,
    mib,
    peak_mib,
    replace_linear4bit,
    reset,
    time_qwen3,
    time_whisper,
)
from data import SR  # noqa: E402
from kv_template import pad_to_30s  # noqa: E402
from quant import is_language_decoder, is_speech_encoder  # noqa: E402
from run_eval import generate_from_features, load_model  # noqa: E402

OUT = Path("/workspace/SpeechPTQ/results/diagnostics/eff_table_four.json")
BITS = json.loads(Path("/workspace/SpeechPTQ/results/diagnostics/param_bits.json").read_text())
AUDIO_SEC = 8.0
MAX_NEW = 24
BUDGET_MIB = 4096
WHISPER_BATCHES = (1, 2, 4, 8, 12, 16)
QWEN_BATCHES = (1, 2, 4, 8, 12, 16, 24)


def pcie_gbps(cal_mib: int = 512) -> float:
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
    del host, dev
    return (nbytes / 1e9) / (ms / 1e3)


def h2d_ms(mib_val: float, gbps: float) -> float:
    return (mib_val / 1024.0) / gbps * 1e3


def iso4(rows: list[dict]) -> dict | None:
    cand = [r for r in rows if r.get("peak_mib") is not None and r["peak_mib"] <= BUDGET_MIB]
    if not cand:
        return None
    return max(cand, key=lambda r: r["utt_s"])


def rtf_bs1(rows: list[dict]) -> float | None:
    row = next((r for r in rows if r.get("batch") == 1), None)
    if row is None:
        return None
    return round(row["ms_per_utt"] / 1000.0 / AUDIO_SEC, 6)


def sweep(fn, batches) -> list[dict]:
    rows = []
    for bs in batches:
        reset()
        try:
            row = fn(bs)
            rows.append(row)
            print(f"  bs={bs} utt/s={row['utt_s']} peak={row['peak_mib']}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"  bs={bs} OOM", flush=True)
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"  bs={bs} fail {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
            break
    return rows


def classify(name: str, family: str) -> str:
    if is_lm_head(name):
        return "lm_head"
    if is_speech_encoder(name):
        return "encoder"
    if is_language_decoder(name):
        return "decoder"
    if family == "qwen" and "projector" in name.lower():
        return "projector"
    return "other"


def xw_bytes(family: str, kind: str, bucket: str) -> tuple[float, float]:
    """(bytes_per_act, bytes_per_weight) for one GEMM operand pair."""
    if kind == "fp16" or bucket in ("lm_head", "projector", "other"):
        return 2.0, 2.0
    # Quantized Linear: W4. Activations follow the recipe + Qwen LLM A16.
    w = 0.5
    if family == "qwen" and bucket == "decoder":
        return 2.0, w
    if kind == "w4a8":
        return 1.0, w
    if kind == "w4a4":
        return 0.5, w
    raise ValueError(kind)


@torch.inference_mode()
def gemm_calls_whisper(model, processor) -> list[tuple[str, int, int, int]]:
    calls: list[tuple[str, int, int, int]] = []
    hooks = []

    def make_hook(name: str):
        def hook(mod, inp, _out):
            x = inp[0]
            k = int(x.shape[-1])
            m = int(x.numel() // k)
            n = int(mod.out_features)
            calls.append((name, m, k, n))

        return hook

    for name, mod in model.named_modules():
        if type(mod) is nn.Linear:
            hooks.append(mod.register_forward_hook(make_hook(name)))
    waves = [pad_to_30s(w) for w in dummy_waves(1, AUDIO_SEC)]
    feats = processor(waves, sampling_rate=SR, return_tensors="pt").input_features
    feats = feats.to(DEVICE, dtype=torch.float16)
    generate_from_features(model, processor, feats, MAX_NEW)
    torch.cuda.synchronize()
    for h in hooks:
        h.remove()
    return calls


@torch.inference_mode()
def gemm_calls_qwen(model, processor) -> list[tuple[str, int, int, int]]:
    calls: list[tuple[str, int, int, int]] = []
    hooks = []

    def make_hook(name: str):
        def hook(mod, inp, _out):
            x = inp[0]
            if not torch.is_tensor(x) or x.ndim < 2:
                return
            k = int(x.shape[-1])
            if not hasattr(mod, "in_features") or k != int(mod.in_features):
                return
            m = int(x.numel() // k)
            n = int(mod.out_features)
            calls.append((name, m, k, n))

        return hook

    for name, mod in model.named_modules():
        if type(mod) is nn.Linear:
            hooks.append(mod.register_forward_hook(make_hook(name)))
    waves = dummy_waves(1, AUDIO_SEC)
    inputs = processor.apply_transcription_request(audio=waves, language="English")
    inputs = inputs.to(DEVICE, model.dtype)
    model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False, num_beams=1, use_cache=True)
    torch.cuda.synchronize()
    for h in hooks:
        h.remove()
    return calls


def reduce_gemm(calls, family: str) -> dict:
    out = {}
    for kind in ("fp16", "w4a8", "w4a4"):
        x_b = w_b = 0.0
        by = {"encoder": 0.0, "decoder": 0.0, "lm_head": 0.0, "projector": 0.0, "other": 0.0}
        n_calls = 0
        for name, m, k, n in calls:
            bucket = classify(name, family)
            xb, wb = xw_bytes(family, kind, bucket)
            layer = m * k * xb + n * k * wb
            x_b += m * k * xb
            w_b += n * k * wb
            by[bucket] += layer
            n_calls += 1
        tot = x_b + w_b
        out[kind] = {
            "gemm_input_mib": round(tot / 1024**2, 3),
            "act_mib": round(x_b / 1024**2, 3),
            "weight_operand_mib": round(w_b / 1024**2, 3),
            "n_calls": n_calls,
            "by_bucket_mib": {k: round(v / 1024**2, 3) for k, v in by.items() if v},
        }
    return out


def pack_row(name, family, kind, h2d, gbps, gemm, rows):
    iso = iso4(rows)
    return {
        "name": name,
        "family": family,
        "kind": kind,
        "h2d_mib": round(h2d, 1),
        "h2d_ms": round(h2d_ms(h2d, gbps), 1),
        "gemm_input_mib": gemm["gemm_input_mib"],
        "gemm_act_mib": gemm["act_mib"],
        "gemm_weight_mib": gemm["weight_operand_mib"],
        "utt_s_4gb": None if iso is None else iso["utt_s"],
        "iso_4gb": iso,
        "rtf_bs1": rtf_bs1(rows),
        "ms_per_utt_bs1": next((r["ms_per_utt"] for r in rows if r.get("batch") == 1), None),
        "sweep": rows,
    }


def qwen_load(model_id: str):
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    model.to(DEVICE).eval()
    return proc, model


def main():
    gbps = pcie_gbps()
    print(f"PCIe {gbps:.2f} GB/s", flush=True)
    payload = {
        "pcie_GBps": round(gbps, 2),
        "audio_sec": AUDIO_SEC,
        "max_new_tokens": MAX_NEW,
        "budget_mib": BUDGET_MIB,
        "note": (
            "H2D is packed weight bytes. GEMM input bytes are hooked from one "
            "bs=1 generate (8s audio, Whisper padded to 30s) with A8 vs A4 "
            "operand widths. utt/s and RTF use packed W4 (bnb fp4) for both "
            "W4A8 and W4A4; that kernel computes activations in FP16."
        ),
        "rows": [],
    }

    w_fp = BITS["whisper"]["recipes"]["fp16"]["total_mib"]
    w_w4 = BITS["whisper"]["recipes"]["paper"]["total_mib"]
    q_fp = BITS["qwen3"]["recipes"]["fp16"]["total_mib"]
    q_w4 = BITS["qwen3"]["recipes"]["paper"]["total_mib"]

    print("== whisper gemm hooks + fp16 runtime ==", flush=True)
    reset()
    proc, model = load_model("openai/whisper-large-v3")
    w_calls = gemm_calls_whisper(model, proc)
    w_gemm = reduce_gemm(w_calls, "whisper")
    print("  gemm", {k: v["gemm_input_mib"] for k, v in w_gemm.items()}, flush=True)
    w_fp16_rows = sweep(lambda bs: time_whisper(model, proc, bs, max_new=MAX_NEW), WHISPER_BATCHES)
    payload["rows"].append(pack_row("whisper_fp16", "whisper", "fp16", w_fp, gbps, w_gemm["fp16"], w_fp16_rows))
    del model
    reset()

    print("== whisper packed W4 runtime (W4A8 & W4A4) ==", flush=True)
    proc, model = load_model("openai/whisper-large-v3")
    n = replace_linear4bit(model, "all", torch.float16, "fp4")
    model.to(DEVICE)
    print("  bnb replaced", n, "load", round(mib(), 1), flush=True)
    w_w4_rows = sweep(lambda bs: time_whisper(model, proc, bs, max_new=MAX_NEW), WHISPER_BATCHES)
    payload["rows"].append(
        pack_row("whisper_w4a8_baseline", "whisper", "w4a8", w_w4, gbps, w_gemm["w4a8"], w_w4_rows)
    )
    payload["rows"].append(
        pack_row("whisper_w4a4_ours", "whisper", "w4a4", w_w4, gbps, w_gemm["w4a4"], w_w4_rows)
    )
    del model
    reset()

    print("== qwen gemm hooks + bf16 runtime ==", flush=True)
    reset()
    proc, model = qwen_load("Qwen/Qwen3-ASR-1.7B-hf")
    q_calls = gemm_calls_qwen(model, proc)
    q_gemm = reduce_gemm(q_calls, "qwen")
    print("  gemm", {k: v["gemm_input_mib"] for k, v in q_gemm.items()}, flush=True)
    q_fp_rows = sweep(lambda bs: time_qwen3(model, proc, bs, max_new=MAX_NEW), QWEN_BATCHES)
    payload["rows"].append(pack_row("qwen_fp16", "qwen", "fp16", q_fp, gbps, q_gemm["fp16"], q_fp_rows))
    del model
    reset()

    print("== qwen packed W4 runtime (enc+LLM, W4A8 & W4A4) ==", flush=True)
    proc, model = qwen_load("Qwen/Qwen3-ASR-1.7B-hf")
    n = replace_linear4bit(model, "all", torch.bfloat16, "fp4")
    model.to(DEVICE)
    print("  bnb replaced", n, "load", round(mib(), 1), flush=True)
    q_w4_rows = sweep(lambda bs: time_qwen3(model, proc, bs, max_new=MAX_NEW), QWEN_BATCHES)
    payload["rows"].append(
        pack_row("qwen_w4a8_baseline", "qwen", "w4a8", q_w4, gbps, q_gemm["w4a8"], q_w4_rows)
    )
    payload["rows"].append(
        pack_row("qwen_w4a4_ours", "qwen", "w4a4", q_w4, gbps, q_gemm["w4a4"], q_w4_rows)
    )
    del model
    reset()

    payload["whisper_gemm_detail"] = w_gemm
    payload["qwen_gemm_detail"] = q_gemm
    OUT.write_text(json.dumps(payload, indent=2))
    print("wrote", OUT, flush=True)
    print_table(payload)


def print_table(payload: dict):
    print("\n=== efficiency table ===")
    hdr = f"{'model':8s} {'recipe':8s} {'H2D MiB':>8s} {'H2D ms':>7s} {'GEMM MiB':>9s} {'utt/s@4G':>9s} {'RTF bs1':>9s}"
    print(hdr)
    for r in payload["rows"]:
        print(
            f"{r['family']:8s} {r['kind']:8s} {r['h2d_mib']:8.1f} {r['h2d_ms']:7.1f} "
            f"{r['gemm_input_mib']:9.2f} {str(r['utt_s_4gb']):>9s} {str(r['rtf_bs1']):>9s}"
        )


if __name__ == "__main__":
    main()
