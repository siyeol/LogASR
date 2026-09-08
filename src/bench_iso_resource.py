"""Four iso-resource metrics vs FP16/BF16.

1. Iso-VRAM utt/s (max batch under a peak-memory budget)
2. Decode tok/s (weight-only W4 kernels vs FP16/BF16 GEMM + e2e generate)
3. Weight traffic / H2D (GB, ms)
4. Fits / max batch

Packed runtime uses bitsandbytes Linear4bit so weights stay 4-bit in VRAM
(transformers CompressedLinear is gone; GPTQ checkpoints decompress to FP16).
Proposed = log-INT4 encoder FFN + W4A16 packed language stack.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import SR  # noqa: E402
from kv_template import pad_to_30s  # noqa: E402
from layer_groups import layer_group  # noqa: E402
from log_int4 import LogInt4Linear, LogPackedFp16Linear  # noqa: E402
from quant import is_language_decoder, is_speech_encoder  # noqa: E402
from run_eval import generate_from_encoder, generate_from_features, load_model  # noqa: E402

DEVICE = torch.device("cuda")
OUT_DIR = Path("/workspace/SpeechPTQ/results/diagnostics")
OUT = OUT_DIR / "iso_resource.json"
BUDGETS_MIB = [3072, 3584, 4096, 5120, 6144, 8192, 10240, 12288, 16384]
SKIP_LM = ("lm_head", "proj_out")

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def mib() -> float:
    return torch.cuda.memory_allocated() / (1024**2)


def peak_mib() -> float:
    return torch.cuda.max_memory_allocated() / (1024**2)


def reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def save(payload: dict):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prev = {}
    if OUT.exists():
        try:
            prev = json.loads(OUT.read_text())
        except json.JSONDecodeError:
            prev = {}
    prev.update(payload)
    OUT.write_text(json.dumps(prev, indent=2))
    print("wrote", OUT, "keys", sorted(prev.keys()))


def dummy_waves(n: int, sec: float = 8.0) -> list[np.ndarray]:
    t = int(sec * SR)
    rng = np.random.default_rng(0)
    return [rng.standard_normal(t, dtype=np.float32) * 0.01 for _ in range(n)]


def is_lm_head(name: str) -> bool:
    n = name.lower()
    return n.endswith("lm_head") or n.endswith("proj_out") or ".proj_out" in n


def module_bytes(mod: nn.Module) -> int:
    n = 0
    seen = set()
    for t in list(mod.parameters()) + list(mod.buffers()):
        if not torch.is_tensor(t):
            continue
        ptr = t.data_ptr() if t.untyped_storage().size() else id(t)
        if ptr in seen:
            continue
        seen.add(ptr)
        try:
            n += t.untyped_storage().nbytes()
        except Exception:
            n += t.numel() * t.element_size()
    return n


def replace_linear4bit(model: nn.Module, scope: str, compute_dtype, quant_type: str = "fp4") -> int:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit

    n = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear) or isinstance(mod, Linear4bit):
            continue
        if is_lm_head(name):
            continue
        if scope == "decoder" and not is_language_decoder(name):
            continue
        if scope == "encoder" and not is_speech_encoder(name):
            continue
        if scope == "enc_ffn":
            if not is_speech_encoder(name):
                continue
            low = name.lower()
            if not any(k in low for k in ("fc1", "fc2", "up_proj", "down_proj", "gate_proj", "mlp")):
                continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        new = Linear4bit(
            mod.in_features,
            mod.out_features,
            bias=mod.bias is not None,
            compute_dtype=compute_dtype,
            compress_statistics=True,
            quant_type=quant_type,
            device=mod.weight.device,
        )
        with torch.no_grad():
            new.weight = bnb.nn.Params4bit(
                mod.weight.data.detach().contiguous(),
                requires_grad=False,
                compress_statistics=True,
                quant_type=quant_type,
                module=new,
            )
            if mod.bias is not None:
                new.bias = nn.Parameter(mod.bias.data.detach().clone(), requires_grad=False)
        setattr(parent, child, new)
        n += 1
    return n


def replace_enc_ffn_log_int4(model: nn.Module) -> int:
    n = 0
    for name, mod in list(model.named_modules()):
        if type(mod) is not nn.Linear:
            continue
        if layer_group(name) != "enc_ffn":
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        amax = float(mod.weight.detach().abs().amax().item())
        inner = LogInt4Linear(mod, name, "log_token", amax)

        class _Cast(nn.Module):
            def __init__(self, core: LogInt4Linear):
                super().__init__()
                self.core = core

            def forward(self, x):
                y = self.core(x.to(torch.float16))
                return y.to(x.dtype)

        setattr(parent, child, _Cast(inner))
        n += 1
    return n


def replace_enc_ffn_pack_fp16(model: nn.Module) -> int:
    n = 0
    for name, mod in list(model.named_modules()):
        if type(mod) is not nn.Linear:
            continue
        if layer_group(name) != "enc_ffn":
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LogPackedFp16Linear(mod, name))
        n += 1
    return n


def sync_ms(fn, warmup=20, iters=80) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


# ---------------------------------------------------------------------------
# Triton weight-only W4 GEMV (decode M=1) — affine symmetric, group 128
# ---------------------------------------------------------------------------
def pack_w4_symmetric(w: torch.Tensor, group: int = 128):
    n, k = w.shape
    assert k % group == 0
    wf = w.float().view(n, k // group, group)
    scale = wf.abs().amax(-1).clamp_min(1e-8) / 7.0
    q = torch.round(wf / scale.unsqueeze(-1)).clamp(-8, 7).to(torch.int8)
    q = q.view(n, k)
    u = (q.to(torch.int16) + 8).to(torch.uint8)
    packed = (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()
    return packed, scale.to(w.dtype).contiguous()


@triton.jit
def _w4_gemv_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    y_ptr,
    N,
    K,
    G,
    stride_qn,
    stride_sn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_K2: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    n_mask = offs_n < N
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        x = tl.load(x_ptr + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        packed_k = (k0 // 2) + tl.arange(0, BLOCK_K2)
        pk_mask = packed_k < (K // 2)
        packed = tl.load(
            q_ptr + offs_n[:, None] * stride_qn + packed_k[None, :],
            mask=n_mask[:, None] & pk_mask[None, :],
            other=0,
        )
        lo = (packed & 0xF).to(tl.int32) - 8
        hi = (packed >> 4).to(tl.int32) - 8
        q = tl.reshape(tl.stack((lo, hi), axis=2), (BLOCK_N, BLOCK_K)).to(tl.float32)
        g = (k0 // G) + tl.arange(0, BLOCK_K) // G
        scale = tl.load(
            s_ptr + offs_n[:, None] * stride_sn + g[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=1.0,
        ).to(tl.float32)
        acc += tl.sum(q * scale * x[None, :], axis=1)
    tl.store(y_ptr + offs_n, acc, mask=n_mask)


def triton_w4_gemv(x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, group: int = 128):
    n, k2 = packed.shape
    k = k2 * 2
    y = torch.empty(n, device=x.device, dtype=torch.float32)
    bn = 8
    bk = 128
    _w4_gemv_kernel[(triton.cdiv(n, bn),)](
        x.reshape(-1).contiguous(),
        packed,
        scale,
        y,
        n,
        k,
        group,
        packed.stride(0),
        scale.stride(0),
        BLOCK_N=bn,
        BLOCK_K=bk,
        BLOCK_K2=bk // 2,
        num_warps=4,
    )
    return y.to(x.dtype)


def phase_gemm() -> dict:
    print("== GEMM / decode tok-equivalent microbench ==")
    rows = []
    torch.backends.cuda.matmul.allow_tf32 = True

    def add(row):
        print(json.dumps(row))
        rows.append(row)

    # --- FP16 vs tinygemm INT4 vs bnb vs triton W4 ---
    import bitsandbytes as bnb
    from torchao.quantization import Int4WeightOnlyConfig, quantize_
    from torchao.quantization.quantize_.workflows.int4.int4_packing_format import (
        Int4PackingFormat,
    )

    cfg_tile = Int4WeightOnlyConfig(
        group_size=128,
        int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
        set_inductor_config=False,
    )

    shapes = [
        ("qwen_llm_down", 6144, 2048, [1, 8, 64, 256, 1024, 2048]),
        ("qwen_llm_attn", 2048, 2048, [1, 8, 64, 256]),
        ("whisper_dec_fc1", 1280, 5120, [1, 8, 32, 128, 512, 1500]),
        ("whisper_enc_fc1", 1280, 5120, [1500, 6000, 24000]),
    ]
    for tag, k, n, ms in shapes:
        for dtype, dt_name in ((torch.float16, "fp16"), (torch.bfloat16, "bf16")):
            if tag.startswith("whisper") and dtype is torch.bfloat16:
                continue
            if tag.startswith("qwen") and dtype is torch.float16 and tag != "qwen_llm_down":
                continue
            lin = nn.Linear(k, n, bias=True, device=DEVICE, dtype=dtype)
            with torch.no_grad():
                lin.weight.mul_(0.02)
            packed, scale = pack_w4_symmetric(lin.weight)
            # bnb
            b = bnb.nn.Linear4bit(k, n, bias=True, compute_dtype=dtype, quant_type="fp4")
            b.load_state_dict({"weight": lin.weight.detach().cpu(), "bias": lin.bias.detach().cpu()}, strict=False)
            b = b.to(DEVICE)
            # tinygemm (bf16 only)
            qtile = None
            if dtype is torch.bfloat16:
                qtile = nn.Linear(k, n, bias=True, device=DEVICE, dtype=dtype)
                with torch.no_grad():
                    qtile.weight.copy_(lin.weight)
                    qtile.bias.copy_(lin.bias)
                try:
                    quantize_(qtile, cfg_tile)
                except Exception as e:
                    qtile = None
                    print(" tinygemm skip", e)

            for m in ms:
                x = torch.randn(m, k, device=DEVICE, dtype=dtype)
                fp = sync_ms(lambda: lin(x), warmup=15, iters=50)
                row = {
                    "tag": tag,
                    "m": m,
                    "k": k,
                    "n": n,
                    "dtype": dt_name,
                    "fp_ms": round(fp, 4),
                }
                try:
                    bm = sync_ms(lambda: b(x), warmup=15, iters=50)
                    row["bnb_fp4_ms"] = round(bm, 4)
                    row["bnb_speedup"] = round(fp / bm, 3)
                except Exception as e:
                    row["bnb_err"] = str(e)[:120]
                if qtile is not None:
                    try:
                        tm = sync_ms(lambda: qtile(x), warmup=15, iters=50)
                        row["tinygemm_ms"] = round(tm, 4)
                        row["tinygemm_speedup"] = round(fp / tm, 3)
                    except Exception as e:
                        row["tinygemm_err"] = str(e)[:120]
                if m == 1:
                    x1 = x.reshape(-1)
                    try:
                        tr = sync_ms(lambda: triton_w4_gemv(x1, packed, scale), warmup=25, iters=80)
                        row["triton_w4_ms"] = round(tr, 4)
                        row["triton_speedup"] = round(fp / tr, 3)
                    except Exception as e:
                        row["triton_err"] = f"{type(e).__name__}: {e}"[:200]
                add(row)
            del lin, b, qtile
            reset()

    # Theoretical DRAM bytes / decode step (read all weights once)
    from json import loads

    bits = loads((OUT_DIR / "param_bits.json").read_text())
    traffic = {}
    for model in ("whisper", "qwen3"):
        fp = bits[model]["recipes"]["fp16"]["total_mib"]
        paper = bits[model]["recipes"]["paper"]["total_mib"]
        traffic[model] = {
            "fp16_weight_mib_per_step": fp,
            "paper_weight_mib_per_step": paper,
            "traffic_reduction": round(fp / paper, 3),
        }
    return {"gemm": rows, "weight_traffic_mib": traffic}


def dir_weight_bytes(path: str) -> int:
    p = Path(path)
    if not p.exists():
        return 0
    total = 0
    for f in p.rglob("*"):
        if f.suffix in {".safetensors", ".bin"}:
            total += f.stat().st_size
    return total


def pcie_copy_ms(nbytes: int, iters: int = 8) -> float:
    """Pinned H2D of `nbytes` (rounded up to 2)."""
    n = max(nbytes // 2, 1)
    host = torch.empty(n, dtype=torch.float16, pin_memory=True)
    dev = torch.empty(n, dtype=torch.float16, device=DEVICE)
    for _ in range(2):
        dev.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dev.copy_(host, non_blocking=True)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def hf_snapshot(repo: str) -> str | None:
    home = Path(os.environ.get("HF_HOME", "/workspace/.hf_home"))
    name = "models--" + repo.replace("/", "--")
    snaps = home / "hub" / name / "snapshots"
    if not snaps.is_dir():
        return None
    cands = [p for p in snaps.iterdir() if p.is_dir()]
    return str(cands[0]) if cands else None


def phase_h2d() -> dict:
    print("== H2D / checkpoint bytes ==")
    rows = {}
    pairs = {
        "whisper_fp16": hf_snapshot("openai/whisper-large-v3"),
        "whisper_gptq": "/workspace/SpeechPTQ/models/whisper-large-v3-W4A16-G128",
        "qwen3_bf16": hf_snapshot("Qwen/Qwen3-ASR-1.7B-hf"),
        "qwen3_gptq_llm": "/workspace/SpeechPTQ/models/qwen3-asr-1.7b-W4A16-G128-llm",
    }
    # calibrate PCIe with 512 MiB
    cal_bytes = 512 * 1024 * 1024
    cal_ms = pcie_copy_ms(cal_bytes)
    gbps = (cal_bytes / 1e9) / (cal_ms / 1e3)
    rows["pcie"] = {"cal_mib": 512, "ms": round(cal_ms, 2), "GBps": round(gbps, 2)}
    print(" PCIe", rows["pcie"])
    for name, path in pairs.items():
        if not path or not Path(path).exists():
            rows[name] = {"error": f"missing {path}"}
            continue
        b = dir_weight_bytes(path)
        pred_ms = (b / 1e9) / gbps * 1e3
        # actual copy of min(files, 2GB) to avoid huge alloc
        copy_b = min(b, 2 * 1024**3) if b else 0
        actual = pcie_copy_ms(copy_b) if copy_b else None
        scale = b / copy_b if copy_b else None
        rows[name] = {
            "path": path,
            "weight_bytes": b,
            "weight_mib": round(b / 1024**2, 1),
            "predicted_h2d_ms": round(pred_ms, 1),
            "measured_partial_ms": None if actual is None else round(actual, 1),
            "extrapolated_h2d_ms": None if actual is None else round(actual * scale, 1),
        }
        print(" ", name, rows[name])

    # timed from_pretrained (real load)
    def time_load(label, fn):
        reset()
        t0 = time.perf_counter()
        obj = fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        allocated = mib()
        nbytes = module_bytes(obj[1] if isinstance(obj, tuple) else obj)
        del obj
        reset()
        return {
            "label": label,
            "load_s": round(elapsed, 2),
            "alloc_mib": round(allocated, 1),
            "storage_mib": round(nbytes / 1024**2, 1),
        }

    loads = []
    try:
        loads.append(time_load("whisper_fp16", lambda: load_model("openai/whisper-large-v3")))
        print(" load", loads[-1])
    except Exception as e:
        loads.append({"label": "whisper_fp16", "error": str(e)})
    try:
        loads.append(
            time_load("whisper_gptq_decomp", lambda: load_model("/workspace/SpeechPTQ/models/whisper-large-v3-W4A16-G128"))
        )
        print(" load", loads[-1])
    except Exception as e:
        loads.append({"label": "whisper_gptq_decomp", "error": str(e)})
    rows["from_pretrained"] = loads
    return {"h2d": rows}


def iso_from_sweep(rows: list[dict]) -> dict:
    out = {}
    for b in BUDGETS_MIB:
        cand = [r for r in rows if r.get("peak_mib") is not None and r["peak_mib"] <= b]
        out[str(b)] = max(cand, key=lambda r: r["utt_s"]) if cand else None
    return out


def max_batch(rows: list[dict]) -> int | None:
    ok = [r["batch"] for r in rows if "utt_s" in r]
    return max(ok) if ok else None


@torch.inference_mode()
def time_whisper(model, processor, bs: int, n_iters: int = 3, max_new: int = 24) -> dict:
    waves = [pad_to_30s(w) for w in dummy_waves(bs, 8.0)]
    feats = processor(waves, sampling_rate=SR, return_tensors="pt").input_features
    feats = feats.to(DEVICE, dtype=torch.float16)

    def run():
        generate_from_features(model, processor, feats, max_new)

    run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "batch": bs,
        "utt_s": round(bs * n_iters / elapsed, 4),
        "ms_per_utt": round(elapsed / (bs * n_iters) * 1e3, 2),
        "peak_mib": round(peak_mib(), 1),
        "alloc_mib": round(mib(), 1),
    }


@torch.inference_mode()
def time_whisper_decode(model, processor, max_new: int = 32) -> dict:
    waves = [pad_to_30s(w) for w in dummy_waves(1, 8.0)]
    feats = processor(waves, sampling_rate=SR, return_tensors="pt").input_features
    feats = feats.to(DEVICE, dtype=torch.float16)
    enc = model.model.encoder(feats)
    generate_from_encoder(model, enc, max_new)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = generate_from_encoder(model, enc, max_new)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ntok = int(out.shape[-1])
    return {
        "decode_s": round(elapsed, 4),
        "out_tokens": ntok,
        "tok_s": round(ntok / elapsed, 2),
        "max_new": max_new,
    }


@torch.inference_mode()
def time_qwen3(model, processor, bs: int, n_iters: int = 2, max_new: int = 24) -> dict:
    waves = dummy_waves(bs, 8.0)

    def run():
        inputs = processor.apply_transcription_request(audio=waves, language="English")
        inputs = inputs.to(DEVICE, model.dtype)
        model.generate(**inputs, max_new_tokens=max_new, do_sample=False, num_beams=1, use_cache=True)

    run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "batch": bs,
        "utt_s": round(bs * n_iters / elapsed, 4),
        "ms_per_utt": round(elapsed / (bs * n_iters) * 1e3, 2),
        "peak_mib": round(peak_mib(), 1),
        "alloc_mib": round(mib(), 1),
    }


@torch.inference_mode()
def time_qwen3_decode(model, processor, max_new: int = 32) -> dict:
    waves = dummy_waves(1, 8.0)
    inputs = processor.apply_transcription_request(audio=waves, language="English")
    inputs = inputs.to(DEVICE, model.dtype)
    # warmup
    model.generate(**inputs, max_new_tokens=max_new, do_sample=False, num_beams=1, use_cache=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False, num_beams=1, use_cache=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ntok = int(out.shape[-1])
    return {
        "decode_s": round(elapsed, 4),
        "out_tokens": ntok,
        "tok_s": round(ntok / elapsed, 2),
        "max_new": max_new,
    }


def sweep(fn, batches) -> list[dict]:
    rows = []
    for bs in batches:
        reset()
        try:
            row = fn(bs)
            rows.append(row)
            print(f"  bs={bs} utt/s={row['utt_s']} peak={row['peak_mib']}MiB")
        except torch.cuda.OutOfMemoryError:
            print(f"  bs={bs} OOM")
            torch.cuda.empty_cache()
            rows.append({"batch": bs, "oom": True})
            break
        except Exception as e:
            print(f"  bs={bs} fail {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
            rows.append({"batch": bs, "error": f"{type(e).__name__}: {e}"})
            break
    return rows


def pack_result(method: str, kind: str, load: float, rows: list[dict], extra=None, decode=None):
    d = {
        "method": method,
        "kind": kind,
        "load_mib": round(load, 1),
        "storage_mib": extra.get("storage_mib") if extra else None,
        "replaced": extra.get("replaced") if extra else None,
        "sweep": rows,
        "iso": iso_from_sweep([r for r in rows if "utt_s" in r]),
        "max_batch": max_batch(rows),
        "decode": decode,
        "fits": {str(b): load <= b for b in BUDGETS_MIB},
    }
    if extra:
        for k, v in extra.items():
            if k not in d:
                d[k] = v
    return d


def phase_whisper() -> dict:
    print("== Whisper e2e ==")
    out = []
    batches = (1, 2, 4, 8, 12, 16, 24, 32, 48)

    print("-- whisper_fp16 --")
    reset()
    proc, model = load_model("openai/whisper-large-v3")
    load = mib()
    storage = module_bytes(model) / 1024**2
    dec = time_whisper_decode(model, proc)
    print("  decode", dec)
    rows = sweep(lambda bs: time_whisper(model, proc, bs), batches)
    out.append(pack_result("whisper_fp16", "existing", load, rows, {"storage_mib": round(storage, 1)}, dec))
    del model
    reset()
    save({"whisper": out})

    print("-- whisper_bnb_w4_all_except_lm (existing-style 4-bit runtime) --")
    proc, model = load_model("openai/whisper-large-v3")
    n = replace_linear4bit(model, "all", torch.float16, "fp4")
    model.to(DEVICE)
    torch.cuda.synchronize()
    load = mib()
    storage = module_bytes(model) / 1024**2
    print("  replaced", n, "load", round(load, 1), "storage", round(storage, 1))
    try:
        dec = time_whisper_decode(model, proc)
    except Exception as e:
        dec = {"error": str(e)}
    print("  decode", dec)
    rows = sweep(lambda bs: time_whisper(model, proc, bs), batches)
    out.append(
        pack_result(
            "whisper_bnb_w4_all",
            "existing_4bit_runtime",
            load,
            rows,
            {"storage_mib": round(storage, 1), "replaced": n},
            dec,
        )
    )
    del model
    reset()
    save({"whisper": out})

    print("-- whisper_proposed log-int4 enc-FFN + bnb W4 decoder --")
    proc, model = load_model("openai/whisper-large-v3")
    n_dec = replace_linear4bit(model, "decoder", torch.float16, "fp4")
    n_enc = replace_enc_ffn_log_int4(model)
    model.to(DEVICE)
    torch.cuda.synchronize()
    load = mib()
    storage = module_bytes(model) / 1024**2
    print("  dec", n_dec, "enc_ffn", n_enc, "load", round(load, 1), "storage", round(storage, 1))
    try:
        dec = time_whisper_decode(model, proc)
    except Exception as e:
        dec = {"error": str(e)}
    print("  decode", dec)
    rows = sweep(lambda bs: time_whisper(model, proc, bs), batches)
    out.append(
        pack_result(
            "whisper_proposed_log_encffn_bnb_dec",
            "proposed",
            load,
            rows,
            {"storage_mib": round(storage, 1), "replaced": {"decoder_bnb": n_dec, "enc_ffn_log": n_enc}},
            dec,
        )
    )
    del model
    reset()
    return {"whisper": out}


def phase_whisper_packfp16() -> dict:
    """Paper-oriented Whisper: log-W4 packed encoder FFN + FP16 GEMM, W4 elsewhere."""
    prev = []
    if OUT.exists():
        try:
            prev = [
                m
                for m in json.loads(OUT.read_text()).get("whisper", [])
                if m.get("method") != "whisper_proposed_log_packfp16_w4rest"
            ]
        except json.JSONDecodeError:
            prev = []
    for m in prev:
        rows = [r for r in (m.get("sweep") or []) if "utt_s" in r]
        if rows:
            m["iso"] = iso_from_sweep(rows)
            m["fits"] = {str(b): (m.get("load_mib") or 1e9) <= b for b in BUDGETS_MIB}

    print("== whisper proposed pack-fp16 encoder FFN + W4 rest ==")
    batches = (1, 2, 4, 8, 12, 16, 24, 32, 48)
    reset()
    proc, model = load_model("openai/whisper-large-v3")
    n_ffn = replace_enc_ffn_pack_fp16(model)
    n_w4 = replace_linear4bit(model, "all", torch.float16, "fp4")
    model.to(DEVICE)
    torch.cuda.synchronize()
    load = mib()
    storage = module_bytes(model) / 1024**2
    print("  enc_ffn_packfp16", n_ffn, "w4_rest", n_w4, "load", round(load, 1), "storage", round(storage, 1))
    try:
        dec = time_whisper_decode(model, proc)
    except Exception as e:
        dec = {"error": str(e)}
    print("  decode", dec)
    rows = sweep(lambda bs: time_whisper(model, proc, bs), batches)
    prev.append(
        pack_result(
            "whisper_proposed_log_packfp16_w4rest",
            "proposed",
            load,
            rows,
            {"storage_mib": round(storage, 1), "replaced": {"enc_ffn_packfp16": n_ffn, "w4_rest": n_w4}},
            dec,
        )
    )
    del model
    reset()
    return {"whisper": prev}


def qwen3_load_bf16():
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    mid = "Qwen/Qwen3-ASR-1.7B-hf"
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForMultimodalLM.from_pretrained(mid, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.to(DEVICE).eval()
    return proc, model


def phase_qwen3() -> dict:
    print("== Qwen3 e2e ==")
    out = []
    batches = (1, 2, 4, 8, 12, 16, 24)

    print("-- qwen3_bf16 --")
    reset()
    proc, model = qwen3_load_bf16()
    load = mib()
    storage = module_bytes(model) / 1024**2
    dec = time_qwen3_decode(model, proc)
    print("  decode", dec)
    rows = sweep(lambda bs: time_qwen3(model, proc, bs), batches)
    out.append(pack_result("qwen3_bf16", "existing", load, rows, {"storage_mib": round(storage, 1)}, dec))
    del model
    reset()
    save({"qwen3": out})

    print("-- qwen3_bnb_w4_llm (language_model only) --")
    proc, model = qwen3_load_bf16()
    n = replace_linear4bit(model, "decoder", torch.bfloat16, "fp4")
    model.to(DEVICE)
    torch.cuda.synchronize()
    load = mib()
    storage = module_bytes(model) / 1024**2
    print("  replaced", n, "load", round(load, 1))
    try:
        dec = time_qwen3_decode(model, proc)
    except Exception as e:
        dec = {"error": str(e)}
    print("  decode", dec)
    rows = sweep(lambda bs: time_qwen3(model, proc, bs), batches)
    out.append(
        pack_result(
            "qwen3_bnb_w4_llm",
            "existing_4bit_runtime",
            load,
            rows,
            {"storage_mib": round(storage, 1), "replaced": n},
            dec,
        )
    )
    del model
    reset()
    save({"qwen3": out})

    print("-- qwen3_proposed bnb W4 LLM + log-int4 encoder FFN --")
    proc, model = qwen3_load_bf16()
    n_llm = replace_linear4bit(model, "decoder", torch.bfloat16, "fp4")
    n_enc = replace_enc_ffn_log_int4(model)
    n_enc_rest = replace_linear4bit(model, "encoder", torch.bfloat16, "fp4")
    model.to(DEVICE)
    torch.cuda.synchronize()
    load = mib()
    storage = module_bytes(model) / 1024**2
    print("  llm", n_llm, "enc_ffn", n_enc, "enc_rest", n_enc_rest, "load", round(load, 1))
    try:
        dec = time_qwen3_decode(model, proc)
    except Exception as e:
        dec = {"error": str(e)}
    print("  decode", dec)
    rows = sweep(lambda bs: time_qwen3(model, proc, bs), batches)
    out.append(
        pack_result(
            "qwen3_proposed_bnb_llm_log_encffn",
            "proposed",
            load,
            rows,
            {"storage_mib": round(storage, 1), "replaced": {"llm_bnb": n_llm, "enc_ffn_log": n_enc, "enc_rest_bnb": n_enc_rest}},
            dec,
        )
    )
    del model
    reset()
    return {"qwen3": out}


def phase_qwen3_proposed_only() -> dict:
    """Re-run proposed only; keep existing BF16 / bnb-LLM rows."""
    prev = []
    if OUT.exists():
        try:
            prev = [m for m in json.loads(OUT.read_text()).get("qwen3", []) if m.get("method") != "qwen3_proposed_bnb_llm_log_encffn"]
        except json.JSONDecodeError:
            prev = []
    print("== Qwen3 proposed only ==")
    batches = (1, 2, 4, 8, 12, 16, 24)
    reset()
    proc, model = qwen3_load_bf16()
    n_llm = replace_linear4bit(model, "decoder", torch.bfloat16, "fp4")
    n_enc = replace_enc_ffn_log_int4(model)
    n_enc_rest = replace_linear4bit(model, "encoder", torch.bfloat16, "fp4")
    model.to(DEVICE)
    torch.cuda.synchronize()
    load = mib()
    storage = module_bytes(model) / 1024**2
    print("  llm", n_llm, "enc_ffn", n_enc, "enc_rest", n_enc_rest, "load", round(load, 1), "storage", round(storage, 1))
    try:
        dec = time_qwen3_decode(model, proc)
    except Exception as e:
        dec = {"error": str(e)}
    print("  decode", dec)
    rows = sweep(lambda bs: time_qwen3(model, proc, bs), batches)
    prev.append(
        pack_result(
            "qwen3_proposed_bnb_llm_log_encffn",
            "proposed",
            load,
            rows,
            {
                "storage_mib": round(storage, 1),
                "replaced": {"llm_bnb": n_llm, "enc_ffn_log": n_enc, "enc_rest_bnb": n_enc_rest},
            },
            dec,
        )
    )
    del model
    reset()
    return {"qwen3": prev}


def summarize(data: dict) -> dict:
    wins = []
    notes = []
    for key in ("whisper", "qwen3"):
        methods = data.get(key) or []
        base = next((m for m in methods if m["kind"] == "existing"), None)
        if not base:
            continue
        for m in methods:
            if m is base:
                continue
            # decode
            bd, md = base.get("decode") or {}, m.get("decode") or {}
            if "tok_s" in bd and "tok_s" in md and md["tok_s"] > bd["tok_s"]:
                wins.append(
                    {
                        "metric": "decode_tok_s",
                        "model": key,
                        "winner": m["method"],
                        "vs": base["method"],
                        "values": {"base": bd["tok_s"], "other": md["tok_s"]},
                    }
                )
            # iso utt/s per budget
            for b, brow in (m.get("iso") or {}).items():
                arow = (base.get("iso") or {}).get(b)
                if brow and arow and brow["utt_s"] > arow["utt_s"]:
                    wins.append(
                        {
                            "metric": "iso_utt_s",
                            "budget_mib": int(b),
                            "model": key,
                            "winner": m["method"],
                            "vs": base["method"],
                            "values": {
                                "base_utt_s": arow["utt_s"],
                                "other_utt_s": brow["utt_s"],
                                "base_bs": arow["batch"],
                                "other_bs": brow["batch"],
                            },
                        }
                    )
            # max batch
            if m.get("max_batch") and base.get("max_batch") and m["max_batch"] > base["max_batch"]:
                wins.append(
                    {
                        "metric": "max_batch",
                        "model": key,
                        "winner": m["method"],
                        "vs": base["method"],
                        "values": {"base": base["max_batch"], "other": m["max_batch"]},
                    }
                )
            if m.get("load_mib") and base.get("load_mib") and m["load_mib"] < base["load_mib"] * 0.9:
                notes.append(
                    {
                        "metric": "load_mib",
                        "model": key,
                        "method": m["method"],
                        "base": base["load_mib"],
                        "other": m["load_mib"],
                    }
                )
    h2d = data.get("h2d") or {}
    if "whisper_fp16" in h2d and "whisper_gptq" in h2d:
        a, b = h2d["whisper_fp16"], h2d["whisper_gptq"]
        if a.get("weight_mib") and b.get("weight_mib") and b["weight_mib"] < a["weight_mib"]:
            wins.append(
                {
                    "metric": "h2d_bytes",
                    "model": "whisper",
                    "winner": "whisper_gptq",
                    "vs": "whisper_fp16",
                    "values": {"base_mib": a["weight_mib"], "other_mib": b["weight_mib"]},
                }
            )
    if "qwen3_bf16" in h2d and "qwen3_gptq_llm" in h2d:
        a, b = h2d["qwen3_bf16"], h2d["qwen3_gptq_llm"]
        if a.get("weight_mib") and b.get("weight_mib") and b["weight_mib"] < a["weight_mib"]:
            wins.append(
                {
                    "metric": "h2d_bytes",
                    "model": "qwen3",
                    "winner": "qwen3_gptq_llm",
                    "vs": "qwen3_bf16",
                    "values": {"base_mib": a["weight_mib"], "other_mib": b["weight_mib"]},
                }
            )
    gemm = data.get("gemm") or []
    for r in gemm:
        for k in ("bnb_speedup", "tinygemm_speedup", "triton_speedup"):
            if r.get(k) and r[k] > 1.0:
                wins.append({"metric": "gemm", "speedup_key": k, **{x: r[x] for x in ("tag", "m", "dtype", k, "fp_ms")}})
    return {"wins": wins, "notes": notes}


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase",
        default="all",
        choices=[
            "gemm",
            "h2d",
            "whisper",
            "whisper_packfp16",
            "qwen3",
            "qwen3_proposed",
            "all",
            "summarize",
        ],
    )
    args = p.parse_args()
    data = {}
    if OUT.exists():
        try:
            data = json.loads(OUT.read_text())
        except json.JSONDecodeError:
            data = {}
    if args.phase in ("gemm", "all"):
        data.update(phase_gemm())
        save(data)
    if args.phase in ("h2d", "all"):
        data.update(phase_h2d())
        save(data)
    if args.phase in ("whisper", "all"):
        data.update(phase_whisper())
        save(data)
    if args.phase == "whisper_packfp16":
        data.update(phase_whisper_packfp16())
        save(data)
    if args.phase in ("qwen3", "all"):
        data.update(phase_qwen3())
        save(data)
    if args.phase == "qwen3_proposed":
        data.update(phase_qwen3_proposed_only())
        save(data)
    data["summary"] = summarize(data)
    save(data)
    print("SUMMARY", json.dumps(data["summary"], indent=2)[:4000])


if __name__ == "__main__":
    main()
