"""Iso-memory throughput: max batch and utt/s under a VRAM budget.

Existing: FP16/BF16, GPTQ W4A16 (decompress vs packed).
Proposed: log INT4 encoder-FFN kernel (Whisper) / GPTQ+log encoder (Qwen3).
"""

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
from run_eval import _compressed_tensors_load_kwargs, generate_from_features, load_model
from run_eval_qwen3 import compressed_tensors_load_kwargs

DEVICE = torch.device("cuda")
OUT = Path("/workspace/SpeechPTQ/results/diagnostics/iso_mem_throughput.json")
BUDGETS_MIB = [6144, 8192, 10240, 12288, 16384]


def mib() -> float:
    return torch.cuda.memory_allocated() / (1024**2)


def peak_mib() -> float:
    return torch.cuda.max_memory_allocated() / (1024**2)


def reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def dummy_waves(n: int, sec: float = 8.0) -> list[np.ndarray]:
    t = int(sec * SR)
    rng = np.random.default_rng(0)
    return [rng.standard_normal(t, dtype=np.float32) * 0.01 for _ in range(n)]


@torch.inference_mode()
def time_whisper(model, processor, bs: int, n_iters: int = 4) -> dict:
    waves = [pad_to_30s(w) for w in dummy_waves(bs, 8.0)]
    feats = processor(waves, sampling_rate=SR, return_tensors="pt").input_features
    feats = feats.to(DEVICE, dtype=torch.float16)
    def run():
        generate_from_features(model, processor, feats, 24)
    for _ in range(2):
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
def time_qwen3(model, processor, bs: int, n_iters: int = 3) -> dict:
    waves = dummy_waves(bs, 8.0)
    def run():
        inputs = processor.apply_transcription_request(audio=waves, language="English")
        inputs = inputs.to(DEVICE, model.dtype)
        model.generate(**inputs, max_new_tokens=24, do_sample=False, num_beams=1, use_cache=True)
    for _ in range(1):
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


def sweep(fn, batches=(1, 2, 4, 8, 12, 16, 24, 32)) -> list[dict]:
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
            break
        except Exception as e:
            print(f"  bs={bs} fail {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
            break
    return rows


def iso_from_sweep(rows: list[dict]) -> dict:
    out = {}
    for b in BUDGETS_MIB:
        cand = [r for r in rows if r["peak_mib"] <= b]
        if not cand:
            out[str(b)] = None
        else:
            best = max(cand, key=lambda r: r["utt_s"])
            out[str(b)] = best
    return out


def whisper_fp16():
    print("== whisper fp16 ==")
    reset()
    proc, model = load_model("openai/whisper-large-v3")
    load = mib()
    rows = sweep(lambda bs: time_whisper(model, proc, bs))
    del model
    return {"method": "whisper_fp16", "kind": "existing", "load_mib": round(load, 1), "sweep": rows, "iso": iso_from_sweep(rows)}


def whisper_gptq(compressed: bool):
    tag = "whisper_gptq_packed" if compressed else "whisper_gptq_decomp"
    print(f"== {tag} ==")
    reset()
    mid = "/workspace/SpeechPTQ/models/whisper-large-v3-W4A16-G128"
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, AutoConfig
    from transformers.utils.quantization_config import CompressedTensorsConfig
    proc = WhisperProcessor.from_pretrained(mid)
    cfg = AutoConfig.from_pretrained(mid)
    qc = cfg.quantization_config
    if isinstance(qc, CompressedTensorsConfig):
        qc.run_compressed = compressed
        kw = {"quantization_config": qc, "torch_dtype": torch.float16, "low_cpu_mem_usage": True}
    else:
        kw = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
        kw.update(_compressed_tensors_load_kwargs(mid) if not compressed else {})
        if compressed:
            kw["quantization_config"] = CompressedTensorsConfig.from_dict(
                {**getattr(cfg, "quantization_config", {}), "run_compressed": True}
                if isinstance(getattr(cfg, "quantization_config", None), dict)
                else {"run_compressed": True}
            )
    try:
        model = WhisperForConditionalGeneration.from_pretrained(mid, attn_implementation="sdpa", **kw)
    except Exception as e:
        print(" load fail", e)
        return {"method": tag, "kind": "existing", "error": str(e)}
    model.to(DEVICE).eval()
    load = mib()
    rows = sweep(lambda bs: time_whisper(model, proc, bs))
    del model
    return {"method": tag, "kind": "existing", "load_mib": round(load, 1), "sweep": rows, "iso": iso_from_sweep(rows)}


def whisper_proposed_int4_ffn():
    print("== whisper proposed log-W4 INT4 encoder FFN ==")
    reset()
    proc, model = load_model("openai/whisper-large-v3")
    n = replace_linears(model, {}, "log", "log_token", "enc_ffn", backend="int4")
    print(" replaced", n)
    model.to(DEVICE).eval()
    load = mib()
    rows = sweep(lambda bs: time_whisper(model, proc, bs))
    del model
    return {
        "method": "whisper_proposed_log_w4a4_enc_ffn_int4",
        "kind": "proposed",
        "replaced": n,
        "load_mib": round(load, 1),
        "sweep": rows,
        "iso": iso_from_sweep(rows),
    }


def qwen3_load(model_id: str, compressed: bool | None):
    from transformers import AutoModelForMultimodalLM, AutoProcessor, AutoConfig
    from transformers.utils.quantization_config import CompressedTensorsConfig
    proc = AutoProcessor.from_pretrained(model_id)
    kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if compressed is False:
        kw.update(compressed_tensors_load_kwargs(model_id))
    elif compressed is True:
        cfg = AutoConfig.from_pretrained(model_id)
        qc = getattr(cfg, "quantization_config", None)
        if isinstance(qc, CompressedTensorsConfig):
            qc.run_compressed = True
            kw["quantization_config"] = qc
        elif isinstance(qc, dict):
            kw["quantization_config"] = CompressedTensorsConfig.from_dict({**qc, "run_compressed": True})
    model = AutoModelForMultimodalLM.from_pretrained(model_id, **kw)
    model.to(DEVICE).eval()
    return proc, model


def qwen3_bf16():
    print("== qwen3 bf16 ==")
    reset()
    proc, model = qwen3_load("Qwen/Qwen3-ASR-1.7B-hf", None)
    load = mib()
    rows = sweep(lambda bs: time_qwen3(model, proc, bs), batches=(1, 2, 4, 8, 12, 16))
    del model
    return {"method": "qwen3_bf16", "kind": "existing", "load_mib": round(load, 1), "sweep": rows, "iso": iso_from_sweep(rows)}


def qwen3_gptq(compressed: bool):
    tag = "qwen3_gptq_packed" if compressed else "qwen3_gptq_decomp"
    print(f"== {tag} ==")
    reset()
    mid = "/workspace/SpeechPTQ/models/qwen3-asr-1.7b-W4A16-G128-llm"
    try:
        proc, model = qwen3_load(mid, compressed)
    except Exception as e:
        print(" load fail", e)
        return {"method": tag, "kind": "existing", "error": str(e)}
    load = mib()
    rows = sweep(lambda bs: time_qwen3(model, proc, bs), batches=(1, 2, 4, 8, 12, 16))
    del model
    return {"method": tag, "kind": "existing", "load_mib": round(load, 1), "sweep": rows, "iso": iso_from_sweep(rows)}


def qwen3_proposed():
    print("== qwen3 proposed GPTQ-decomp + log encoder W4A8 fake ==")
    reset()
    mid = "/workspace/SpeechPTQ/models/qwen3-asr-1.7b-W4A16-G128-llm"
    proc, model = qwen3_load(mid, False)
    n = replace_linears(model, {}, "log", "uniform_a8_dyn", "encoder", backend="fake")
    print(" replaced", n)
    model.to(DEVICE).eval()
    load = mib()
    rows = sweep(lambda bs: time_qwen3(model, proc, bs), batches=(1, 2, 4, 8, 12, 16))
    del model
    return {
        "method": "qwen3_proposed_gptq_llm_log_enc_w4a8",
        "kind": "proposed",
        "replaced": n,
        "load_mib": round(load, 1),
        "sweep": rows,
        "iso": iso_from_sweep(rows),
    }


def main():
    results = []
    for fn in (
        whisper_fp16,
        lambda: whisper_gptq(False),
        lambda: whisper_gptq(True),
        whisper_proposed_int4_ffn,
        qwen3_bf16,
        lambda: qwen3_gptq(False),
        lambda: qwen3_gptq(True),
        qwen3_proposed,
    ):
        try:
            results.append(fn())
        except Exception as e:
            print("METHOD FAIL", fn, e)
            results.append({"error": str(e), "fn": getattr(fn, "__name__", str(fn))})
        reset()
    OUT.write_text(json.dumps(results, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
