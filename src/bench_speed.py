"""Find operating points where proposed INT8 GEMM beats FP16 Linear.

Shapes match Whisper encoder FFN/attn and Qwen3 AuT FFN / LLM FFN.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from log_int4 import LogInt4Linear, log_indices, log_lut_int8, pack_nibbles, quant_act_int8, alpha_from_x, _int_mm_fp16

DEVICE = torch.device("cuda")
OUT = Path("/workspace/SpeechPTQ/results/diagnostics/speed_bench.json")


def sync_time(fn, warmup=15, iters=50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def uniform_a8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    z = torch.clamp((x.float() * (127.0 / amax)).round(), -127, 127).to(torch.int8)
    scale = (amax.squeeze(-1) / 127.0).float()
    return z, scale


def bench_linear(m: int, k: int, n: int, tag: str) -> dict:
    x = torch.randn(m, k, device=DEVICE, dtype=torch.float16)
    w = torch.randn(n, k, device=DEVICE, dtype=torch.float16) * 0.02
    b = torch.randn(n, device=DEVICE, dtype=torch.float16)
    linear = torch.nn.Linear(k, n, bias=True).to(DEVICE, dtype=torch.float16)
    with torch.no_grad():
        linear.weight.copy_(w)
        linear.bias.copy_(b)

    fp = sync_time(lambda: linear(x))

    # fake log-W4 A8: dequant then fp16 gemm
    absmax = w.abs().amax().clamp_min(1e-8)
    alpha_w = absmax / 127.0
    y = log_indices(w, alpha_w)
    lut = log_lut_int8(DEVICE)
    z_w = lut[(y.to(torch.int16) + 7).long()].float() * alpha_w
    w_fake = z_w.to(torch.float16)

    def fake():
        amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        xq = (x / amax * 127).round().clamp(-127, 127) * (amax / 127)
        return F.linear(xq, w_fake, b)

    fake_ms = sync_time(fake)

    # proposed: log W4 (as int8 LUT) + uniform A8 + int_mm
    z_w_i8 = lut[(y.to(torch.int16) + 7).long()]
    w_kn = z_w_i8.T.contiguous()
    alpha_w_s = (absmax / 127.0).reshape(()).float()

    def w4a8_int():
        z_a, sa = uniform_a8(x)
        return _int_mm_fp16(z_a, w_kn, sa, alpha_w_s, b, m)

    int_ms = sync_time(w4a8_int)

    # log-A4 int8 path (LogInt4Linear)
    log_lin = LogInt4Linear(linear, tag, "log_token", 8.0).to(DEVICE)

    def log_a4():
        return log_lin(x)

    try:
        log_ms = sync_time(log_a4)
    except Exception as e:
        log_ms = float("nan")
        print(f"  log_a4 failed: {e}")

    out = {
        "tag": tag,
        "m": m,
        "k": k,
        "n": n,
        "fp16_ms": round(fp, 4),
        "fake_w4a8_ms": round(fake_ms, 4),
        "int8_w4a8_ms": round(int_ms, 4),
        "int8_log_a4_ms": None if log_ms != log_ms else round(log_ms, 4),
        "int8_w4a8_speedup": round(fp / int_ms, 3),
        "log_a4_speedup": None if log_ms != log_ms else round(fp / log_ms, 3),
        "fake_speedup": round(fp / fake_ms, 3),
    }
    print(json.dumps(out))
    return out


def bench_whisper_encoder() -> dict:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from kv_template import pad_to_30s
    from data import SR, load_audio, load_split
    from quant import replace_linears

    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    items = load_split("test-clean")[:16]
    waves = [pad_to_30s(load_audio(x["path"])) for x in items]
    feats = processor(waves, sampling_rate=SR, return_tensors="pt").input_features.to(
        DEVICE, dtype=torch.float16
    )

    def load():
        m = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3", torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(DEVICE).eval()
        return m

    model = load()

    def enc():
        return model.model.encoder(feats).last_hidden_state

    fp = sync_time(enc, warmup=5, iters=20)
    del model
    torch.cuda.empty_cache()

    model = load()
    n = replace_linears(model, {}, "log", "log_token", "enc_ffn", backend="int4")
    print(f"[encoder] replaced FFN int4 n={n}")

    def enc_ffn():
        return model.model.encoder(feats).last_hidden_state

    ffn = sync_time(enc_ffn, warmup=5, iters=20)
    del model
    torch.cuda.empty_cache()

    model = load()
    n2 = replace_linears(model, {}, "log", "log_token", "encoder", backend="int4")
    print(f"[encoder] replaced encoder int4 n={n2}")

    def enc_all():
        return model.model.encoder(feats).last_hidden_state

    all_ms = sync_time(enc_all, warmup=5, iters=20)
    del model
    torch.cuda.empty_cache()

    # fake enc ffn
    model = load()
    replace_linears(model, {}, "log", "log_token", "enc_ffn", backend="fake")

    def enc_fake():
        return model.model.encoder(feats).last_hidden_state

    fake = sync_time(enc_fake, warmup=5, iters=20)
    del model
    torch.cuda.empty_cache()

    out = {
        "batch": 16,
        "frames": 1500,
        "fp16_ms": round(fp, 3),
        "proposed_ffn_int4_ms": round(ffn, 3),
        "proposed_ffn_int4_speedup": round(fp / ffn, 3),
        "proposed_encoder_int4_ms": round(all_ms, 3),
        "proposed_encoder_int4_speedup": round(fp / all_ms, 3),
        "fake_ffn_ms": round(fake, 3),
        "fake_ffn_speedup": round(fp / fake, 3),
    }
    print("[whisper encoder]", json.dumps(out, indent=2))
    return out


def bench_qwen3_aut() -> dict:
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    from data import load_audio, load_split
    from quant import replace_linears
    from run_eval_qwen3 import compressed_tensors_load_kwargs

    model_id = "Qwen/Qwen3-ASR-1.7B-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    items = load_split("test-clean")[:8]
    waves = [load_audio(x["path"]) for x in items]
    packed = processor.apply_transcription_request(audio=waves, language="English")
    feats = packed["input_features"].to(DEVICE, dtype=torch.bfloat16)
    mask = packed["input_features_mask"].to(DEVICE)

    def load():
        kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
        kw.update(compressed_tensors_load_kwargs(model_id))
        m = AutoModelForMultimodalLM.from_pretrained(model_id, **kw).to(DEVICE).eval()
        return m

    model = load()
    tower = model.model.audio_tower

    def aut():
        return tower(feats, mask)

    try:
        fp = sync_time(aut, warmup=4, iters=12)
    except Exception as e:
        print(f"aut fp failed: {e}")
        del model
        return {"error": str(e)}

    del model
    torch.cuda.empty_cache()

    model = load()
    n = replace_linears(model, {}, "log", "log_token", "enc_ffn", backend="int4")
    print(f"[qwen3 aut] replaced FFN int4 n={n}")
    tower = model.model.audio_tower

    def aut_q():
        return tower(feats, mask)

    q = sync_time(aut_q, warmup=4, iters=12)
    del model
    torch.cuda.empty_cache()

    out = {
        "batch": 8,
        "fp16_ms": round(fp, 3),
        "proposed_ffn_int4_ms": round(q, 3),
        "proposed_ffn_int4_speedup": round(fp / q, 3),
    }
    print("[qwen3 aut]", json.dumps(out, indent=2))
    return out


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    rows = []
    # Whisper encoder FFN fc1: 1280 -> 5120, M = B * 1500
    for b in (1, 4, 8, 16, 32):
        rows.append(bench_linear(1500 * b, 1280, 5120, f"whisper_enc_fc1_B{b}"))
    # Whisper encoder attn q: 1280 -> 1280
    for b in (1, 8, 16):
        rows.append(bench_linear(1500 * b, 1280, 1280, f"whisper_enc_q_B{b}"))
    # Qwen3 AuT fc1: 1024 -> 4096, ~12.5 Hz, 8s -> 100 tokens, 30s -> 375
    for m, lab in ((100, "8s"), (375, "30s"), (375 * 4, "30s_B4"), (375 * 8, "30s_B8")):
        rows.append(bench_linear(m, 1024, 4096, f"qwen3_aut_fc1_{lab}"))
    # Qwen3 LLM down_proj 6144 -> 2048, prefill vs decode
    for m, lab in ((1, "decode"), (64, "prefill64"), (256, "prefill256"), (2048, "prefill2k")):
        rows.append(bench_linear(m, 6144, 2048, f"qwen3_llm_down_{lab}"))

    print("\n===== module forwards =====")
    whisper_enc = bench_whisper_encoder()
    qwen_aut = bench_qwen3_aut()
    payload = {
        "gpu": torch.cuda.get_device_name(0),
        "layers": rows,
        "whisper_encoder_bs16": whisper_enc,
        "qwen3_aut_bs8": qwen_aut,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
