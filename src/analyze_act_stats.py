"""Collect weight + activation histograms / kurtosis by module role."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from data import SR, load_audio, load_split
from kv_template import pad_to_30s
from layer_groups import layer_group
from quant import is_language_decoder, is_speech_encoder  # noqa: F401  # grouping via layer_groups

DEVICE = torch.device("cuda")
OUT = Path("/workspace/SpeechPTQ/results/diagnostics")
RESERVOIR = 250_000


def excess_kurtosis(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    if x.size < 8:
        return float("nan")
    x = x - x.mean()
    m2 = np.mean(x * x)
    if m2 < 1e-18:
        return 0.0
    m4 = np.mean(x * x * x * x)
    return float(m4 / (m2 * m2) - 3.0)


def summarize(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"n": 0}
    a = np.abs(arr)
    return {
        "n": int(arr.size),
        "absmax": float(a.max()),
        "mean_abs": float(a.mean()),
        "p50_abs": float(np.quantile(a, 0.50)),
        "p90_abs": float(np.quantile(a, 0.90)),
        "p99_abs": float(np.quantile(a, 0.99)),
        "p999_abs": float(np.quantile(a, 0.999)),
        "std": float(arr.std()),
        "excess_kurtosis": excess_kurtosis(arr),
        "log2_dynamic_range": float(np.log2(max(a.max() / max(np.quantile(a, 0.50), 1e-12), 1.0))),
    }


class Reservoir:
    def __init__(self, k: int = RESERVOIR):
        self.k = k
        self.buf = None
        self.n = 0

    def add(self, x: torch.Tensor):
        v = x.detach().float().reshape(-1)
        if v.numel() == 0:
            return
        if v.numel() > 4096:
            v = v[torch.randint(0, v.numel(), (4096,), device=v.device)]
        v = v.cpu().numpy()
        self.n += int(v.size)
        if self.buf is None:
            self.buf = v[: self.k].copy()
            return
        cat = np.concatenate([self.buf, v])
        if cat.size > self.k:
            self.buf = cat[np.random.choice(cat.size, self.k, replace=False)]
        else:
            self.buf = cat


def weight_stats(model: nn.Module) -> tuple[dict, dict]:
    groups: dict[str, Reservoir] = defaultdict(Reservoir)
    per_layer = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        w = mod.weight.detach().float().cpu().numpy().reshape(-1)
        g = layer_group(name)
        groups[g].add(torch.from_numpy(w[: min(len(w), 65536)]))
        per_layer[name] = {"group": g, **summarize(w[:: max(1, w.size // 20000)])}
    out = {g: summarize(r.buf if r.buf is not None else np.array([])) for g, r in groups.items()}
    return out, per_layer


def hook_activations(model: nn.Module) -> tuple[dict, list]:
    groups: dict[str, Reservoir] = defaultdict(Reservoir)
    hooks = []

    def make(name: str):
        g = layer_group(name)

        def hook(mod, inp, out):
            x = inp[0]
            if torch.is_tensor(x):
                groups[g].add(x)

        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(make(name)))
    return groups, hooks


def run_whisper(n: int):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from run_eval import generate_from_features

    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3", torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.to(DEVICE).eval()
    w_group, w_layer = weight_stats(model)
    groups, hooks = hook_activations(model)
    items = load_split("test-other")[:n]
    with torch.inference_mode():
        for i in range(0, len(items), 4):
            chunk = items[i : i + 4]
            waves = [pad_to_30s(load_audio(x["path"])) for x in chunk]
            feats = processor(waves, sampling_rate=SR, return_tensors="pt").input_features
            feats = feats.to(DEVICE, dtype=torch.float16)
            generate_from_features(model, processor, feats, 64)
            print(f"[whisper] {min(i + 4, len(items))}/{len(items)}")
    for h in hooks:
        h.remove()
    a_group = {g: summarize(r.buf if r.buf is not None else np.array([])) for g, r in groups.items()}
    return {"weights": w_group, "activations": a_group, "n_utts": n, "weight_layers": len(w_layer)}


def run_qwen3(n: int):
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    from run_eval_qwen3 import compressed_tensors_load_kwargs, transcribe_batch

    model_id = "Qwen/Qwen3-ASR-1.7B-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
    kw.update(compressed_tensors_load_kwargs(model_id))
    model = AutoModelForMultimodalLM.from_pretrained(model_id, **kw)
    model.to(DEVICE).eval()
    w_group, w_layer = weight_stats(model)
    groups, hooks = hook_activations(model)
    items = load_split("test-other")[:n]
    with torch.inference_mode():
        for i in range(0, len(items), 4):
            chunk = items[i : i + 4]
            waves = [load_audio(x["path"]) for x in chunk]
            transcribe_batch(model, processor, waves, 64)
            print(f"[qwen3] {min(i + 4, len(items))}/{len(items)}")
    for h in hooks:
        h.remove()
    a_group = {g: summarize(r.buf if r.buf is not None else np.array([])) for g, r in groups.items()}
    return {"weights": w_group, "activations": a_group, "n_utts": n, "weight_layers": len(w_layer)}


def plot_bars(payload: dict, title: str, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["enc_ffn", "enc_attn", "enc_other", "llm_ffn", "llm_attn", "llm_other", "projector", "lm_head", "other"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, ylab in (
        (axes[0], "activations", "activation excess kurtosis"),
        (axes[1], "weights", "weight excess kurtosis"),
    ):
        gs = [g for g in order if g in payload[key] and payload[key][g].get("n", 0) > 0]
        ys = [payload[key][g]["excess_kurtosis"] for g in gs]
        ax.bar(gs, ys, color="#3b6ea5")
        ax.set_ylabel(ylab)
        ax.set_title(key)
        ax.tick_params(axis="x", rotation=35)
        ax.axhline(0, color="gray", lw=0.6)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["whisper", "qwen3", "both"], default="both")
    p.add_argument("--n", type=int, default=32)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.model in ("whisper", "both"):
        w = run_whisper(args.n)
        path = OUT / "whisper_act_weight_stats.json"
        path.write_text(json.dumps({k: v for k, v in w.items() if k != "weight_layers"}, indent=2))
        plot_bars(w, "Whisper-large-v3 (test-other sample)", OUT / "whisper_kurtosis.png")
        print(json.dumps(w["activations"], indent=2))
        print(f"wrote {path}")
        del w
        torch.cuda.empty_cache()
    if args.model in ("qwen3", "both"):
        q = run_qwen3(args.n)
        path = OUT / "qwen3_act_weight_stats.json"
        path.write_text(json.dumps({k: v for k, v in q.items() if k != "weight_layers"}, indent=2))
        plot_bars(q, "Qwen3-ASR-1.7B AuT + LLM (test-other sample)", OUT / "qwen3_kurtosis.png")
        print(json.dumps(q["activations"], indent=2))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
