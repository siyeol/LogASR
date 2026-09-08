"""Fig. 1: encoder FFN vs decoder/LLM FFN |x| histograms (Whisper + Qwen3).

Collects a 32-utt probe (same protocol as analyze_act_stats.py), writes
publication PDF/PNG plus bin JSON for the canvas. Not an OOD WER run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from data import SR, load_audio, load_split
from kv_template import pad_to_30s
from layer_groups import layer_group

DEVICE = torch.device("cuda")
OUT = Path("/workspace/SpeechPTQ/results/diagnostics")
FIG = Path("/workspace/SpeechPTQ/results/figures")
RESERVOIR = 250_000
KEEP_GROUPS = ("enc_ffn", "llm_ffn")
N_BINS = 48
LOG_LO, LOG_HI = -4.0, 0.0  # log10(|x| / absmax)


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


def summarize(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"n": 0}
    a = np.abs(arr)
    return {
        "n": int(arr.size),
        "absmax": float(a.max()),
        "mean_abs": float(a.mean()),
        "p50_abs": float(np.quantile(a, 0.50)),
        "p99_abs": float(np.quantile(a, 0.99)),
        "p999_abs": float(np.quantile(a, 0.999)),
        "excess_kurtosis": excess_kurtosis(arr),
        "log2_dynamic_range": float(
            np.log2(max(a.max() / max(np.quantile(a, 0.50), 1e-12), 1.0))
        ),
        "frac_below_uniform_halfbin": float((a < (a.max() / 14.0)).mean()),
        "frac_below_log_alpha": float((a < (a.max() / 127.0)).mean()),
    }


def hist_normalized(arr: np.ndarray) -> dict:
    a = np.abs(arr).astype(np.float64)
    amax = max(float(a.max()), 1e-12)
    u = np.clip(a / amax, 1e-12, 1.0)
    logu = np.log10(u)
    edges = np.linspace(LOG_LO, LOG_HI, N_BINS + 1)
    dens, _ = np.histogram(logu, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return {
        "log10_centers": centers.tolist(),
        "density": dens.tolist(),
        "absmax": amax,
        **summarize(arr),
    }


def weight_reservoirs(model: nn.Module) -> dict[str, Reservoir]:
    groups: dict[str, Reservoir] = {g: Reservoir() for g in KEEP_GROUPS}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        g = layer_group(name)
        if g not in groups:
            continue
        w = mod.weight.detach().float().reshape(-1)
        if w.numel() > 65536:
            w = w[torch.randint(0, w.numel(), (65536,), device=w.device)]
        groups[g].add(w)
    return groups


def hook_ffn(model: nn.Module) -> tuple[dict[str, Reservoir], list]:
    groups: dict[str, Reservoir] = {g: Reservoir() for g in KEEP_GROUPS}
    hooks = []

    def make(g: str):
        def hook(mod, inp, out):
            x = inp[0]
            if torch.is_tensor(x):
                groups[g].add(x)

        return hook

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        g = layer_group(name)
        if g in groups:
            hooks.append(mod.register_forward_hook(make(g)))
    return groups, hooks


def pack(groups: dict[str, Reservoir]) -> dict:
    return {
        g: hist_normalized(r.buf if r.buf is not None else np.array([0.0]))
        for g, r in groups.items()
    }


def run_whisper(n: int) -> dict:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from run_eval import generate_from_features

    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3", torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.to(DEVICE).eval()
    w_groups = weight_reservoirs(model)
    a_groups, hooks = hook_ffn(model)
    items = load_split("test-other")[:n]
    with torch.inference_mode():
        for i in range(0, len(items), 4):
            chunk = items[i : i + 4]
            waves = [pad_to_30s(load_audio(x["path"])) for x in chunk]
            feats = processor(waves, sampling_rate=SR, return_tensors="pt").input_features
            feats = feats.to(DEVICE, dtype=torch.float16)
            generate_from_features(model, processor, feats, 64)
            print(f"[whisper] {min(i + 4, len(items))}/{len(items)}", flush=True)
    for h in hooks:
        h.remove()
    out = {
        "model": "whisper-large-v3",
        "n_utts": n,
        "activations": pack(a_groups),
        "weights": pack(w_groups),
    }
    del model
    torch.cuda.empty_cache()
    return out


def run_qwen3(n: int) -> dict:
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    from run_eval_qwen3 import compressed_tensors_load_kwargs, transcribe_batch

    model_id = "Qwen/Qwen3-ASR-1.7B-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
    kw.update(compressed_tensors_load_kwargs(model_id))
    model = AutoModelForMultimodalLM.from_pretrained(model_id, **kw)
    model.to(DEVICE).eval()
    w_groups = weight_reservoirs(model)
    a_groups, hooks = hook_ffn(model)
    items = load_split("test-other")[:n]
    with torch.inference_mode():
        for i in range(0, len(items), 4):
            chunk = items[i : i + 4]
            waves = [load_audio(x["path"]) for x in chunk]
            transcribe_batch(model, processor, waves, 64)
            print(f"[qwen3] {min(i + 4, len(items))}/{len(items)}", flush=True)
    for h in hooks:
        h.remove()
    out = {
        "model": "Qwen3-ASR-1.7B",
        "n_utts": n,
        "activations": pack(a_groups),
        "weights": pack(w_groups),
    }
    del model
    torch.cuda.empty_cache()
    return out


def plot_fig1(payload: dict, path_pdf: Path, path_png: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    enc_c, llm_c = "#1f4e79", "#c45c26"
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.6), sharex=True, sharey=False)
    panels = [
        (axes[0, 0], "whisper", "activations", "(a) Whisper-large-v3 activations"),
        (axes[0, 1], "qwen3", "activations", "(b) Qwen3-ASR-1.7B activations"),
        (axes[1, 0], "whisper", "weights", "(c) Whisper-large-v3 weights"),
        (axes[1, 1], "qwen3", "weights", "(d) Qwen3-ASR-1.7B weights"),
    ]
    x_uni = np.log10(1.0 / 7.0)
    x_log = np.log10(1.0 / 127.0)
    for ax, model, kind, title in panels:
        block = payload[model][kind]
        for g, color, label in (
            ("enc_ffn", enc_c, "Encoder FFN"),
            ("llm_ffn", llm_c, "Decoder / LLM FFN"),
        ):
            h = block[g]
            k = h["excess_kurtosis"]
            ktxt = f"{k:.1e}" if k >= 100 else f"{k:.1f}"
            ax.plot(
                h["log10_centers"],
                h["density"],
                color=color,
                lw=1.6,
                label=f"{label}  (κ={ktxt})",
            )
            ax.fill_between(h["log10_centers"], h["density"], color=color, alpha=0.12)
        ax.axvline(x_uni, color="#555555", ls="--", lw=0.8, zorder=0)
        ax.axvline(x_log, color="#555555", ls=":", lw=0.8, zorder=0)
        ax.set_title(title, loc="left", pad=4)
        ax.set_xlim(LOG_LO, LOG_HI)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, loc="upper left")
    axes[1, 0].set_xlabel(r"$\log_{10}(|x|/A)$")
    axes[1, 1].set_xlabel(r"$\log_{10}(|x|/A)$")
    axes[0, 0].set_ylabel("density")
    axes[1, 0].set_ylabel("density")
    handles = [
        Line2D([0], [0], color="#555555", ls="--", lw=0.8, label=r"uniform first bin  $A/7$"),
        Line2D([0], [0], color="#555555", ls=":", lw=0.8, label=r"log $\alpha=A/127$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def canvas_series(payload: dict) -> dict:
    """Downsample bins for the canvas LineChart (≤24 categories)."""
    stride = 2
    out = {}
    for model in ("whisper", "qwen3"):
        out[model] = {}
        for kind in ("activations", "weights"):
            enc = payload[model][kind]["enc_ffn"]
            llm = payload[model][kind]["llm_ffn"]
            cents = enc["log10_centers"][::stride]
            out[model][kind] = {
                "categories": [f"{c:.1f}" for c in cents],
                "enc": [round(v, 4) for v in enc["density"][::stride]],
                "llm": [round(v, 4) for v in llm["density"][::stride]],
                "enc_kurt": enc["excess_kurtosis"],
                "llm_kurt": llm["excess_kurtosis"],
                "enc_p50_over_a": enc["p50_abs"] / enc["absmax"],
                "llm_p50_over_a": llm["p50_abs"] / llm["absmax"],
                "enc_p99_over_a": enc["p99_abs"] / enc["absmax"],
                "llm_p99_over_a": llm["p99_abs"] / llm["absmax"],
                "enc_frac_uni": enc["frac_below_uniform_halfbin"],
                "llm_frac_uni": llm["frac_below_uniform_halfbin"],
                "enc_log2dr": enc["log2_dynamic_range"],
                "llm_log2dr": llm["log2_dynamic_range"],
            }
    return out


def main():
    n = 32
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    payload = {"n_utts": n, "split": "librispeech-test-other", "note": "FFN Linear inputs; |x|/absmax"}
    payload["whisper"] = run_whisper(n)
    payload["qwen3"] = run_qwen3(n)
    raw_path = OUT / "fig1_histograms.json"
    raw_path.write_text(json.dumps(payload))
    canvas = canvas_series(payload)
    (OUT / "fig1_canvas.json").write_text(json.dumps(canvas, indent=2))
    plot_fig1(payload, FIG / "fig1_enc_ffn_vs_llm_ffn.pdf", FIG / "fig1_enc_ffn_vs_llm_ffn.png")
    print(f"wrote {raw_path}", flush=True)
    print(f"wrote {FIG / 'fig1_enc_ffn_vs_llm_ffn.pdf'}", flush=True)
    print("[fig1] done", flush=True)


if __name__ == "__main__":
    main()
