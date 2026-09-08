"""Fig. 2: per-channel |act| max — LLM outlier channels vs speech FFN.

SmoothQuant-style sorted channel plot. Grouping helps when a few channels
set the tensor absmax; speech encoder FFN is not that structure.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
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
KEEP = ("enc_ffn", "llm_ffn")
N_GRID = 128


def hook_channel_amax(model: nn.Module):
    buf: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    hooks = []

    def make(name: str, g: str):
        def hook(mod, inp, out):
            x = inp[0]
            if not torch.is_tensor(x) or x.ndim < 2:
                return
            amax = x.detach().float().abs().amax(dim=tuple(range(x.ndim - 1))).cpu()
            prev = buf[g].get(name)
            buf[g][name] = amax if prev is None else torch.maximum(prev, amax)

        return hook

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        g = layer_group(name)
        if g in KEEP:
            hooks.append(mod.register_forward_hook(make(name, g)))
    return buf, hooks


def curves_from_buf(buf: dict) -> dict:
    out = {}
    grid = np.linspace(0.0, 1.0, N_GRID)
    for g, layers in buf.items():
        ranked = []
        ratios = []
        for name, t in layers.items():
            v = t.numpy().astype(np.float64)
            v = np.sort(v)[::-1]
            if v.size < 8 or v[v.size // 2] <= 0:
                continue
            med = float(np.median(v))
            ratios.append(float(v[0] / max(med, 1e-12)))
            xs = np.linspace(0.0, 1.0, v.size)
            ranked.append(np.interp(grid, xs, v / med))
        if not ranked:
            continue
        arr = np.stack(ranked, 0)
        worst_i = int(np.argmax(ratios))
        out[g] = {
            "rank": grid.tolist(),
            "median_over_median": np.median(arr, 0).tolist(),
            "p25": np.quantile(arr, 0.25, 0).tolist(),
            "p75": np.quantile(arr, 0.75, 0).tolist(),
            "max_over_layers": np.max(arr, 0).tolist(),
            "worst_layer": ranked[worst_i].tolist(),
            "n_layers": int(arr.shape[0]),
            "outlier_ratio_median": float(np.median(ratios)),
            "outlier_ratio_max": float(np.max(ratios)),
        }
    return out


def run_whisper(n: int) -> dict:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    from run_eval import generate_from_features

    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3", torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.to(DEVICE).eval()
    buf, hooks = hook_channel_amax(model)
    items = load_split("test-other")[:n]
    with torch.inference_mode():
        for i in range(0, len(items), 4):
            chunk = items[i : i + 4]
            waves = [pad_to_30s(load_audio(x["path"])) for x in chunk]
            feats = processor(waves, sampling_rate=SR, return_tensors="pt").input_features
            generate_from_features(model, processor, feats.to(DEVICE, dtype=torch.float16), 32)
            print(f"[whisper-ch] {min(i + 4, len(items))}/{len(items)}", flush=True)
    for h in hooks:
        h.remove()
    out = {"model": "whisper-large-v3", "n_utts": n, **curves_from_buf(buf)}
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
    buf, hooks = hook_channel_amax(model)
    items = load_split("test-other")[:n]
    with torch.inference_mode():
        for i in range(0, len(items), 4):
            chunk = items[i : i + 4]
            transcribe_batch(model, processor, [load_audio(x["path"]) for x in chunk], 32)
            print(f"[qwen-ch] {min(i + 4, len(items))}/{len(items)}", flush=True)
    for h in hooks:
        h.remove()
    out = {"model": "Qwen3-ASR-1.7B", "n_utts": n, **curves_from_buf(buf)}
    del model
    torch.cuda.empty_cache()
    return out


def plot_fig2(payload: dict, path_pdf: Path, path_png: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    enc_c, llm_c = "#1f4e79", "#c45c26"
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7), sharey=True)
    panels = [
        (axes[0], "whisper", "(a) Whisper-large-v3"),
        (axes[1], "qwen3", "(b) Qwen3-ASR-1.7B"),
    ]
    for ax, key, title in panels:
        block = payload[key]
        for g, color, label in (
            ("enc_ffn", enc_c, "Encoder FFN"),
            ("llm_ffn", llm_c, "Decoder / LLM FFN"),
        ):
            h = block[g]
            x = np.array(h["rank"]) * 100.0
            y = np.array(h["median_over_median"])
            lo, hi = np.array(h["p25"]), np.array(h["p75"])
            ax.plot(x, y, color=color, lw=1.7, label=f"{label}  med {h['outlier_ratio_median']:.1f}× / worst {h['outlier_ratio_max']:.0f}×")
            ax.plot(x, np.array(h["worst_layer"]), color=color, lw=1.0, ls="--", alpha=0.85)
            ax.fill_between(x, lo, hi, color=color, alpha=0.12)
        ax.set_title(title, loc="left")
        ax.set_xlim(0, 100)
        ax.set_yscale("log")
        ax.set_xlabel("channel rank (%)")
        ax.legend(frameon=False, loc="upper right")
        ax.axhline(1.0, color="#888888", lw=0.6, ls=":")
    axes[0].set_ylabel(r"channel $\max|x|$ / median")
    from matplotlib.lines import Line2D
    fig.legend(
        handles=[
            Line2D([0], [0], color="k", lw=1.6, label="median layer"),
            Line2D([0], [0], color="k", lw=1.0, ls="--", label="worst layer"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_pdf, bbox_inches="tight")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def downsample_canvas(h: dict, stride: int = 4) -> dict:
    return {
        "rank": [round(100.0 * x, 1) for x in h["rank"][::stride]],
        "y": [round(v, 4) for v in h["median_over_median"][::stride]],
        "ratio": h["outlier_ratio_median"],
        "n_layers": h["n_layers"],
    }


def main():
    n = 8
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"n_utts": n, "split": "librispeech-test-other"}
    payload["whisper"] = run_whisper(n)
    payload["qwen3"] = run_qwen3(n)
    (OUT / "fig2_channel_outliers.json").write_text(json.dumps(payload))
    canvas = {}
    for m in ("whisper", "qwen3"):
        canvas[m] = {
            g: {
                **downsample_canvas(payload[m][g]),
                "worst": [round(v, 4) for v in payload[m][g]["worst_layer"][::4]],
                "ratio_max": payload[m][g]["outlier_ratio_max"],
            }
            for g in ("enc_ffn", "llm_ffn")
        }
    (OUT / "fig2_canvas.json").write_text(json.dumps(canvas, indent=2))
    plot_fig2(payload, FIG / "fig2_channel_outliers.pdf", FIG / "fig2_channel_outliers.png")
    print("[fig2] done", flush=True)


if __name__ == "__main__":
    main()
