"""Fig. 1: Whisper encoder FFN vs Qwen3-ASR LLM FFN (activations + weights).

Reuses the full LibriSpeech test-clean Whisper encoder histograms already
collected in fig1_whisper_testclean.json. Collects Qwen3 LLM FFN on the
same 2620 utterances (one teacher-forced prefill per batch, no generate).
"""

from __future__ import annotations

import inspect
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

from data import load_audio, load_split
from layer_groups import layer_group
from plot_fig1_whisper_full import FIG, LOG_HI, LOG_LO, N_BINS, OUT, OnlineLogHist

DEVICE = torch.device("cuda")
SPLIT = "test-clean"
BATCH = 2
WHISPER_JSON = OUT / "fig1_whisper_testclean.json"
PNG = FIG / "fig1_enc_ffn_vs_llm_ffn.png"
JSON_OUT = OUT / "fig1_enc_vs_llm.json"
QWEN_ID = "Qwen/Qwen3-ASR-1.7B-hf"


def _forward_kwargs(model, inputs: dict) -> dict:
    names = set(inspect.signature(model.forward).parameters)
    return {k: v for k, v in inputs.items() if k in names}


def hook_llm_ffn(model: nn.Module, device: torch.device):
    hist = OnlineLogHist(device)
    hooks = []

    def hook(_mod, inp, _out):
        x = inp[0]
        if torch.is_tensor(x):
            hist.add(x)

    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and layer_group(name) == "llm_ffn":
            hooks.append(mod.register_forward_hook(hook))
            n += 1
    print(f"[fig1] hooked {n} Qwen LLM FFN Linears", flush=True)
    return hist, hooks


def weight_llm_ffn(model: nn.Module, device: torch.device) -> OnlineLogHist:
    hist = OnlineLogHist(device)
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and layer_group(name) == "llm_ffn":
            hist.add(mod.weight.detach())
            n += 1
    print(f"[fig1] Qwen LLM FFN weight tensors={n}", flush=True)
    return hist


def collect_qwen(items: list[dict]) -> dict:
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    from run_eval_qwen3 import compressed_tensors_load_kwargs

    processor = AutoProcessor.from_pretrained(QWEN_ID)
    kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
    kw.update(compressed_tensors_load_kwargs(QWEN_ID))
    model = AutoModelForMultimodalLM.from_pretrained(QWEN_ID, **kw)
    model.to(DEVICE).eval()
    w_hist = weight_llm_ffn(model, DEVICE)
    a_hist, hooks = hook_llm_ffn(model, DEVICE)
    with torch.inference_mode():
        for i in range(0, len(items), BATCH):
            chunk = items[i : i + BATCH]
            waves = [load_audio(x["path"]) for x in chunk]
            packed = processor.apply_transcription_request(audio=waves, language="English")
            packed = packed.to(DEVICE, model.dtype)
            model(**_forward_kwargs(model, dict(packed)))
            done = min(i + BATCH, len(items))
            if done % 80 == 0 or done == len(items) or i == 0:
                print(f"[fig1] Qwen LLM activations {done}/{len(items)}", flush=True)
    for h in hooks:
        h.remove()
    out = {
        "model": "Qwen3-ASR-1.7B LLM",
        "n_utts": len(items),
        "activations": a_hist.pack(),
        "weights": w_hist.pack(),
    }
    del model
    torch.cuda.empty_cache()
    return out


def plot_enc_vs_llm(payload: dict, path_png: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    enc_c, llm_c = "#1f4e79", "#c45c26"
    x_uni = float(np.log10(1.0 / 7.0))
    x_log = float(np.log10(1.0 / 127.0))
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.55), sharex=True, sharey=False)
    series = [
        ("encoder", enc_c, "Whisper encoder FFN"),
        ("llm", llm_c, "Qwen3 LLM FFN"),
    ]
    for ax, kind, title in (
        (axes[0], "activations", "(a) FFN activations"),
        (axes[1], "weights", "(b) FFN weights"),
    ):
        ymax = 0.0
        for key, color, label in series:
            h = payload[key][kind]
            frac = 100.0 * h["frac_below_log_alpha_A_over_127"]
            ax.plot(
                h["log10_centers"],
                h["density"],
                color=color,
                lw=1.8,
                label=rf"{label}  ({frac:.0f}% below $\alpha$)",
            )
            ax.fill_between(h["log10_centers"], h["density"], color=color, alpha=0.14)
            ymax = max(ymax, max(h["density"]) if h["density"] else 0.0)
        ax.axvline(x_uni, color="#333333", ls="--", lw=0.95, zorder=0)
        ax.axvline(x_log, color="#333333", ls=":", lw=0.95, zorder=0)
        ax.set_title(title, loc="left", pad=6)
        ax.set_xlim(LOG_LO, LOG_HI)
        ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1.0)
        ax.set_xlabel(r"$\log_{10}(|x|/A)$")
        ax.set_ylabel("Probability density")
        ax.legend(frameon=False, loc="upper left")
        ax.set_xticks([-4, -3, -2, -1, 0])

    n = payload["n_utts"]
    fig.suptitle(
        f"Speech-encoder FFN vs LLM FFN  ·  LibriSpeech test-clean  (n={n})",
        fontsize=10.5,
        y=0.98,
    )
    handles = [
        Line2D([0], [0], color="#333333", ls="--", lw=0.95, label=r"uniform 4-bit first bin  $A/7$"),
        Line2D(
            [0],
            [0],
            color="#333333",
            ls=":",
            lw=0.95,
            label=r"log 4-bit resolution  $\alpha=A/127$",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.text(
        0.5,
        -0.08,
        r"$A$ = per-tensor absmax (the 4-bit scale of that FFN Linear)",
        ha="center",
        va="top",
        fontsize=8,
        color="#333333",
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22, top=0.86)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    if not WHISPER_JSON.is_file():
        raise FileNotFoundError(f"missing {WHISPER_JSON}")
    whisper = json.loads(WHISPER_JSON.read_text())
    items = load_split(SPLIT)
    print(f"[fig1] split={SPLIT} n={len(items)}  whisper encoder from {WHISPER_JSON}", flush=True)
    qwen = collect_qwen(items)
    payload = {
        "n_utts": len(items),
        "split": f"librispeech-{SPLIT}",
        "bins": N_BINS,
        "note": (
            "Whisper-large-v3 encoder FFN vs Qwen3-ASR-1.7B LLM FFN. "
            "Activations: every FFN Linear input, scaled by per-tensor absmax. "
            "Whisper encoder from full test-clean teacher-forced encoder; "
            "Qwen LLM from ASR prefill (apply_transcription_request, no generate)."
        ),
        "encoder": {
            "model": "whisper-large-v3 encoder FFN",
            "activations": whisper["activations"]["enc_ffn"],
            "weights": whisper["weights"]["enc_ffn"],
        },
        "llm": {
            "model": qwen["model"],
            "activations": qwen["activations"],
            "weights": qwen["weights"],
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload))
    plot_enc_vs_llm(payload, PNG)
    print(f"wrote {JSON_OUT}", flush=True)
    print(f"wrote {PNG}", flush=True)
    for side in ("encoder", "llm"):
        for kind in ("activations", "weights"):
            h = payload[side][kind]
            print(
                f"  {side:8s} {kind:12s} n={h['n']:.3e}  "
                f"below A/7={100*h['frac_below_uniform_A_over_7']:.1f}%  "
                f"below α={100*h['frac_below_log_alpha_A_over_127']:.1f}%",
                flush=True,
            )


if __name__ == "__main__":
    main()
