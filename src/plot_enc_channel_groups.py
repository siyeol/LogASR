"""Activation channel curves on full LibriSpeech test-clean.

Rows: Whisper encoder / Qwen encoder / Qwen decoder.
Cols: (all) / attention / FFN.
Hooks Linear *inputs* (activations), not weights. Encoder-only (Whisper)
or ASR prefill (Qwen) — no generate.
"""

from __future__ import annotations

import inspect
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
from plot_fig2_channels import N_GRID, curves_from_buf

DEVICE = torch.device("cuda")
OUT = Path("/workspace/SpeechPTQ/results/diagnostics")
FIG = Path("/workspace/SpeechPTQ/results/figures")
SPLIT = "test-clean"
ENC_GROUPS = ("enc_all", "enc_attn", "enc_ffn")
DEC_GROUPS = ("llm_all", "llm_attn", "llm_ffn")
ALL_GROUPS = ENC_GROUPS + DEC_GROUPS
WHISPER_BS = 8
QWEN_BS = 2
PNG = FIG / "fig_enc_channel_groups.png"
PNG_PAPER = FIG / "fig_enc_channel_groups_paper.png"
PDF_PAPER = FIG / "fig_enc_channel_groups_paper.pdf"
JSON_OUT = OUT / "fig_enc_channel_groups.json"


def hook_channels(model: nn.Module, prefixes: tuple[str, ...]):
    buf: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    hooks = []
    n_hook: dict[str, int] = defaultdict(int)

    def make(name: str, keys: tuple[str, ...]):
        def hook(_mod, inp, _out):
            x = inp[0]
            if not torch.is_tensor(x) or x.ndim < 2:
                return
            amax = x.detach().float().abs().amax(dim=tuple(range(x.ndim - 1))).cpu()
            for k in keys:
                prev = buf[k].get(name)
                buf[k][name] = amax if prev is None else torch.maximum(prev, amax)

        return hook

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        g = layer_group(name)
        keys: list[str] = []
        if g.startswith("enc_") and "enc" in prefixes:
            keys.append("enc_all")
            if g in ENC_GROUPS:
                keys.append(g)
        if g.startswith("llm_") and "llm" in prefixes:
            keys.append("llm_all")
            if g in DEC_GROUPS:
                keys.append(g)
        if not keys:
            continue
        for k in keys:
            n_hook[k] += 1
        hooks.append(mod.register_forward_hook(make(name, tuple(keys))))
    print(f"[enc-ch] hooked {dict(n_hook)}", flush=True)
    return buf, hooks


def run_whisper(items: list[dict]) -> dict:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3", torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.to(DEVICE).eval()
    buf, hooks = hook_channels(model, ("enc",))
    enc = model.model.encoder
    with torch.inference_mode():
        for i in range(0, len(items), WHISPER_BS):
            chunk = items[i : i + WHISPER_BS]
            waves = [pad_to_30s(load_audio(x["path"])) for x in chunk]
            feats = processor(waves, sampling_rate=SR, return_tensors="pt").input_features
            enc(feats.to(DEVICE, dtype=torch.float16))
            done = min(i + WHISPER_BS, len(items))
            if done % 160 == 0 or done == len(items) or i == 0:
                print(f"[enc-ch] whisper {done}/{len(items)}", flush=True)
    for h in hooks:
        h.remove()
    out = {"model": "whisper-large-v3", "n_utts": len(items), **curves_from_buf(buf)}
    del model
    torch.cuda.empty_cache()
    return out


def run_qwen(items: list[dict]) -> dict:
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    from run_eval_qwen3 import compressed_tensors_load_kwargs

    model_id = "Qwen/Qwen3-ASR-1.7B-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True)
    kw.update(compressed_tensors_load_kwargs(model_id))
    model = AutoModelForMultimodalLM.from_pretrained(model_id, **kw)
    model.to(DEVICE).eval()
    buf, hooks = hook_channels(model, ("enc", "llm"))
    fwd_names = set(inspect.signature(model.forward).parameters)
    with torch.inference_mode():
        for i in range(0, len(items), QWEN_BS):
            chunk = items[i : i + QWEN_BS]
            waves = [load_audio(x["path"]) for x in chunk]
            packed = processor.apply_transcription_request(audio=waves, language="English")
            packed = packed.to(DEVICE, model.dtype)
            model(**{k: v for k, v in dict(packed).items() if k in fwd_names})
            done = min(i + QWEN_BS, len(items))
            if done % 160 == 0 or done == len(items) or i == 0:
                print(f"[enc-ch] qwen {done}/{len(items)}", flush=True)
    for h in hooks:
        h.remove()
    out = {"model": "Qwen3-ASR-1.7B", "n_utts": len(items), **curves_from_buf(buf)}
    del model
    torch.cuda.empty_cache()
    return out


def plot(payload: dict, path: Path, extra: tuple[Path, ...] = ()):
    import matplotlib
    from matplotlib import font_manager

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font_manager._load_fontmanager(try_read_cache=False)
    tnr_dir = Path("/usr/share/fonts/truetype/msttcorefonts")
    for fname in (
        "Times_New_Roman.ttf",
        "Times_New_Roman_Bold.ttf",
        "Times_New_Roman_Italic.ttf",
        "Times_New_Roman_Bold_Italic.ttf",
        "times.ttf",
        "timesbd.ttf",
        "timesi.ttf",
        "timesbi.ttf",
    ):
        fpath = tnr_dir / fname
        if fpath.is_file():
            font_manager.fontManager.addfont(str(fpath))
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    colors = {
        "enc_all": "#333333",
        "enc_attn": "#2a9d8f",
        "enc_ffn": "#1f4e79",
        "llm_all": "#333333",
        "llm_attn": "#2a9d8f",
        "llm_ffn": "#c45c26",
    }
    col_titles = ("(all)", "attention", "FFN")
    rows = (
        ("whisper", "Whisper encoder", ENC_GROUPS),
        ("qwen3", "Qwen encoder", ENC_GROUPS),
        ("qwen3", "Qwen decoder", DEC_GROUPS),
    )
    ymax = 1e5
    for _mkey, _mname, groups in rows:
        block = payload[_mkey]
        for g in groups:
            h = block.get(g)
            if h:
                ymax = max(ymax, float(h["outlier_ratio_max"]) * 1.5)
    fig, axes = plt.subplots(3, 3, figsize=(7.16, 7.4), sharex=True, sharey=True)
    for r, (mkey, mname, groups) in enumerate(rows):
        block = payload[mkey]
        for c, g in enumerate(groups):
            ax = axes[r, c]
            h = block.get(g)
            if not h:
                ax.set_title(f"{col_titles[c]} (missing)", loc="center")
                continue
            x = np.array(h["rank"]) * 100.0
            ax.plot(x, h["worst_layer"], color=colors[g], lw=1.6, label="worst layer")
            ax.plot(
                x,
                h["median_over_median"],
                color=colors[g],
                lw=1.0,
                ls="--",
                alpha=0.85,
                label="median layer",
            )
            ax.fill_between(x, h["p25"], h["p75"], color=colors[g], alpha=0.12)
            ax.axhline(1.0, color="#888888", lw=0.5, ls=":")
            ax.set_yscale("log")
            ax.set_xlim(0, 50)
            ax.set_ylim(0.4, ymax)
            if r == 0:
                ax.set_title(col_titles[c], loc="center", pad=3, fontsize=9)
            ax.text(
                0.97,
                0.95,
                f"worst {h['outlier_ratio_max']:.0f}×\n"
                f"med {h['outlier_ratio_median']:.1f}×  L={h['n_layers']}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="#333333",
                linespacing=1.25,
                fontfamily="Times New Roman",
            )
            if r == 2:
                ax.set_xlabel("Channel rank (%)", fontsize=9)
            if c == 0:
                ax.set_ylabel(r"channel $\max|x|$ / median", fontsize=9)
                ax.text(
                    -0.40,
                    0.5,
                    mname,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontfamily="Times New Roman",
                )
    n = payload["n_utts"]
    fig.suptitle(
        f"Activation channel outliers  ·  LibriSpeech test-clean  (n={n})",
        fontsize=11,
        y=0.985,
    )
    fig.legend(
        handles=[
            plt.Line2D([0], [0], color="#333333", lw=1.6, label="worst layer"),
            plt.Line2D([0], [0], color="#333333", lw=1.0, ls="--", label="median layer"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.56, 0.04),
        fontsize=8,
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.93, bottom=0.12, left=0.15, hspace=0.28, wspace=0.16)
    fig.text(
        0.56,
        0.012,
        r"Activation = Linear input.  $1\times$ = typical channel.  "
        r"Qwen decoder = LLM stack on ASR prefill.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#333333",
    )
    FIG.mkdir(parents=True, exist_ok=True)
    for p in (path, *extra):
        fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    if JSON_OUT.is_file() and os.environ.get("PLOT_ONLY") == "1":
        payload = json.loads(JSON_OUT.read_text())
        plot(payload, PNG, extra=(PNG_PAPER, PDF_PAPER))
        print(f"replot {PNG}", flush=True)
        print(f"replot {PNG_PAPER}", flush=True)
        print(f"replot {PDF_PAPER}", flush=True)
        return
    items = load_split(SPLIT)
    print(f"[enc-ch] split={SPLIT} n={len(items)}", flush=True)
    if os.environ.get("QWEN_ONLY") == "1" and JSON_OUT.is_file():
        payload = json.loads(JSON_OUT.read_text())
        print("[enc-ch] reuse Whisper encoder; collect Qwen encoder+decoder", flush=True)
        qwen = run_qwen(items)
        payload["qwen3"] = qwen
        payload["n_utts"] = len(items)
    else:
        whisper = run_whisper(items)
        qwen = run_qwen(items)
        payload = {
            "split": f"librispeech-{SPLIT}",
            "n_utts": len(items),
            "kind": "activation_channel_absmax",
            "whisper": whisper,
            "qwen3": qwen,
        }
    OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload))
    plot(payload, PNG, extra=(PNG_PAPER, PDF_PAPER))
    print(f"wrote {JSON_OUT}", flush=True)
    print(f"wrote {PNG}", flush=True)
    print(f"wrote {PNG_PAPER}", flush=True)
    print(f"wrote {PDF_PAPER}", flush=True)
    for m in ("whisper", "qwen3"):
        for g in ALL_GROUPS:
            h = payload[m].get(g)
            if not h:
                continue
            print(
                f"  {m:8s} {g:8s} L={h['n_layers']:3d}  "
                f"worst={h['outlier_ratio_max']:.1f}×  med={h['outlier_ratio_median']:.1f}×",
                flush=True,
            )


if __name__ == "__main__":
    main()
