"""Fig. 1 (Whisper only): encoder FFN vs decoder FFN, activations and weights.

Uses the full LibriSpeech test-clean split (2620 utterances). Histograms
accumulate every FFN Linear input / weight element, each scaled by its
own per-tensor absmax A (the 4-bit scale). PNG only.
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
KEEP_GROUPS = ("enc_ffn", "llm_ffn")
N_BINS = 48
LOG_LO, LOG_HI = -4.0, 0.0
SPLIT = "test-clean"
BATCH = 8
DEC_LEN = 32


class OnlineLogHist:
    """Exact histogram of log10(|x|/A) plus parallel 4th-moment stats of x/A."""

    def __init__(self, device: torch.device):
        self.device = device
        self.counts = torch.zeros(N_BINS, device=device, dtype=torch.float64)
        self.n = torch.zeros((), device=device, dtype=torch.int64)
        self.n_below_uni = torch.zeros((), device=device, dtype=torch.int64)
        self.n_below_log = torch.zeros((), device=device, dtype=torch.int64)
        self.n_tensors = 0
        self.n_m = torch.zeros((), device=device, dtype=torch.float64)
        self.mean = torch.zeros((), device=device, dtype=torch.float64)
        self.m2 = torch.zeros((), device=device, dtype=torch.float64)
        self.m3 = torch.zeros((), device=device, dtype=torch.float64)
        self.m4 = torch.zeros((), device=device, dtype=torch.float64)

    @torch.no_grad()
    def add(self, x: torch.Tensor):
        if not torch.is_tensor(x) or x.numel() == 0:
            return
        xf = x.detach()
        if xf.dtype != torch.float32:
            xf = xf.float()
        amax = xf.abs().amax().clamp_min(1e-12)
        ax = xf.abs()
        n = xf.numel()
        self.n += n
        self.n_tensors += 1
        self.n_below_uni += (ax < (amax / 7.0)).sum()
        self.n_below_log += (ax < (amax / 127.0)).sum()
        logu = torch.log10((ax / amax).clamp_min(1e-12).reshape(-1))
        self.counts += torch.histc(logu, bins=N_BINS, min=LOG_LO, max=LOG_HI).double()
        u = (xf / amax).reshape(-1).double()
        n_b = torch.tensor(float(n), device=self.device, dtype=torch.float64)
        mean_b = u.mean()
        d = u - mean_b
        d2 = d * d
        m2_b = d2.sum()
        m3_b = (d2 * d).sum()
        m4_b = (d2 * d2).sum()
        self._combine(n_b, mean_b, m2_b, m3_b, m4_b)

    def _combine(self, n_b, mean_b, m2_b, m3_b, m4_b):
        n_a, mean_a, m2_a, m3_a, m4_a = self.n_m, self.mean, self.m2, self.m3, self.m4
        n = n_a + n_b
        n_f = n.clamp_min(1.0)
        delta = mean_b - mean_a
        self.n_m = n
        self.mean = (n_a * mean_a + n_b * mean_b) / n_f
        self.m2 = m2_a + m2_b + delta * delta * n_a * n_b / n_f
        self.m3 = (
            m3_a
            + m3_b
            + (delta ** 3) * n_a * n_b * (n_a - n_b) / (n_f * n_f)
            + 3.0 * delta * (n_a * m2_b - n_b * m2_a) / n_f
        )
        self.m4 = (
            m4_a
            + m4_b
            + (delta ** 4) * n_a * n_b * (n_a * n_a - n_a * n_b + n_b * n_b) / (n_f ** 3)
            + 6.0 * (delta * delta) * (n_a * n_a * m2_b + n_b * n_b * m2_a) / (n_f * n_f)
            + 4.0 * delta * (n_a * m3_b - n_b * m3_a) / n_f
        )

    def pack(self) -> dict:
        counts = self.counts.float().cpu().numpy()
        n = int(self.n.item())
        width = (LOG_HI - LOG_LO) / N_BINS
        dens = (counts / max(n * width, 1e-18)).tolist()
        edges = np.linspace(LOG_LO, LOG_HI, N_BINS + 1)
        centers = (0.5 * (edges[:-1] + edges[1:])).tolist()
        m2 = float(self.m2.item())
        m4 = float(self.m4.item())
        n_m = float(self.n_m.item())
        if n_m > 8 and m2 > 0:
            kurt = n_m * m4 / (m2 * m2) - 3.0
        else:
            kurt = float("nan")
        n_uni = int(self.n_below_uni.item())
        n_log = int(self.n_below_log.item())
        return {
            "n": n,
            "n_tensors": self.n_tensors,
            "log10_centers": centers,
            "density": dens,
            "counts": counts.astype(int).tolist(),
            "excess_kurtosis_x_over_A": kurt,
            "frac_below_uniform_A_over_7": n_uni / max(n, 1),
            "frac_below_log_alpha_A_over_127": n_log / max(n, 1),
        }


def hook_ffn(model: nn.Module, device: torch.device):
    groups = {g: OnlineLogHist(device) for g in KEEP_GROUPS}
    hooks = []

    def make(g: str):
        def hook(_mod, inp, _out):
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


def weight_hists(model: nn.Module, device: torch.device) -> dict[str, OnlineLogHist]:
    groups = {g: OnlineLogHist(device) for g in KEEP_GROUPS}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        g = layer_group(name)
        if g not in groups:
            continue
        groups[g].add(mod.weight.detach())
    return groups


def plot_whisper(payload: dict, path_png: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    from matplotlib.lines import Line2D

    enc_c, dec_c = "#1f4e79", "#c45c26"
    x_uni = float(np.log10(1.0 / 7.0))
    x_log = float(np.log10(1.0 / 127.0))
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.55), sharex=True, sharey=False)

    panels = [
        (axes[0], "activations", "(a) FFN activations"),
        (axes[1], "weights", "(b) FFN weights"),
    ]
    for ax, kind, title in panels:
        block = payload[kind]
        ymax = 0.0
        for g, color, label in (
            ("enc_ffn", enc_c, "Encoder FFN"),
            ("llm_ffn", dec_c, "Decoder FFN"),
        ):
            h = block[g]
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
        f"Whisper-large-v3  ·  LibriSpeech test-clean  (n={n})",
        fontsize=11,
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


def decoder_teacher_ids(processor, model, texts: list[str]) -> torch.Tensor:
    """Whisper prefix + tokenized reference, padded/truncated to DEC_LEN."""
    start = model.config.decoder_start_token_id
    prompt = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    prefix = [start] + [tok for _, tok in prompt]
    body_len = max(DEC_LEN - len(prefix), 1)
    body = processor.tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=body_len,
        return_tensors="pt",
    ).input_ids
    pref = torch.tensor([prefix] * len(texts), dtype=torch.long)
    return torch.cat([pref, body], dim=1)[:, :DEC_LEN]


def main():
    items = load_split(SPLIT)
    print(f"[fig1-wh] split={SPLIT} n={len(items)}", flush=True)
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3", torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.to(DEVICE).eval()
    print("[fig1-wh] collecting weights (all FFN parameters)", flush=True)
    w_groups = weight_hists(model, DEVICE)
    a_groups, hooks = hook_ffn(model, DEVICE)
    with torch.inference_mode():
        for i in range(0, len(items), BATCH):
            chunk = items[i : i + BATCH]
            waves = [pad_to_30s(load_audio(x["path"])) for x in chunk]
            feats = processor(waves, sampling_rate=SR, return_tensors="pt").input_features
            feats = feats.to(DEVICE, dtype=torch.float16)
            dec = decoder_teacher_ids(processor, model, [x["text"] for x in chunk]).to(DEVICE)
            model(input_features=feats, decoder_input_ids=dec)
            done = min(i + BATCH, len(items))
            if done % 80 == 0 or done == len(items):
                print(f"[fig1-wh] activations {done}/{len(items)}", flush=True)
    for h in hooks:
        h.remove()
    payload = {
        "model": "whisper-large-v3",
        "split": f"librispeech-{SPLIT}",
        "n_utts": len(items),
        "decoder_len": DEC_LEN,
        "note": (
            "Online histogram over every FFN Linear input on the full test-clean "
            "split (encoder + teacher-forced decoder with reference text) and every "
            "FFN weight element. Each tensor is scaled by its own absmax A."
        ),
        "activations": {g: a_groups[g].pack() for g in KEEP_GROUPS},
        "weights": {g: w_groups[g].pack() for g in KEEP_GROUPS},
    }
    del model
    torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "fig1_whisper_testclean.json"
    json_path.write_text(json.dumps(payload))
    png_path = FIG / "fig1_whisper_testclean.png"
    plot_whisper(payload, png_path)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {png_path}", flush=True)
    for kind in ("activations", "weights"):
        for g in KEEP_GROUPS:
            h = payload[kind][g]
            print(
                f"  {kind:12s} {g:8s} n={h['n']:.3e}  "
                f"below A/7={100*h['frac_below_uniform_A_over_7']:.1f}%  "
                f"κ(x/A)={h['excess_kurtosis_x_over_A']:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
