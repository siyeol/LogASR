"""Fig. 1: why log (grid) vs why not LLM grouping (channels).

(a) Whisper encoder FFN |x|/A vs the 4-bit grid — no text overlay.
(b) Sorted channel absmax: speech-encoder FFN (dense) vs LLM FFN (needle).

Replots existing JSONs. No GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from plot_fig1_whisper_full import FIG, LOG_HI, LOG_LO, OUT

ACT_JSON = OUT / "fig1_whisper_testclean.json"
CH_JSON = OUT / "fig2_channel_outliers.json"
PNG = FIG / "fig1_grid_and_channels.png"


def plot(act: dict, ch: dict, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

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
    enc_c, llm_c = "#1f4e79", "#c45c26"
    x_uni = float(np.log10(1.0 / 7.0))
    x_log = float(np.log10(1.0 / 127.0))
    h = act["activations"]["enc_ffn"]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.35))

    ax = axes[0]
    ax.axvspan(LOG_LO, x_uni, color="#9aa3ad", alpha=0.14, zorder=0)
    ax.plot(h["log10_centers"], h["density"], color=enc_c, lw=1.9)
    ax.fill_between(h["log10_centers"], h["density"], color=enc_c, alpha=0.16)
    ax.axvline(x_uni, color="#333333", ls="--", lw=0.95)
    ax.axvline(x_log, color="#333333", ls=":", lw=0.95)
    ymax = max(h["density"]) * 1.18
    ax.text(x_uni - 0.08, ymax * 0.92, r"uniform $A/7$", ha="right", va="top", fontsize=7.5, color="#333333")
    ax.text(x_log + 0.08, ymax * 0.92, r"log $\alpha{=}A/127$", ha="left", va="top", fontsize=7.5, color="#333333")
    frac7 = 100 * h["frac_below_uniform_A_over_7"]
    frac_a = 100 * h["frac_below_log_alpha_A_over_127"]
    ax.text(
        0.03,
        0.05,
        rf"{frac7:.0f}% of mass below $A/7$" + "\n" + rf"{frac_a:.0f}% below $\alpha$",
        transform=ax.transAxes,
        fontsize=8,
        color=enc_c,
        va="bottom",
    )
    ax.set_title("(a) Encoder FFN activations vs 4-bit grid", loc="left", pad=6)
    ax.set_xlim(LOG_LO, LOG_HI)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r"$\log_{10}(|x|/A)$")
    ax.set_ylabel("Probability density")
    ax.set_xticks([-4, -3, -2, -1, 0])

    ax = axes[1]
    enc = ch["whisper"]["enc_ffn"]
    llm = ch["qwen3"]["llm_ffn"]
    ax.plot(
        np.array(enc["rank"]) * 100,
        enc["worst_layer"],
        color=enc_c,
        lw=1.8,
        label=rf"Whisper encoder FFN  (worst ${enc['outlier_ratio_max']:.0f}\times$)",
    )
    ax.plot(
        np.array(llm["rank"]) * 100,
        llm["worst_layer"],
        color=llm_c,
        lw=1.8,
        label=rf"Qwen3 LLM FFN  (worst ${llm['outlier_ratio_max']:.0f}\times$)",
    )
    ax.set_yscale("log")
    ax.set_xlim(0, 45)
    ax.set_ylim(0.5, 8e4)
    ax.axhline(1.0, color="#888888", lw=0.6, ls=":")
    ax.set_title("(b) Channel absmax / median (worst layer)", loc="left", pad=6)
    ax.set_xlabel("Channel rank (%)")
    ax.set_ylabel(r"channel $\max|x|$ / median")
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    ax.annotate(
        "dense tail",
        xy=(12.6, 325),
        xytext=(22, 1200),
        fontsize=8,
        color=enc_c,
        arrowprops=dict(arrowstyle="->", color=enc_c, lw=0.7),
    )
    ax.annotate(
        "1-channel needle",
        xy=(0.4, 37042),
        xytext=(8, 25000),
        fontsize=8,
        color=llm_c,
        arrowprops=dict(arrowstyle="->", color=llm_c, lw=0.7),
    )

    n_a = act["n_utts"]
    n_c = ch["n_utts"]
    fig.suptitle("Why log on the speech encoder, not LLM-style grouping", fontsize=11, y=0.98)
    handles = [
        Patch(facecolor="#9aa3ad", alpha=0.35, edgecolor="none", label=r"uniform INT4 cannot resolve ($|x|<A/7$)"),
        Line2D([0], [0], color="#333333", ls="--", lw=0.95, label=r"uniform first bin $A/7$"),
        Line2D([0], [0], color="#333333", ls=":", lw=0.95, label=r"log resolution $\alpha=A/127$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.text(
        0.5,
        -0.09,
        rf"(a) Whisper encoder FFN, LibriSpeech test-clean $n={n_a}$.  "
        rf"(b) worst-layer channel curve, test-other $n={n_c}$.  "
        r"$A$ = per-tensor absmax.",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#333333",
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22, top=0.86, wspace=0.32)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    act = json.loads(ACT_JSON.read_text())
    ch = json.loads(CH_JSON.read_text())
    plot(act, ch, PNG)
    print(f"wrote {PNG}")


if __name__ == "__main__":
    main()
