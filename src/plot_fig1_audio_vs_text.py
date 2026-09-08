"""Fig. 1: Whisper audio vs text FFN activations (no Qwen, no weights).

Replots the full test-clean histograms in fig1_whisper_testclean.json.
(a) density  (b) CDF — so the mass left of A/7 vs α=A/127 is readable at a glance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from plot_fig1_whisper_full import FIG, LOG_HI, LOG_LO, N_BINS, OUT

JSON_IN = OUT / "fig1_whisper_testclean.json"
PNG = FIG / "fig1_audio_vs_text_activations.png"


def cdf_from_hist(h: dict) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(h["counts"], dtype=np.float64)
    n = float(h["n"])
    underflow = max(n - counts.sum(), 0.0)
    centers = np.asarray(h["log10_centers"], dtype=np.float64)
    cdf = np.cumsum(counts) / n
    cdf = np.clip(cdf + underflow / n, 0.0, 1.0)
    return centers, cdf


def plot(payload: dict, path: Path):
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
    audio_c, text_c = "#1f4e79", "#c45c26"
    x_uni = float(np.log10(1.0 / 7.0))
    x_log = float(np.log10(1.0 / 127.0))
    acts = payload["activations"]
    audio, text = acts["enc_ffn"], acts["llm_ffn"]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.45), sharex=True)

    ax = axes[0]
    ymax = 0.0
    for h, color, label in (
        (audio, audio_c, "Audio  (encoder FFN)"),
        (text, text_c, "Text  (decoder FFN)"),
    ):
        ax.plot(h["log10_centers"], h["density"], color=color, lw=1.8, label=label)
        ax.fill_between(h["log10_centers"], h["density"], color=color, alpha=0.12)
        ymax = max(ymax, max(h["density"]))
    ax.axvspan(LOG_LO, x_uni, color="#9aa3ad", alpha=0.12, zorder=0)
    ax.axvline(x_uni, color="#333333", ls="--", lw=0.95, zorder=1)
    ax.axvline(x_log, color="#333333", ls=":", lw=0.95, zorder=1)
    ax.set_title("(a) Activation density", loc="left", pad=6)
    ax.set_xlim(LOG_LO, LOG_HI)
    ax.set_ylim(0, ymax * 1.12)
    ax.set_xlabel(r"$\log_{10}(|x|/A)$")
    ax.set_ylabel("Probability density")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xticks([-4, -3, -2, -1, 0])

    ax = axes[1]
    for h, color, label in (
        (audio, audio_c, "Audio"),
        (text, text_c, "Text"),
    ):
        xs, ys = cdf_from_hist(h)
        ax.plot(xs, ys, color=color, lw=1.8, label=label)
    ax.axvspan(LOG_LO, x_uni, color="#9aa3ad", alpha=0.12, zorder=0)
    ax.axvline(x_uni, color="#333333", ls="--", lw=0.95, zorder=1)
    ax.axvline(x_log, color="#333333", ls=":", lw=0.95, zorder=1)
    ax.axhline(audio["frac_below_log_alpha_A_over_127"], color=audio_c, ls=":", lw=0.6, alpha=0.7)
    ax.axhline(text["frac_below_log_alpha_A_over_127"], color=text_c, ls=":", lw=0.6, alpha=0.7)
    a_a = 100 * audio["frac_below_log_alpha_A_over_127"]
    t_a = 100 * text["frac_below_log_alpha_A_over_127"]
    ax.annotate(
        rf"Audio {a_a:.0f}% below $\alpha$",
        xy=(x_log, audio["frac_below_log_alpha_A_over_127"]),
        xytext=(x_log + 0.35, audio["frac_below_log_alpha_A_over_127"] + 0.08),
        fontsize=7.5,
        color=audio_c,
        arrowprops=dict(arrowstyle="-", color=audio_c, lw=0.6),
    )
    ax.annotate(
        rf"Text {t_a:.0f}% below $\alpha$",
        xy=(x_log, text["frac_below_log_alpha_A_over_127"]),
        xytext=(x_log + 0.55, text["frac_below_log_alpha_A_over_127"] - 0.14),
        fontsize=7.5,
        color=text_c,
        arrowprops=dict(arrowstyle="-", color=text_c, lw=0.6),
    )
    ax.set_title("(b) Cumulative mass", loc="left", pad=6)
    ax.set_xlim(LOG_LO, LOG_HI)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"$\log_{10}(|x|/A)$")
    ax.set_ylabel(r"$P(|x| \leq t\cdot A)$")
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks([-4, -3, -2, -1, 0])

    n = payload["n_utts"]
    fig.suptitle(
        f"Whisper-large-v3  ·  audio vs text FFN activations  ·  test-clean (n={n})",
        fontsize=10.5,
        y=0.98,
    )
    handles = [
        Patch(facecolor="#9aa3ad", alpha=0.35, edgecolor="none", label="uniform 4-bit cannot resolve  ($|x|<A/7$)"),
        Line2D([0], [0], color="#333333", ls="--", lw=0.95, label=r"uniform first bin  $A/7$"),
        Line2D([0], [0], color="#333333", ls=":", lw=0.95, label=r"log resolution  $\alpha=A/127$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.text(
        0.5,
        -0.10,
        r"$A$ = per-tensor absmax. Gray band = mass collapsed by grouping-free uniform INT4.",
        ha="center",
        va="top",
        fontsize=8,
        color="#333333",
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24, top=0.86)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    payload = json.loads(JSON_IN.read_text())
    plot(payload, PNG)
    print(f"wrote {PNG}")


if __name__ == "__main__":
    main()
