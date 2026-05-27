from __future__ import annotations

import os

import matplotlib.pyplot as plt


def apply_style():
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 17,
        "legend.title_fontsize": 18,
        "figure.titlesize": 24,
        "axes.linewidth": 1.4,
        "lines.linewidth": 2.5,
    })


def save_fig(fig, path_base: str, dpi: int = 150):
    base = os.path.splitext(path_base)[0]
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=dpi, bbox_inches="tight")
    print(f"saved {base}.{{pdf,png}}")
