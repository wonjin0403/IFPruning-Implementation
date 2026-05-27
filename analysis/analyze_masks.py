from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.save_fig import save_fig, apply_style
apply_style()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--masks", required=True, help=".pt produced by eval/extract_masks.py")
    p.add_argument("--out_dir", required=True)
    return p.parse_args()


def jaccard_matrix(indices: torch.Tensor) -> np.ndarray:
    N, L, K = indices.shape
    sets = [[set(indices[n, l].tolist()) for l in range(L)] for n in range(N)]
    M = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i, N):
            js = []
            for l in range(L):
                a, b = sets[i][l], sets[j][l]
                inter = len(a & b)
                union = len(a | b)
                js.append(inter / max(1, union))
            M[i, j] = M[j, i] = float(np.mean(js))
    return M


def plot_jaccard(M, categories, out_base):
    order = sorted(range(len(categories)), key=lambda i: categories[i])
    M_sorted = M[np.ix_(order, order)]
    cats_sorted = [categories[i] for i in order]

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(M_sorted, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=22)

    boundaries = []
    cur = cats_sorted[0]
    start = 0
    cat_centers = []
    for i, c in enumerate(cats_sorted + [None]):
        if c != cur:
            boundaries.append(i - 0.5)
            cat_centers.append((start + i - 1) / 2)
            start = i
            cur = c
    for b in boundaries[:-1]:
        ax.axhline(b, color="red", linewidth=0.7)
        ax.axvline(b, color="red", linewidth=0.7)

    uniq_cats = []
    seen = set()
    for c in cats_sorted:
        if c not in seen:
            uniq_cats.append(c)
            seen.add(c)
    ax.set_xticks(cat_centers[: len(uniq_cats)])
    ax.set_xticklabels(uniq_cats, rotation=45, ha="right", fontsize=24)
    ax.set_yticks(cat_centers[: len(uniq_cats)])
    ax.set_yticklabels(uniq_cats, fontsize=24)

    fig.tight_layout()
    save_fig(fig, out_base)


def plot_channel_freq(hard_indices, num_layers, intermediate_size, out_base, n_show_layers=4):
    N, L, K = hard_indices.shape
    freq = np.zeros((L, intermediate_size), dtype=np.float32)
    for n in range(N):
        for l in range(L):
            freq[l, hard_indices[n, l].numpy()] += 1
    freq /= N

    layers_to_show = np.linspace(0, L - 1, n_show_layers, dtype=int)
    fig, axes = plt.subplots(len(layers_to_show), 1, figsize=(12, 2.5 * len(layers_to_show)), sharex=True)
    if len(layers_to_show) == 1:
        axes = [axes]
    for ax, l in zip(axes, layers_to_show):
        ax.bar(range(intermediate_size), freq[l], width=1.0, color="steelblue")
        ax.set_ylabel(f"layer {l}", fontsize=18)
        ax.set_ylim(0, 1.05)
        ax.axhline(K / intermediate_size, color="red", linestyle="--", linewidth=2,
                   label=f"uniform baseline {K/intermediate_size:.2f}")
        ax.tick_params(axis='both', labelsize=16)
    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("channel index")
    fig.tight_layout()
    save_fig(fig, out_base)
    return freq


def plot_mask_value_dist(soft_masks, hard_indices, out_base):
    N, L, D = soft_masks.shape
    vals = []
    for n in range(N):
        for l in range(L):
            idx = hard_indices[n, l]
            vals.append(soft_masks[n, l, idx].numpy())
    vals = np.concatenate(vals)
    fig, ax = plt.subplots(figsize=(10, 6))
    counts, bin_edges, _ = ax.hist(vals, bins=50, color="steelblue", edgecolor="black", alpha=0.85)
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e6:.1f}M"))
    ax.axvline(float(vals.mean()), color="red", linestyle="--", linewidth=2,
               label=f"mean={vals.mean():.3f}")
    ax.set_xlabel("Soft mask value at top-k positions")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, out_base)
    return vals


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    payload = torch.load(args.masks, map_location="cpu")
    soft = payload["soft_masks"]
    hard = payload["hard_indices"]
    categories = payload["categories"]
    meta = payload["meta"]
    N, L, D = soft.shape
    K = meta["topk"]
    print(f"Loaded {N} prompts, {L} layers, {D} channels, top-k={K}")

    print("Computing Jaccard matrix ...")
    J = jaccard_matrix(hard)
    plot_jaccard(J, categories, os.path.join(args.out_dir, "jaccard_heatmap"))

    print("Plotting channel frequency ...")
    freq = plot_channel_freq(hard, L, D, os.path.join(args.out_dir, "channel_freq"))

    print("Plotting mask value distribution ...")
    mask_vals = plot_mask_value_dist(soft, hard, os.path.join(args.out_dir, "mask_value_dist"))

    cats = np.array(categories)
    same = J[(cats[:, None] == cats[None, :]) & ~np.eye(N, dtype=bool)]
    diff = J[cats[:, None] != cats[None, :]]
    stats = {
        "meta": meta,
        "num_prompts": N,
        "categories_unique": sorted(set(categories)),
        "jaccard_within_category_mean": float(same.mean()) if same.size else 0.0,
        "jaccard_across_category_mean": float(diff.mean()) if diff.size else 0.0,
        "jaccard_differentiation_gap": float(same.mean() - diff.mean()) if same.size and diff.size else 0.0,
        "selection_freq_entropy_per_layer": [
            float(-(p[p > 0] * np.log(p[p > 0])).sum()) for p in freq
        ],
        "mask_value_mean": float(mask_vals.mean()),
        "mask_value_std": float(mask_vals.std()),
        "mask_value_min": float(mask_vals.min()),
        "mask_value_max": float(mask_vals.max()),
    }
    stats_path = os.path.join(args.out_dir, "mask_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSummary:")
    print(f"  within-category Jaccard: {stats['jaccard_within_category_mean']:.4f}")
    print(f"  across-category Jaccard: {stats['jaccard_across_category_mean']:.4f}")
    print(f"  differentiation gap:     {stats['jaccard_differentiation_gap']:.4f}  (positive = good differentiation)")
    print(f"  mask value mean / std:   {stats['mask_value_mean']:.3f} / {stats['mask_value_std']:.3f}")
    print(f"  stats -> {stats_path}")


if __name__ == "__main__":
    main()
