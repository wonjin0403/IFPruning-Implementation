from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.save_fig import save_fig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True, help="VARIANT=PATH pairs")
    p.add_argument("--out", required=True, help="Figure output base path")
    p.add_argument("--csv", required=True, help="CSV table output")
    p.add_argument("--metric", default="acc", help="Metric key in the results dict (acc, acc_norm, exact_match, ...)")
    return p.parse_args()


def main():
    args = parse_args()

    variants = []
    data = {}
    all_tasks = set()
    for spec in args.results:
        v, p = spec.split("=", 1)
        with open(p) as f:
            raw = json.load(f)
        variants.append(v)
        results = raw.get("results", raw)
        d = {}
        for t, m in results.items():
            score = None
            for key in [args.metric, f"{args.metric},none", f"{args.metric}_norm,none", "acc,none", "exact_match,strict-match", "exact_match,flexible-extract"]:
                if key in m:
                    score = m[key]
                    break
            if score is not None:
                d[t] = float(score) * 100
                all_tasks.add(t)
        data[v] = d

    tasks = sorted(all_tasks)
    print(f"variants: {variants}")
    print(f"tasks: {tasks}")

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant"] + tasks)
        for v in variants:
            w.writerow([v] + [f"{data[v].get(t, float('nan')):.2f}" for t in tasks])
    print(f"CSV -> {args.csv}")

    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 1.6), 5))
    x = np.arange(len(tasks))
    width = 0.8 / len(variants)
    for i, v in enumerate(variants):
        scores = [data[v].get(t, float("nan")) for t in tasks]
        offset = (i - (len(variants) - 1) / 2) * width
        bars = ax.bar(x + offset, scores, width, label=v)
        for b, s in zip(bars, scores):
            if not np.isnan(s):
                ax.text(b.get_x() + b.get_width() / 2, s + 0.5, f"{s:.1f}",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_ylabel(f"{args.metric} (%)")
    ax.set_title("Benchmark scores by mask variant")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, args.out)


if __name__ == "__main__":
    main()
