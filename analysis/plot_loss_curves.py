from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.save_fig import save_fig, apply_style
apply_style()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help="One or more PATH=LABEL pairs.")
    p.add_argument("--metric", default="loss",
                   help="Key in metrics.jsonl to plot (e.g., loss, grad_norm_llama, sel_entropy).")
    p.add_argument("--out", required=True, help="Output base path (no extension).")
    p.add_argument("--title", default=None)
    p.add_argument("--xlabel", default="step")
    p.add_argument("--ylabel", default=None)
    p.add_argument("--ylog", action="store_true", help="log scale on y axis")
    p.add_argument("--smooth", type=int, default=1, help="Window for moving average smoothing (1 = no smoothing).")
    p.add_argument("--figsize", nargs=2, type=float, default=(8, 5))
    return p.parse_args()


def load_metrics(path):
    rows = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def smooth(values, window):
    if window <= 1:
        return values
    import numpy as np
    arr = np.array(values, dtype=float)
    pad = window // 2
    padded = np.pad(arr, pad, mode="edge")
    kernel = np.ones(window) / window
    return list(np.convolve(padded, kernel, mode="valid")[: len(values)])


def main():
    args = parse_args()
    fig, ax = plt.subplots(figsize=tuple(args.figsize))

    for spec in args.runs:
        if "=" not in spec:
            print(f"Skipping bad spec (no '='): {spec}", file=sys.stderr)
            continue
        path, label = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"WARNING: missing metrics file: {path}")
            continue
        rows = load_metrics(path)
        if not rows:
            print(f"WARNING: empty metrics: {path}")
            continue
        steps = [r["step"] for r in rows if args.metric in r and r[args.metric] is not None]
        vals = [r[args.metric] for r in rows if args.metric in r and r[args.metric] is not None]
        if not vals:
            print(f"WARNING: no '{args.metric}' values in {path}")
            continue
        vals_smooth = smooth(vals, args.smooth)
        ax.plot(steps, vals_smooth, label=label, linewidth=1.5)

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel or args.metric)
    if args.title:
        ax.set_title(args.title)
    if args.ylog:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    save_fig(fig, args.out)


if __name__ == "__main__":
    main()
