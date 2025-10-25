#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_training.py
Plot round-wise aggregated metrics from a training run directory.

It reads files named round_XXXX_meta.json (both no-HA and HA servers write these),
and plots available metrics such as loss/acc and val_* if present.

Usage:
  python -m experiments.plot_training --run results/<run_dir> --out results/metrics.png
  # Compare two runs side-by-side
  python -m experiments.plot_training --run results/<ha_run> --run2 results/<noha_run> --out results/compare_metrics.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_round_metas(run_dir: Path) -> Dict[int, Dict]:
    metas: Dict[int, Dict] = {}
    for p in sorted(run_dir.glob("round_*_meta.json")):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            r = int(data.get("round"))
            metas[r] = data
        except Exception:
            pass
    return metas


def extract_series(metas: Dict[int, Dict], keys: List[str]) -> Dict[str, List[float]]:
    rounds = sorted(metas.keys())
    series: Dict[str, List[float]] = {k: [] for k in keys}
    for r in rounds:
        m = metas[r].get("metrics", {})
        for k in keys:
            v = m.get(k)
            series[k].append(float(v) if isinstance(v, (int, float)) else float("nan"))
    return rounds, series


def plot_one(ax, run_name: str, rounds: List[int], series: Dict[str, List[float]]):
    colors = {
        "loss": "tab:blue",
        "acc": "tab:green",
        "val_loss": "tab:orange",
        "val_acc": "tab:red",
        "val_macro_f1": "tab:purple",
    }
    for k, ys in series.items():
        if all((v != v) for v in ys):  # all NaN
            continue
        ax.plot(rounds, ys, marker="o", label=k, color=colors.get(k))
    ax.set_title(run_name)
    ax.set_xlabel("round")
    ax.set_ylabel("metric")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")


def main():
    ap = argparse.ArgumentParser(description="Plot training metrics per round")
    ap.add_argument("--run", type=str, required=True)
    ap.add_argument("--run2", type=str, default=None)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    run1 = Path(args.run)
    metas1 = load_round_metas(run1)
    rounds1, series1 = extract_series(metas1, ["loss", "acc", "val_loss", "val_acc", "val_macro_f1"])

    if args.run2:
        run2 = Path(args.run2)
        metas2 = load_round_metas(run2)
        rounds2, series2 = extract_series(metas2, ["loss", "acc", "val_loss", "val_acc", "val_macro_f1"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        plot_one(axes[0], run1.name, rounds1, series1)
        plot_one(axes[1], run2.name, rounds2, series2)
        fig.tight_layout()
        fig.savefig(args.out, dpi=200)
        print(f"[plot_training] wrote {args.out}")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        plot_one(ax, run1.name, rounds1, series1)
        fig.tight_layout()
        fig.savefig(args.out, dpi=200)
        print(f"[plot_training] wrote {args.out}")


if __name__ == "__main__":
    main()


