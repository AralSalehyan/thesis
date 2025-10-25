#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize.py
Visualize HA demo timelines (PREPARE/ACK/COMMIT) and compare scenarios.

Usage examples:
  # Plot a single demo timeline
  python -m experiments.summarize --events results/demo_xxx_normal_n5k3 --out results/normal_timeline.png

  # Side-by-side comparison of normal vs leader_crash
  python -m experiments.summarize \
    --events results/demo_xxx_normal_n5k3 \
    --events2 results/demo_yyy_leader_crash_n5k3 \
    --out results/compare_timeline.png

Input format:
  events.jsonl produced by experiments/ha_demo.py with records like:
    {"t": <float>, "type": "PREPARE"|"ACK"|"ACK_QUORUM"|"TAKEOVER"|"COMMIT", "round": int, ...}
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_events_dir(events_dir: Path) -> List[Dict]:
    events_path = events_dir / "events.jsonl"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing events.jsonl in {events_dir}")
    events: List[Dict] = []
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def group_by_round(events: List[Dict]) -> Dict[int, List[Dict]]:
    rounds: Dict[int, List[Dict]] = {}
    for ev in events:
        r = int(ev.get("round", -1))
        if r < 0:
            continue
        rounds.setdefault(r, []).append(ev)
    # sort by time
    for r in rounds:
        rounds[r].sort(key=lambda e: e.get("t", 0.0))
    return rounds


def normalize_times(round_events: Dict[int, List[Dict]]) -> float:
    t0 = min(ev["t"] for evs in round_events.values() for ev in evs)
    for evs in round_events.values():
        for ev in evs:
            ev["t_rel"] = ev["t"] - t0
    return t0


def plot_timeline(ax, round_events: Dict[int, List[Dict]], title: str):
    # Map event types to markers/colors
    style = {
        "PREPARE": ("o", "tab:blue"),
        "ACK": (".", "tab:gray"),
        "ACK_QUORUM": ("^", "tab:orange"),
        "TAKEOVER": ("s", "tab:red"),
        "COMMIT": ("*", "tab:green"),
    }
    ys = sorted(round_events.keys())
    for y in ys:
        evs = round_events[y]
        for ev in evs:
            marker, color = style.get(ev.get("type", "ACK"), (".", "black"))
            ax.scatter(ev["t_rel"], y, marker=marker, color=color, s=60, zorder=3)
    ax.set_title(title)
    ax.set_xlabel("time (s, relative)")
    ax.set_ylabel("round")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    ax.set_yticks(ys)

    # Legend
    handles = []
    labels = []
    for k, (m, c) in style.items():
        h = ax.scatter([], [], marker=m, color=c, s=60)
        handles.append(h)
        labels.append(k)
    ax.legend(handles, labels, loc="best", frameon=True)


def main():
    ap = argparse.ArgumentParser(description="Visualize HA demo events")
    ap.add_argument("--events", type=str, required=True, help="Path to events directory (contains events.jsonl)")
    ap.add_argument("--events2", type=str, default=None, help="Optional second events directory for side-by-side")
    ap.add_argument("--out", type=str, required=True, help="Output PNG path")
    args = ap.parse_args()

    ev_dir1 = Path(args.events)
    evs1 = load_events_dir(ev_dir1)
    by_round1 = group_by_round(evs1)
    normalize_times(by_round1)

    if args.events2:
        ev_dir2 = Path(args.events2)
        evs2 = load_events_dir(ev_dir2)
        by_round2 = group_by_round(evs2)
        normalize_times(by_round2)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        plot_timeline(axes[0], by_round1, title=ev_dir1.name)
        plot_timeline(axes[1], by_round2, title=ev_dir2.name)
        fig.tight_layout()
        out = Path(args.out)
        fig.savefig(out, dpi=200)
        print(f"[summarize] wrote {out}")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        plot_timeline(ax, by_round1, title=ev_dir1.name)
        fig.tight_layout()
        out = Path(args.out)
        fig.savefig(out, dpi=200)
        print(f"[summarize] wrote {out}")


if __name__ == "__main__":
    main()

