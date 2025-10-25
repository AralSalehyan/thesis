#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_synth_data.py
Create a tiny synthetic dataset for the fallback HAR loader:
  data/processed/subject_<ID>.npz with arrays X:[n,T,F], y:[n]
  data/splits/sites.json mapping sites (A,B,...) to subject IDs

Usage:
  python -m experiments.make_synth_data --sites 2 --subjects-per-site 2 --n 200 --T 128 --F 6
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=2)
    ap.add_argument("--subjects-per-site", type=int, default=2)
    ap.add_argument("--n", type=int, default=200, help="windows per subject")
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--F", type=int, default=6)
    ap.add_argument("--classes", type=int, default=8)
    ap.add_argument("--out-dir", type=str, default="data/processed")
    ap.add_argument("--splits", type=str, default="data/splits/sites.json")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    splits_json = Path(args.splits)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_json.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(123)

    site_names = [chr(ord('A') + i) for i in range(args.sites)]
    sites_map = {}
    subj_counter = 1
    for site in site_names:
        ids = []
        for _ in range(args.subjects_per_site):
            sid = f"S{subj_counter:02d}"
            ids.append(sid)

            X = rng.normal(size=(args.n, args.T, args.F)).astype(np.float32)
            # Add a simple class structure: first half low mean, second half high mean
            X[: args.n // 2] += 0.5
            X[args.n // 2 :] -= 0.5
            y = rng.integers(low=0, high=args.classes, size=(args.n,), dtype=np.int64)

            np.savez_compressed(out_dir / f"subject_{sid}.npz", X=X, y=y)
            subj_counter += 1
        sites_map[site] = ids

    with splits_json.open("w", encoding="utf-8") as f:
        json.dump(sites_map, f, indent=2)

    print(f"[make_synth_data] wrote NPZ to {out_dir} and splits to {splits_json}")


if __name__ == "__main__":
    main()




