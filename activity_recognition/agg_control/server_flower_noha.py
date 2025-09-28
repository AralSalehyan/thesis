#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
server_flower_noha.py
Baseline Flower server (no High Availability).

What it does:
- Starts a Flower FedAvg server on --port
- Runs for --rounds
- After each round's aggregation, saves:
    - round_<r>_weights.npz  (compressed model weights)
    - round_<r>_meta.json    (hash, sizes, basic metrics)

CLI / Env (matches run_experiment.py):
  --port 8080
  --rounds 50
  --min_completion 0.7     (hint for strategy; not strictly enforced here)
  --t_round 600            (round timeout passed to ServerConfig)
  --seed 1
  --model m1               (only logged; the model architecture lives in clients)
  --out results/<dir>      (where logs/checkpoints go)

Install:
  pip install flwr numpy
"""

import argparse
import io
import json
import os
import time
import hashlib
from typing import Dict, Any

import numpy as np
import flwr as fl
from flwr.server import ServerConfig, start_server
from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

# ---------- small helpers ----------

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def serialize_ndarrays(weights) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, *weights)
    return buf.getvalue()

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

# ---------- custom strategy (just to log/save after aggregate) ----------

class LoggingFedAvg(FedAvg):
    def __init__(self, out_dir: str, model_name: str, **kwargs):
        super().__init__(**kwargs)
        self.out_dir = out_dir
        self.model_name = model_name
        ensure_dir(self.out_dir)

    def aggregate_fit(self, server_round: int, results, failures):
        agg = super().aggregate_fit(server_round, results, failures)
        if agg is None:
            return None

        parameters, metrics = agg

        # Save compressed weights
        weights = parameters_to_ndarrays(parameters)
        blob = serialize_ndarrays(weights)
        h = sha256_bytes(blob)

        wpath = os.path.join(self.out_dir, f"round_{server_round:04d}_weights.npz")
        with open(wpath, "wb") as f:
            f.write(blob)

        # Save small meta json
        meta = {
            "round": server_round,
            "model": self.model_name,
            "weights_npz": os.path.basename(wpath),
            "hash_sha256": h,
            "metrics": metrics if isinstance(metrics, dict) else {},
            "n_arrays": len(weights),
            "array_shapes": [list(w.shape) for w in weights],
            "timestamp": time.time(),
        }
        with open(os.path.join(self.out_dir, f"round_{server_round:04d}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[noha] round {server_round}: saved {wpath} (sha256={h[:8]}...)")
        return agg

# ---------- main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=int(os.getenv("SERVER_PORT", "8080")))
    p.add_argument("--rounds", type=int, default=int(os.getenv("ROUNDS", "50")))
    p.add_argument("--min_completion", type=float, default=float(os.getenv("MIN_COMPLETION", "0.7")))
    p.add_argument("--t_round", type=int, default=int(os.getenv("T_ROUND", "600")))
    p.add_argument("--seed", type=int, default=int(os.getenv("SEED", "1")))
    p.add_argument("--model", type=str, default=os.getenv("MODEL", "m1"))
    p.add_argument("--out", type=str, default=os.getenv("OUT_DIR", "results"))
    args = p.parse_args()

    ensure_dir(args.out)

    # Create strategy; you can tune these fractions or set min_* clients if you like
    strategy = LoggingFedAvg(
        out_dir=args.out,
        model_name=args.model,
        fraction_fit=1.0,          # cross-silo default
        fraction_evaluate=0.0,     # optional
        min_fit_clients=None,
        min_available_clients=None,
    )

    cfg = ServerConfig(num_rounds=args.rounds, round_timeout=args.t_round)

    print(f"[server_flower_noha] starting :{args.port} | rounds={args.rounds} | model={args.model}")
    start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=cfg,
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
