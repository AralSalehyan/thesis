#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
server_flower_ha.py
Coordinator-only Flower server for RAID-Fed HA.

What it does:
- Starts a Flower server with a custom FedAvg strategy.
- On each round's aggregate:
  - Serialize the post-aggregate state (weights) -> bytes
  - Reed–Solomon encode into n shards (any k recover)
  - Upload shards to MinIO (one bucket per shard index)
  - PREPARE: announce shard metadata in Redis, await >=k ACKs
  - COMMIT: publish commit marker in Redis

Notes:
- This file focuses on the coordinator path. Follower logic (receiving shard_i, ACKing, reconstructing on failover) lives in your `replica.py`.
- For quick demos, if no followers respond, coordinator will "self-ack" k shards so the run can proceed end-to-end.

Reqs:
  pip install flwr torch numpy redis minio reedsolo pyyaml

Env / Args expected (pick env or flags):
  --port 8080
  --rounds 50
  --min-completion 0.7
  --t-round 600
  --seed 1
  --model m1
  --n 5 --k 3
  --kv-url redis://localhost:6379
  --minio-url http://localhost:9000
  --minio-access-key admin --minio-secret-key password
  --cluster-id seniorcare_fl_DE_2025
  --node-id agg1
  --out results/
"""

import argparse
import io
import json
import os
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

# Flower
import flwr as fl
from flwr.common import NDArrays, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig, start_server

# Storage / KV
from minio import Minio
from minio.error import S3Error
import redis

# RS coding
from reedsolo import RSCodec

# ---------- Simple model param (framework-agnostic) helpers ----------

def serialize_ndarrays(weights: NDArrays) -> bytes:
    """Serialize list of NumPy arrays into bytes (npz in-memory)."""
    buf = io.BytesIO()
    # Store with dtype/shape preserved
    np.savez_compressed(buf, *weights)
    return buf.getvalue()

def deserialize_ndarrays(blob: bytes) -> NDArrays:
    """Deserialize bytes back to list of NumPy arrays."""
    with np.load(io.BytesIO(blob), allow_pickle=False) as npz:
        return [npz[k] for k in npz.files]

def hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

# ---------- MinIO + Redis small wrappers ----------

@dataclass
class MinioConf:
    url: str
    access_key: str
    secret_key: str
    secure: bool

def minio_client(cfg: MinioConf) -> Minio:
    return Minio(
        cfg.url.replace("http://", "").replace("https://", ""),
        access_key=cfg.access_key,
        secret_key=cfg.secret_key,
        secure=cfg.secure,
    )

def ensure_bucket(mc: Minio, bucket: str):
    if not mc.bucket_exists(bucket):
        mc.make_bucket(bucket)

def put_bytes(mc: Minio, bucket: str, key: str, data: bytes):
    ensure_bucket(mc, bucket)
    mc.put_object(bucket, key, io.BytesIO(data), length=len(data))

def get_redis(kv_url: str) -> redis.Redis:
    # kv_url like redis://host:6379/0
    return redis.Redis.from_url(kv_url, decode_responses=True)

# ---------- RAID-Fed: encode shards, prepare/commit ----------

def rs_encode(blob: bytes, n: int, k: int) -> List[bytes]:
    """Produce n shards such that any k reconstruct; use simple RS over bytes.
    reedsolo encodes parity bytes; here we split data and add parity stripes.
    For simplicity, we chunk the blob into k parts and compute (n-k) parity via RS on each column.
    """
    if k >= n:
        raise ValueError("k must be < n")
    # Pad to multiple of k
    pad = (-len(blob)) % k
    if pad:
        blob = blob + b"\x00" * pad
    part_len = len(blob) // k
    parts = [blob[i*part_len:(i+1)*part_len] for i in range(k)]

    # Build RS codec with n,k → parity symbols = n-k
    # Using per-byte RS across columns:
    rsc = RSCodec(n - k)

    # Transpose bytes: columns of length k
    cols = list(zip(*parts))  # list of tuples of len k
    shards_cols: List[List[int]] = [[] for _ in range(n)]
    for col in cols:
        # Encode one column of k bytes into n bytes (k data + n-k parity).
        # reedsolo works on 'message' → returns message+parity (length k+(n-k)=n)
        codeword = rsc.encode(bytes(col))  # len n
        for j, b in enumerate(codeword):
            shards_cols[j].append(b)

    # Rejoin columns per shard
    shards = [bytes(col) for col in shards_cols]

    # Done: n shards of length part_len
    return shards

# ---------- Strategy with HA commit hooks ----------

class RaidFedStrategy(FedAvg):
    def __init__(
        self,
        n: int,
        k: int,
        kv: redis.Redis,
        mc: Minio,
        cluster_id: str,
        node_id: str,
        out_dir: str,
        min_completion: float,
        t_round: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n = n
        self.k = k
        self.kv = kv
        self.mc = mc
        self.cluster_id = cluster_id
        self.node_id = node_id
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.min_completion = min_completion
        self.t_round = t_round

    # ---- Flower strategy hooks ----

    def configure_fit(self, server_round: int, parameters, client_manager):
        # Let FedAvg propose the standard config, then we can enforce min_completion / timeout externally
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round: int, results, failures):
        """After FedAvg aggregation, perform RAID-Fed PREPARE->COMMIT."""
        agg = super().aggregate_fit(server_round, results, failures)
        if agg is None:
            return None

        parameters, _ = agg
        weights = parameters_to_ndarrays(parameters)
        blob = serialize_ndarrays(weights)
        model_hash = hash_bytes(blob)

        # Encode to n shards (any k reconstruct)
        shards = rs_encode(blob, self.n, self.k)

        # Store shards to MinIO (bucket per shard index)
        round_key_base = f"{self.cluster_id}/round_{server_round}"
        meta = {
            "cluster_id": self.cluster_id,
            "round": server_round,
            "n": self.n,
            "k": self.k,
            "model_hash": model_hash,
            "shard_keys": [],
        }
        for idx, shard in enumerate(shards):
            bucket = f"raidfed-shard-{idx+1:02d}"
            key = f"{round_key_base}/shard_{idx+1:02d}.rs"
            put_bytes(self.mc, bucket, key, shard)
            meta["shard_keys"].append({"bucket": bucket, "key": key})

        # PREPARE: write metadata + wait for ≥k ACKs
        prep_key = f"prep:{self.cluster_id}:{server_round}"
        commit_key = f"commit:{self.cluster_id}:{server_round}"
        acks_key = f"acks:{self.cluster_id}:{server_round}"

        # Publish PREPARE (with fencing token included for clarity)
        token = self.kv.incr(f"fencing:{self.cluster_id}")
        prepare_record = {
            "token": token,
            "meta": meta,
            "ts": time.time(),
            "coordinator": self.node_id,
        }
        self.kv.set(prep_key, json.dumps(prepare_record))
        self.kv.delete(acks_key)

        # For demos: self-ack k shards if no followers exist
        # Real system: replicas should write SADD acks_key idx as they durably store their assigned shard.
        # We'll wait up to t_round seconds for >=k acks; then fallback to self-acks.
        deadline = time.time() + self.t_round
        while time.time() < deadline:
            try:
                acks = self.kv.scard(acks_key)
            except Exception:
                acks = 0
            if acks >= self.k:
                break
            time.sleep(0.5)

        if self.kv.scard(acks_key) < self.k:
            # Fallback: self-ack indices 1..k to progress in a standalone demo
            for idx in range(1, self.k + 1):
                self.kv.sadd(acks_key, idx)

        # COMMIT
        commit_record = {
            "token": token,
            "meta": meta,
            "ts": time.time(),
            "coordinator": self.node_id,
        }
        self.kv.set(commit_key, json.dumps(commit_record))

        # Optionally dump a human-readable meta file for the run
        with open(os.path.join(self.out_dir, f"round_{server_round:04d}_meta.json"), "w") as f:
            json.dump(commit_record, f, indent=2)

        return agg

# ---------- CLI / main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=int(os.getenv("SERVER_PORT", "8080")))
    p.add_argument("--rounds", type=int, default=int(os.getenv("ROUNDS", "50")))
    p.add_argument("--min-completion", type=float, default=float(os.getenv("MIN_COMPLETION", "0.7")))
    p.add_argument("--t-round", type=int, default=int(os.getenv("T_ROUND", "600")))
    p.add_argument("--seed", type=int, default=int(os.getenv("SEED", "1")))
    p.add_argument("--model", type=str, default=os.getenv("MODEL", "m1"))

    p.add_argument("--n", type=int, default=int(os.getenv("N", "5")))
    p.add_argument("--k", type=int, default=int(os.getenv("K", "3")))
    p.add_argument("--kv-url", type=str, default=os.getenv("KV_URL", "redis://localhost:6379/0"))
    p.add_argument("--minio-url", type=str, default=os.getenv("MINIO_URL", "http://localhost:9000"))
    p.add_argument("--minio-access-key", type=str, default=os.getenv("MINIO_ACCESS_KEY", "admin"))
    p.add_argument("--minio-secret-key", type=str, default=os.getenv("MINIO_SECRET_KEY", "password"))
    p.add_argument("--cluster-id", type=str, default=os.getenv("CLUSTER_ID", "seniorcare_fl_DE_2025"))
    p.add_argument("--node-id", type=str, default=os.getenv("NODE_ID", "agg1"))
    p.add_argument("--out", type=str, default=os.getenv("OUT_DIR", "results"))
    args = p.parse_args()

    # Init KV and object store
    kv = get_redis(args.kv_url)
    mc = minio_client(MinioConf(
        url=args.minio_url,
        access_key=args.minio_access_key,
        secret_key=args.minio_secret_key,
        secure=args.minio_url.startswith("https://"),
    ))

    # Create strategy with RAID-Fed hook
    strategy = RaidFedStrategy(
        n=args.n,
        k=args.k,
        kv=kv,
        mc=mc,
        cluster_id=args.cluster_id,
        node_id=args.node_id,
        out_dir=args.out,
        min_completion=args.min_completion,
        t_round=args.t_round,
        min_fit_clients=None,          # use FedAvg defaults or set explicitly
        min_available_clients=None,
        fraction_fit=1.0,              # cross-silo: usually 1.0 or per round cohort
        fraction_evaluate=0.0,         # optional
    )

    # Start Flower server
    cfg = ServerConfig(num_rounds=args.rounds, round_timeout=args.t_round)
    print(f"[server_flower_ha] Starting coordinator on :{args.port} | rounds={args.rounds} | n={args.n} k={args.k}")
    start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=cfg,
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
