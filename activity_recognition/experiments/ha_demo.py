#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ha_demo.py
Simulate RAID-Fed PREPARE/ACK/COMMIT with Redis + MinIO to visualize HA behavior.

Scenarios:
- normal: leader publishes PREPARE, replicas ACK to k, leader COMMITs
- leader_crash: leader publishes PREPARE then "crashes"; follower gets higher token,
  reconstructs from any k shards, and COMMITs

This does NOT run Flower training; it only exercises the HA control/data planes.
Bring up infra/docker-compose.demo.yml before running.

Usage examples:
  python -m experiments.ha_demo --scenario normal --rounds 3 --n 5 --k 3 --out results
  python -m experiments.ha_demo --scenario leader_crash --rounds 3 --n 5 --k 3 --out results
"""

import argparse
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import redis
from minio import Minio
from reedsolo import RSCodec


# ---------------- RS helpers (compatible with server_flower_ha) ----------------

def serialize_weights_blob(n_arrays: int = 8, shapes: List[Tuple[int, ...]] | None = None) -> bytes:
    """Create a small deterministic numpy weights blob to store in MinIO.
    This stands in for aggregated model parameters.
    """
    rng = np.random.default_rng(42)
    if shapes is None:
        shapes = [(32,), (32,), (32, 32), (32,), (64, 32), (64,), (8, 64), (8,)]
    arrs = [rng.normal(size=s).astype(np.float32) for s in shapes[:n_arrays]]
    buf = io.BytesIO()
    np.savez_compressed(buf, *arrs)
    return buf.getvalue()


def rs_encode(blob: bytes, n: int, k: int) -> List[bytes]:
    if k >= n:
        raise ValueError("k must be < n")
    pad = (-len(blob)) % k
    if pad:
        blob = blob + b"\x00" * pad
    part_len = len(blob) // k
    parts = [blob[i * part_len : (i + 1) * part_len] for i in range(k)]
    rsc = RSCodec(n - k)
    cols = list(zip(*parts))
    shards_cols: List[List[int]] = [[] for _ in range(n)]
    for col in cols:
        codeword = rsc.encode(bytes(col))
        for j, b in enumerate(codeword):
            shards_cols[j].append(b)
    return [bytes(col) for col in shards_cols]


def rs_decode_from_any_k(shards_bytes: List[Tuple[int, bytes]], n: int, k: int) -> bytes:
    if len(shards_bytes) < k:
        raise ValueError("Need at least k shards to decode")
    Ls = {len(b) for _, b in shards_bytes}
    if len(Ls) != 1:
        raise ValueError("Shard lengths mismatch")
    shard_len = Ls.pop()
    rsc = RSCodec(n - k)
    shard_map = {idx_1b - 1: b for idx_1b, b in shards_bytes}
    cols_recovered = bytearray()
    for col_i in range(shard_len):
        codeword = [None] * n
        erasures = []
        for pos in range(n):
            sb = shard_map.get(pos, None)
            if sb is None:
                codeword[pos] = 0
                erasures.append(pos)
            else:
                codeword[pos] = sb[col_i]
        data = rsc.decode(bytes(codeword), erase_pos=erasures)[0]
        cols_recovered.extend(data)
    return bytes(cols_recovered).rstrip(b"\x00")


# ---------------- MinIO / Redis helpers ----------------

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


def get_redis(url: str) -> redis.Redis:
    return redis.Redis.from_url(url, decode_responses=True)


def now() -> float:
    return time.time()


def write_event(ev_path: Path, event: Dict):
    with ev_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


# ---------------- Demo core ----------------

def run_demo(scenario: str, rounds: int, n: int, k: int, cluster_id: str,
             kv_url: str, minio_url: str, minio_access: str, minio_secret: str,
             out_root: Path) -> Path:
    out_dir = out_root / f"demo_{int(now())}_{scenario}_n{n}k{k}"
    out_dir.mkdir(parents=True, exist_ok=False)
    ev_path = out_dir / "events.jsonl"

    mc = minio_client(MinioConf(
        url=minio_url,
        access_key=minio_access,
        secret_key=minio_secret,
        secure=minio_url.startswith("https://"),
    ))
    kv = get_redis(kv_url)

    for r in range(1, rounds + 1):
        # Leader prepares shard set
        blob = serialize_weights_blob()
        shards = rs_encode(blob, n=n, k=k)
        round_key_base = f"{cluster_id}/round_{r}"
        meta = {"cluster_id": cluster_id, "round": r, "n": n, "k": k, "shard_keys": []}
        for idx, shard in enumerate(shards):
            bucket = f"raidfed-shard-{idx+1:02d}"
            key = f"{round_key_base}/shard_{idx+1:02d}.rs"
            put_bytes(mc, bucket, key, shard)
            meta["shard_keys"].append({"bucket": bucket, "key": key})
        token = kv.incr(f"fencing:{cluster_id}")
        prep_key = f"prep:{cluster_id}:{r}"
        acks_key = f"acks:{cluster_id}:{r}"
        kv.delete(acks_key)
        kv.set(prep_key, json.dumps({"round": r, "token": int(token), "meta": meta, "ts": now(), "coordinator": "leader"}))
        write_event(ev_path, {"t": now(), "type": "PREPARE", "round": r, "token": int(token)})

        # Simulate ACKs from replicas
        for idx in range(1, n + 1):
            # stagger a bit
            time.sleep(0.05)
            kv.sadd(acks_key, idx)
            write_event(ev_path, {"t": now(), "type": "ACK", "round": r, "shard_idx": idx})
            if kv.scard(acks_key) >= k:
                write_event(ev_path, {"t": now(), "type": "ACK_QUORUM", "round": r, "acks": kv.scard(acks_key)})
                break

        commit_key = f"commit:{cluster_id}:{r}"
        if scenario == "normal":
            kv.set(commit_key, json.dumps({"round": r, "token": int(token), "meta": meta, "ts": now(), "coordinator": "leader"}))
            write_event(ev_path, {"t": now(), "type": "COMMIT", "round": r, "by": "leader"})
        elif scenario == "leader_crash":
            # Leader does not commit; pause then follower takes over if quorum exists
            time.sleep(0.5)
            if int(kv.scard(acks_key)) >= k:
                f_token = kv.incr(f"fencing:{cluster_id}")
                # reconstruct using first k shards
                got: List[Tuple[int, bytes]] = []
                for idx_1b, sk in enumerate(meta["shard_keys"], start=1):
                    # pull only first k
                    if len(got) >= k:
                        break
                    obj = mc.get_object(sk["bucket"], sk["key"])  # noqa
                    try:
                        data = obj.read()
                    finally:
                        obj.close()
                    got.append((idx_1b, data))
                _ = rs_decode_from_any_k(got, n=n, k=k)
                kv.set(commit_key, json.dumps({"round": r, "token": int(f_token), "meta": meta, "ts": now(), "coordinator": "follower"}))
                write_event(ev_path, {"t": now(), "type": "TAKEOVER", "round": r, "token": int(f_token)})
                write_event(ev_path, {"t": now(), "type": "COMMIT", "round": r, "by": "follower"})
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    # Save small run meta
    (out_dir / "meta.json").write_text(json.dumps({
        "scenario": scenario, "rounds": rounds, "n": n, "k": k,
        "cluster_id": cluster_id, "kv_url": kv_url, "minio_url": minio_url,
        "time": now(),
    }, indent=2))

    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Simulate HA PREPARE/ACK/COMMIT with Redis + MinIO")
    ap.add_argument("--scenario", choices=["normal", "leader_crash"], required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--cluster-id", type=str, default="demo_cluster")
    ap.add_argument("--kv-url", type=str, default="redis://localhost:6379/0")
    ap.add_argument("--minio-url", type=str, default="http://localhost:9000")
    ap.add_argument("--minio-access", type=str, default="admin")
    ap.add_argument("--minio-secret", type=str, default="password")
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_dir = run_demo(
        scenario=args.scenario,
        rounds=args.rounds,
        n=args.n,
        k=args.k,
        cluster_id=args.cluster_id,
        kv_url=args.kv_url,
        minio_url=args.minio_url,
        minio_access=args.minio_access,
        minio_secret=args.minio_secret,
        out_root=out_root,
    )
    print(f"[ha_demo] wrote events to {out_dir}")


if __name__ == "__main__":
    main()




