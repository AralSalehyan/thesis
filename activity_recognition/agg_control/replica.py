#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replica.py
Follower replica for RAID-Fed HA with failover (leader takeover).

Responsibilities (as follower):
- Heartbeat into Redis (alive membership).
- Watch for PREPARE of the next round.
- Fetch own shard from MinIO, persist, ACK.
- Wait for COMMIT and advance.

Failure recovery (leader takeover):
- If COMMIT does not arrive within a grace window AND ACKs >= k:
    • Obtain a higher fencing token (KV.incr)
    • Reconstruct the global model from any k shards (RS decode)
    • Publish COMMIT (no re-aggregation)
    • Advance round

Env / CLI:
  --node-id agg2
  --cluster-id seniorcare_fl
  --kv-url redis://redis:6379/0
  --minio-url http://minio:9000
  --minio-access-key admin
  --minio-secret-key password
  --data-dir cache
  --hb-interval 1
  --failover-grace-s 15       # how long to wait for COMMIT after PREPARE before attempting takeover
"""

import argparse
import io
import json
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from minio import Minio
from reedsolo import RSCodec

from kv import get_kv  # our kv.py


# ---------------- MinIO helpers ----------------

def minio_client(url, access, secret, secure):
    return Minio(
        url.replace("http://", "").replace("https://", ""),
        access_key=access,
        secret_key=secret,
        secure=secure,
    )

def fetch_bytes(mc: Minio, bucket: str, key: str) -> bytes:
    obj = mc.get_object(bucket, key)
    try:
        return obj.read()
    finally:
        obj.close()


# ---------------- RS decode (mirror of simple encoder) ----------------
# NOTE: This matches the simple per-column RS we used in server_flower_ha.rs_encode.
# We expect 'n' shards; any 'k' of them reconstruct the original blob (with zero padding stripped).

def rs_decode_from_any_k(shards_bytes: List[Tuple[int, bytes]], n: int, k: int) -> bytes:
    """
    shards_bytes: list of (index_1based, shard_bytes)
    Returns: original blob (unpadded)
    """
    if len(shards_bytes) < k:
        raise ValueError("Need at least k shards to decode")

    # All shard lengths must match
    Ls = {len(b) for _, b in shards_bytes}
    if len(Ls) != 1:
        raise ValueError("Shard lengths mismatch")
    shard_len = Ls.pop()

    # Build per-column codewords, solve for the original k data bytes.
    rsc = RSCodec(n - k)

    # We need the exact columns of the codeword to decode: reedsolo.decode expects full codeword.
    # Our encoder produced length-n codewords per column: data(k bytes) + parity(n-k).
    # When we have only k columns, we reconstruct by *erasure decoding*.
    #
    # reedsolo supports erasures if we provide the positions. We'll construct an array of size n
    # and mark missing positions as '?' (any byte) and pass erasures indexes.

    # Prepare an index->bytes map (0-based index)
    shard_map = {idx_1b - 1: b for idx_1b, b in shards_bytes}  # {0..n-1: bytes}

    cols_recovered = bytearray()
    # Iterate over columns (each shard contributes one byte per column)
    for col_i in range(shard_len):
        codeword = [None] * n
        erasures = []
        for pos in range(n):
            sb = shard_map.get(pos, None)
            if sb is None:
                codeword[pos] = 0  # placeholder
                erasures.append(pos)
            else:
                codeword[pos] = sb[col_i]
        # Erasure decode this column
        data = rsc.decode(bytes(codeword), erase_pos=erasures)[0]  # returns (decoded, x)
        # data is the original k data bytes for this column
        cols_recovered.extend(data)

    # We now have k * shard_len bytes == original blob padded to multiple of k
    full = bytes(cols_recovered)

    # Strip trailing zero padding (we padded to multiple of k during encode)
    return full.rstrip(b"\x00")


# ---------------- Utility ----------------

def shard_index_for_node(node_id: str, n: int) -> int:
    """Map node_id like 'agg3' -> 0-based shard index 2. Fallback to hash if format unknown."""
    try:
        return int(node_id.replace("agg", "")) - 1
    except Exception:
        return hash(node_id) % n


# ---------------- Replica loop ----------------

def replica_loop(node_id: str, cluster_id: str, kv_url: str,
                 minio_url: str, minio_access: str, minio_secret: str,
                 data_dir: str, hb_interval: int = 1, failover_grace_s: int = 15):
    kv = get_kv(kv_url, cluster_id)
    mc = minio_client(minio_url, minio_access, minio_secret, secure=minio_url.startswith("https://"))
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    print(f"[replica {node_id}] started | cluster={cluster_id}")

    last_committed = kv.last_committed_round()

    while True:
        # 1) Heartbeat (liveness)
        kv.heartbeat(node_id)

        target_round = last_committed + 1

        # 2) Observe PREPARE for the next round
        prep = kv.get_prepare(target_round)
        if not prep:
            time.sleep(hb_interval)
            continue

        meta = prep["meta"]
        shard_keys = meta["shard_keys"]           # [{bucket, key}, ...]
        n, k = int(meta["n"]), int(meta["k"])
        r = int(prep["round"])
        coordinator = prep.get("coordinator", "?")
        prep_token = int(prep.get("token", 0))
        model_hash = meta.get("model_hash", "")

        # 3) Fetch and persist my shard, then ACK
        my_idx = shard_index_for_node(node_id, n)  # 0-based
        if 0 <= my_idx < len(shard_keys):
            bucket = shard_keys[my_idx]["bucket"]
            key = shard_keys[my_idx]["key"]
            try:
                data = fetch_bytes(mc, bucket, key)
                shard_path = Path(data_dir) / f"round_{r:04d}_shard_{my_idx+1:02d}.rs"
                shard_path.write_bytes(data)
                ack_n = kv.add_ack(r, node_id, my_idx+1)
                print(f"[replica {node_id}] PREPARE r={r} | stored shard {my_idx+1}/{n} | ACKs={ack_n}")
            except Exception as e:
                print(f"[replica {node_id}] ERROR fetching shard {my_idx+1} for r={r}: {e}")

        # 4) Wait for COMMIT or trigger failover if grace exceeded
        commit = None
        start_wait = time.time()
        while True:
            commit = kv.get_commit(r)
            if commit:
                print(f"[replica {node_id}] COMMIT r={r} by {commit.get('coordinator')} (hash={model_hash[:8]}...)")
                last_committed = r
                break

            # check grace condition for takeover
            if time.time() - start_wait >= failover_grace_s:
                # If quorum exists (ACKs >= k) and no commit, try to take over
                if kv.ack_count(r) >= k:
                    # Take a newer fencing token; only proceed if strictly greater than prep token
                    my_token = kv.next_fencing_token()
                    if my_token > prep_token:
                        print(f"[replica {node_id}] TAKEOVER r={r} | token {my_token} > {prep_token} | reconstructing...")
                        try:
                            # Reconstruct from any k shards available
                            # For simplicity, try in order and take first k that download OK
                            got: List[Tuple[int, bytes]] = []
                            for idx_1b, sk in enumerate(shard_keys, start=1):
                                try:
                                    b = fetch_bytes(mc, sk["bucket"], sk["key"])
                                    got.append((idx_1b, b))
                                except Exception:
                                    pass
                                if len(got) >= k:
                                    break
                            if len(got) < k:
                                print(f"[replica {node_id}] TAKEOVER FAILED: only {len(got)}/{k} shards readable")
                            else:
                                blob = rs_decode_from_any_k(got, n=n, k=k)
                                # Persist reconstructed blob for next round bootstrap/debug
                                recon_path = Path(data_dir) / f"round_{r:04d}_reconstructed.npz"
                                recon_path.write_bytes(blob)
                                # Publish COMMIT
                                kv.set_commit(round_id=r, coordinator=node_id, token=my_token, meta=meta)
                                print(f"[replica {node_id}] TAKEOVER SUCCESS r={r} | COMMIT published | saved {recon_path.name}")
                                last_committed = r
                                break
                        except Exception as e:
                            print(f"[replica {node_id}] TAKEOVER ERROR r={r}: {e}")
                    else:
                        # Someone else likely took over or coordinator recovered
                        pass
                # Reset grace timer to avoid busy loop; continue waiting a bit longer
                start_wait = time.time()

            time.sleep(0.5)

        # Small pause before next loop
        time.sleep(hb_interval)


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-id", type=str, required=True)
    ap.add_argument("--cluster-id", type=str, default=os.getenv("CLUSTER_ID", "seniorcare_fl"))
    ap.add_argument("--kv-url", type=str, default=os.getenv("KV_URL", "redis://localhost:6379/0"))
    ap.add_argument("--minio-url", type=str, default=os.getenv("MINIO_URL", "http://localhost:9000"))
    ap.add_argument("--minio-access-key", type=str, default=os.getenv("MINIO_ACCESS_KEY", "admin"))
    ap.add_argument("--minio-secret-key", type=str, default=os.getenv("MINIO_SECRET_KEY", "password"))
    ap.add_argument("--data-dir", type=str, default=os.getenv("DATA_DIR", "cache"))
    ap.add_argument("--hb-interval", type=int, default=int(os.getenv("HB_INTERVAL", "1")))
    ap.add_argument("--failover-grace-s", type=int, default=int(os.getenv("FAILOVER_GRACE_S", "15")))
    args = ap.parse_args()

    replica_loop(
        node_id=args.node_id,
        cluster_id=args.cluster_id,
        kv_url=args.kv_url,
        minio_url=args.minio_url,
        minio_access=args.minio_access_key,
        minio_secret=args.minio_secret_key,
        data_dir=args.data_dir,
        hb_interval=args.hb_interval,
        failover_grace_s=args.failover_grace_s,
    )


if __name__ == "__main__":
    main()
