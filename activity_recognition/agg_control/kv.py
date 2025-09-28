#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kv.py
Tiny Redis-backed coordination layer for RAID-Fed HA.

Provides:
- Fencing tokens (monotonic counter) to prevent split-brain
- PREPARE / COMMIT records per round
- ACK set for PREPARE quorum (≥k)
- Coordinator announce per round
- Lightweight heartbeats with TTL (to build alive membership)
- Small convenience helpers with sane key naming

Expected Redis URL: redis://host:port/db
Requires: pip install redis
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import redis


# -----------------------------
# Key schema
# -----------------------------
# Fencing counter:         fencing:{cluster}
# Coordinator announce:    coord:{cluster}:{round}               (JSON)
# PREPARE record:          prep:{cluster}:{round}                (JSON)
# PREPARE ACK set:         acks:{cluster}:{round}                (SET of "node_id:shard_idx" or just "shard_idx")
# COMMIT record:           commit:{cluster}:{round}              (JSON)
# Heartbeat per node:      hb:{cluster}:{node_id}                (value: ts or json; uses TTL)
# Last committed round:    last_commit_round:{cluster}           (INT)
# Any extra per-round map: state:{cluster}:{round}:{field}       (optional utility)


@dataclass
class KVConfig:
    url: str                  # e.g., "redis://localhost:6379/0"
    cluster_id: str           # shared cluster namespace
    hb_ttl_s: int = 5         # heartbeat TTL (seconds)
    prefix: str = ""          # optional global prefix for multi-tenancy (e.g., "prod:")


class KV:
    """Small convenience wrapper around Redis for our HA control-plane."""

    def __init__(self, cfg: KVConfig):
        self.cfg = cfg
        self.r = redis.Redis.from_url(cfg.url, decode_responses=True)

    # --------- key helpers ---------
    def _k(self, kind: str, *ids: Any) -> str:
        """Compose a namespaced key."""
        base = f"{kind}:{self.cfg.cluster_id}"
        if ids:
            base += ":" + ":".join(str(x) for x in ids)
        return f"{self.cfg.prefix}{base}"

    # --------- fencing token ---------
    def next_fencing_token(self) -> int:
        """Atomically increment and return fencing token for the cluster."""
        return int(self.r.incr(self._k("fencing")))

    def current_fencing_token(self) -> int:
        v = self.r.get(self._k("fencing"))
        return int(v) if v is not None else 0

    # --------- coordinator announce ---------
    def announce_coordinator(self, round_id: int, node_id: str, token: int, extra: Dict[str, Any] | None = None) -> None:
        rec = {"round": round_id, "node_id": node_id, "token": token, "ts": time.time()}
        if extra: rec["extra"] = extra
        self.r.set(self._k("coord", round_id), json.dumps(rec))

    def get_coordinator(self, round_id: int) -> Optional[Dict[str, Any]]:
        raw = self.r.get(self._k("coord", round_id))
        return json.loads(raw) if raw else None

    # --------- PREPARE / ACK / COMMIT ---------
    def set_prepare(self, round_id: int, coordinator: str, token: int, meta: Dict[str, Any]) -> None:
        """Publish PREPARE with shard metadata (buckets/keys, model_hash, n,k)."""
        rec = {
            "round": round_id,
            "coordinator": coordinator,
            "token": token,
            "meta": meta,
            "ts": time.time(),
        }
        # Reset ACK set for this round
        self.r.delete(self._k("acks", round_id))
        self.r.set(self._k("prep", round_id), json.dumps(rec))

    def get_prepare(self, round_id: int) -> Optional[Dict[str, Any]]:
        raw = self.r.get(self._k("prep", round_id))
        return json.loads(raw) if raw else None

    def add_ack(self, round_id: int, node_id: str, shard_index: int) -> int:
        """
        Add an ACK to the set. Returns current cardinality.
        - You can store either just the shard index or "node_id:idx".
        - Using index-only matches a design where shard i belongs to node i.
        """
        # Store as shard_index to match coordinator's k-of-n threshold
        self.r.sadd(self._k("acks", round_id), int(shard_index))
        return int(self.r.scard(self._k("acks", round_id)))

    def ack_count(self, round_id: int) -> int:
        return int(self.r.scard(self._k("acks", round_id)))

    def set_commit(self, round_id: int, coordinator: str, token: int, meta: Dict[str, Any]) -> None:
        rec = {
            "round": round_id,
            "coordinator": coordinator,
            "token": token,
            "meta": meta,
            "ts": time.time(),
        }
        p = self.r.pipeline()
        p.set(self._k("commit", round_id), json.dumps(rec))
        p.set(self._k("last_commit_round"), int(round_id))
        p.execute()

    def get_commit(self, round_id: int) -> Optional[Dict[str, Any]]:
        raw = self.r.get(self._k("commit", round_id))
        return json.loads(raw) if raw else None

    def last_committed_round(self) -> int:
        v = self.r.get(self._k("last_commit_round"))
        return int(v) if v is not None else -1

    # --------- Heartbeats / membership ---------
    def heartbeat(self, node_id: str, info: Optional[Dict[str, Any]] = None) -> None:
        """Set a TTL'd heartbeat key. Use a short interval (e.g., 1s) and hb_ttl_s≈3–5s."""
        value = {"ts": time.time(), "node_id": node_id}
        if info: value["info"] = info
        # Redis SETEX: set value with expiry
        self.r.setex(self._k("hb", node_id), self.cfg.hb_ttl_s, json.dumps(value))

    def alive_nodes(self) -> List[str]:
        """Return node_ids that currently have a heartbeat key."""
        # SCAN keys; avoid KEYS in production (but dataset sizes are tiny here)
        pattern = self._k("hb", "*")
        node_ids: List[str] = []
        cursor = 0
        while True:
            cursor, keys = self.r.scan(cursor=cursor, match=pattern, count=100)
            for k in keys:
                # k format: hb:{cluster}:{node_id}
                parts = k.split(":")
                node_ids.append(parts[-1])
            if cursor == 0:
                break
        return sorted(set(node_ids))

    # --------- Utility / cleanup ---------
    def clear_round(self, round_id: int) -> None:
        """Delete prep/ack/commit for a given round (rarely needed; mainly for tests)."""
        p = self.r.pipeline()
        p.delete(self._k("prep", round_id))
        p.delete(self._k("acks", round_id))
        p.delete(self._k("commit", round_id))
        p.delete(self._k("coord", round_id))
        p.execute()

    def reset_cluster(self) -> None:
        """Dangerous: wipe cluster-scoped keys (for test resets)."""
        pats = [
            self._k("prep", "*"),
            self._k("acks", "*"),
            self._k("commit", "*"),
            self._k("coord", "*"),
            self._k("hb", "*"),
            self._k("state", "*"),
            self._k("last_commit_round"),
            self._k("fencing"),
        ]
        for pat in pats:
            cursor = 0
            while True:
                cursor, keys = self.r.scan(cursor=cursor, match=pat, count=200)
                if keys:
                    self.r.delete(*keys)
                if cursor == 0:
                    break

    # --------- small helpers for watchers (followers) ---------
    def wait_for_prepare(self, round_id: int, timeout_s: int = 600) -> Optional[Dict[str, Any]]:
        """Poll for PREPARE presence up to timeout_s."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            rec = self.get_prepare(round_id)
            if rec:
                return rec
            time.sleep(0.2)
        return None

    def wait_for_commit(self, round_id: int, timeout_s: int = 600) -> Optional[Dict[str, Any]]:
        """Poll for COMMIT presence up to timeout_s."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            rec = self.get_commit(round_id)
            if rec:
                return rec
            time.sleep(0.2)
        return None


# -----------------------------
# Convenience factory
# -----------------------------
def get_kv(url: str, cluster_id: str, hb_ttl_s: int = 5, prefix: str = "") -> KV:
    """One-liner to construct KV from simple args."""
    return KV(KVConfig(url=url, cluster_id=cluster_id, hb_ttl_s=hb_ttl_s, prefix=prefix))


# -----------------------------
# Example (manual test)
# -----------------------------
if __name__ == "__main__":
    kv = get_kv("redis://localhost:6379/0", "demo_cluster")
    kv.reset_cluster()

    # fencing
    t1 = kv.next_fencing_token()
    t2 = kv.next_fencing_token()
    print("fencing tokens:", t1, t2)

    # prepare/ack/commit
    kv.set_prepare(round_id=1, coordinator="agg1", token=t2, meta={"n": 5, "k": 3})
    print("prepare:", kv.get_prepare(1))
    kv.add_ack(1, "agg2", 1)
    kv.add_ack(1, "agg3", 2)
    print("acks so far:", kv.ack_count(1))
    kv.set_commit(round_id=1, coordinator="agg1", token=t2, meta={"n": 5, "k": 3})
    print("commit:", kv.get_commit(1), "last:", kv.last_committed_round())

    # heartbeats
    kv.heartbeat("agg1")
    kv.heartbeat("agg2")
    print("alive:", kv.alive_nodes())
