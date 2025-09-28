#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a federated HAR70+ experiment with/without High Availability (RAID-Fed).

Examples:
  python experiments/run_experiment.py --model m1 --ha off --rounds 50 --sites 5 --out results
  python experiments/run_experiment.py --model m2 --ha on  --failures light --rounds 50 --sites 5 --out results

This script:
  - Loads configs/base.yaml and configs/models/<model>.yaml
  - Creates a run directory with merged metadata
  - If ha=off: launches a single Flower server + N site clients as subprocesses
  - If ha=on: launches infra/docker-compose.ha.yml (KV + MinIO + N aggregator replicas + N site clients)
  - Optionally runs a failure injector when failures!=none
"""

import argparse
import os
import sys
import json
import yaml
import signal
import string
import pathlib
import subprocess
from datetime import datetime
from typing import Dict, Any, List

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
CONFIGS = REPO / "configs"
INFRA = REPO / "infra"

# -------------------------
# Helpers
# -------------------------

def load_yaml(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Merge dict b into a recursively (a is not modified)."""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def ensure_compose_cmd() -> List[str]:
    """Return a working docker compose command (docker compose or docker-compose)."""
    try:
        subprocess.run(["docker", "compose", "version"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return ["docker", "compose"]
    except Exception:
        pass
    try:
        subprocess.run(["docker-compose", "version"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return ["docker-compose"]
    except Exception:
        raise RuntimeError("Neither 'docker compose' nor 'docker-compose' is available on PATH.")

def ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")

def site_names(n: int) -> List[str]:
    letters = list(string.ascii_uppercase)
    if n <= len(letters):
        return [letters[i] for i in range(n)]
    return [f"S{i+1}" for i in range(n)]

def make_run_dir(out_root: pathlib.Path, model: str, ha: str, failures: str, seed: int, rounds: int, sites: int) -> pathlib.Path:
    name = f"{ts()}_model-{model}_ha-{ha}_fail-{failures}_seed{seed}_r{rounds}_s{sites}"
    run_dir = out_root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def write_json(path: pathlib.Path, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def terminate_tree(proc: subprocess.Popen, gentle_seconds: float = 5.0):
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    import time
    t0 = time.time()
    while proc.poll() is None and (time.time() - t0) < gentle_seconds:
        time.sleep(0.2)
    if proc.poll() is None:
        try:
            if os.name == "nt":
                proc.kill()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()

# -------------------------
# No-HA mode (single server)
# -------------------------

def run_noha(args, merged_cfg: Dict[str, Any], run_dir: pathlib.Path) -> int:
    """
    Launch a single Flower server and N site clients as local subprocesses.
    Assumes:
      - agg_control/server_flower_noha.py exists and accepts the CLI args below
      - site_client/client_flower.py exists and accepts --site/--server args
    """
    base_env = os.environ.copy()
    base_env["PYTHONUNBUFFERED"] = "1"
    base_env["MODEL"] = args.model
    base_env["ROUNDS"] = str(args.rounds)
    base_env["SITES"] = str(args.sites)
    base_env["SEED"] = str(args.seed)
    base_env["MIN_COMPLETION"] = str(args.min_completion)
    base_env["T_ROUND"] = str(args.t_round)
    base_env["OUT_DIR"] = str(run_dir)

    server_port = int(merged_cfg.get("server_port", 8080))
    server_addr = f"127.0.0.1:{server_port}"

    # Start server
    server_cmd = [
        sys.executable, "-m", "agg_control.server_flower_noha",
        "--model", args.model,
        "--rounds", str(args.rounds),
        "--min_completion", str(args.min_completion),
        "--t_round", str(args.t_round),
        "--seed", str(args.seed),
        "--out", str(run_dir),
        "--port", str(server_port),
    ]
    server_log = run_dir / "server.log"
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=open(server_log, "ab"),
        stderr=subprocess.STDOUT,
        cwd=str(REPO),
        preexec_fn=os.setsid if os.name != "nt" else None
    )

    # Start clients
    clients = []
    for s in site_names(args.sites):
        client_cmd = [
            sys.executable, "-m", "site_client.client_flower",
            "--site", s,
            "--server", server_addr,
            "--model", args.model,
            "--seed", str(args.seed),
        ]
        log_path = run_dir / f"client_{s}.log"
        p = subprocess.Popen(
            client_cmd,
            stdout=open(log_path, "ab"),
            stderr=subprocess.STDOUT,
            cwd=str(REPO),
            preexec_fn=os.setsid if os.name != "nt" else None
        )
        clients.append(p)

    # Wait for server to finish
    rc = server_proc.wait()

    # Stop clients once server finishes
    for p in clients:
        terminate_tree(p)

    return rc if rc is not None else 1

# -------------------------
# HA mode (docker compose)
# -------------------------

def run_ha(args, merged_cfg: Dict[str, Any], run_dir: pathlib.Path) -> int:
    """
    Launch infra/docker-compose.ha.yml (KV + MinIO + N aggregator replicas + N site clients).
    Expects your compose stack to exit when training is complete (e.g., coordinator exits).
    """
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1",
        "MODEL": args.model,
        "ROUNDS": str(args.rounds),
        "SITES": str(args.sites),
        "SEED": str(args.seed),
        "MIN_COMPLETION": str(args.min_completion),
        "T_ROUND": str(args.t_round),
        "N": str(merged_cfg.get("n", 5)),
        "K": str(merged_cfg.get("k", 3)),
        "OUT_DIR": str(run_dir),
        "FAILURES": args.failures,  # optional: your compose/services can read this
    })

    compose = ensure_compose_cmd()
    compose_file = str((INFRA / "docker-compose.ha.yml").resolve())

    # Optional: launch failure injector if provided (ignored if missing)
    injector_proc = None
    if args.failures != "none":
        inj_module = REPO / "experiments" / "failure_injector.py"
        if inj_module.exists():
            inj_cmd = [
                sys.executable, "-m", "experiments.failure_injector",
                "--regime", args.failures,
                "--sites", str(args.sites),
                "--rounds", str(args.rounds),
            ]
            inj_log = run_dir / "failure_injector.log"
            injector_proc = subprocess.Popen(
                inj_cmd,
                stdout=open(inj_log, "ab"),
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(REPO),
                preexec_fn=os.setsid if os.name != "nt" else None
            )

    # Run compose (foreground)
    up_cmd = compose + ["-f", compose_file, "up", "--build", "--abort-on-container-exit"]
    comp_log = run_dir / "compose.log"
    rc = 1
    with open(comp_log, "ab") as lf:
        proc = subprocess.Popen(
            up_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(REPO),
            preexec_fn=os.setsid if os.name != "nt" else None
        )
        for line in iter(proc.stdout.readline, b""):
            lf.write(line)
            lf.flush()
        proc.wait()
        rc = proc.returncode

    # Ensure injector stops
    if injector_proc:
        terminate_tree(injector_proc)

    # Tear down the stack
    down_cmd = compose + ["-f", compose_file, "down", "-v"]
    try:
        subprocess.run(down_cmd, check=False, env=env, cwd=str(REPO))
    except Exception:
        pass

    return rc

# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Run FL experiment with/without High Availability (RAID-Fed).")
    parser.add_argument("--model", choices=["m1", "m2", "m3"], required=True)
    parser.add_argument("--ha", choices=["on", "off"], required=True)
    parser.add_argument("--failures", choices=["none", "light", "heavy"], default="none",
                        help="Ignored when ha=off")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--sites", type=int, default=5)
    parser.add_argument("--min_completion", type=float, default=0.7)
    parser.add_argument("--t_round", type=int, default=600, help="Max round time (seconds)")
    parser.add_argument("--out", type=str, default="results", help="Output directory root")
    args = parser.parse_args()

    # Load configs
    base = load_yaml(CONFIGS / "base.yaml")
    model_cfg = load_yaml(CONFIGS / "models" / f"{args.model}.yaml")
    merged_cfg = deep_merge(base, model_cfg)

    # Create run dir
    out_root = (REPO / args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(out_root, args.model, args.ha, args.failures, args.seed, args.rounds, args.sites)

    # Save metadata
    meta = {
        "args": vars(args),
        "base_cfg": base,
        "model_cfg": model_cfg,
        "merged_cfg": merged_cfg,
        "time_utc": datetime.utcnow().isoformat() + "Z",
    }
    write_json(run_dir / "metadata.json", meta)
    print(f"[run_experiment] Run directory: {run_dir}")

    # Branch by HA
    if args.ha == "off":
        rc = run_noha(args, merged_cfg, run_dir)
    else:
        rc = run_ha(args, merged_cfg, run_dir)

    print(f"[run_experiment] Exit code: {rc}")
    sys.exit(rc)

if __name__ == "__main__":
    main()