# Activity Recognition Federated Learning (HAR70+)

Federated learning experiments for HAR70+ with optional High Availability (RAID-Fed). Run locally (no-HA) or with HA via Docker Compose (Redis + MinIO + replicas).

## Quickstart (no HA)

Prereqs: Python 3.12, pip.

```
pip install -r requirements.txt
python experiments/run_experiment.py --model m1 --ha off --rounds 5 --sites 5 --out results
```

This launches a Flower server and N site clients as subprocesses. Logs go into the created run directory under `results/`.

## Quickstart (HA)

Prereqs: Docker + Docker Compose.

```
pip install -r requirements.txt  # only needed if running any local tools
python experiments/run_experiment.py --model m1 --ha on --failures light --rounds 5 --sites 5 --out results
```

This brings up Redis, MinIO, 5 aggregator replicas and 5 site clients. The coordinator writes RAID-Fed shards to MinIO each round and publishes PREPARE/COMMIT to Redis. If the coordinator fails, followers can reconstruct from any k shards and commit.

- MinIO console: http://localhost:9001 (admin/password)
- MinIO S3 API: http://localhost:9000
- Flower ingress: http://localhost:8080 (from `agg1`)

## Configuration

- Base config: `configs/base.yaml`
- Model configs: `configs/models/m{1,2,3}.yaml`
- Orchestrator merges them and records to `results/<run>/metadata.json`.

## Data

Clients expect site-specific NPZ windows via `site_client/datasets.py` or the fallback loader described in `site_client/client_flower.py` (see header). If you have raw CSVs, preprocess them with your pipeline (see `har70_preprocess/`) into NPZ with arrays `X` and `y`.

## Scripts

- `experiments/run_experiment.py`: Orchestrator (HA/no-HA)
- `experiments/summarize.py`: Summaries (TBD)
- `experiments/failure_injector.py`: Optional failure regimes (TBD)

## Containers

- `Dockerfile` builds a CPU image.
- `.dockerignore` reduces context.
- `infra/docker-compose.noha.yml` and `infra/docker-compose.ha.yml` start the stacks.

## Notes

- Windows users: no-HA path runs natively; HA path requires Docker Desktop.
- PyTorch CPU is used by default; set `--device cuda` in clients if you provide GPUs.


