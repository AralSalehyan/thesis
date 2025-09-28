#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
site_client/client_flower.py
Minimal Flower client for HAR70+ windows (cross-silo site).

What it does
------------
- Loads site-specific windows/labels (X: [N, T, F], y: [N]).
- Trains a PyTorch model locally for --local-epochs each FL round.
- Reports weights and simple metrics back to the server.

Data layout (fallback loader; if datasets.py not found)
-------------------------------------------------------
data/
  processed/
    subject_S01.npz  # X:[n, T, F], y:[n]
    subject_S02.npz
    ...
  splits/
    sites.json       # {"A": ["S01","S02",...], "B":[...], ...}

Args / Env (matches your orchestration)
---------------------------------------
--site A
--server 127.0.0.1:8080
--model {m1,m2,m3}
--seed 1
--local-epochs 1
--batch 128
--lr 1e-3
--data-dir data/processed
--splits data/splits/sites.json
--device cpu|cuda

Install
-------
pip install flwr torch numpy pandas
# (optional for F1) pip install scikit-learn
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import flwr as fl
from flwr.common import NDArrays, ndarrays_to_parameters, parameters_to_ndarrays

# Optional F1 (won't crash if sklearn missing)
try:
    from sklearn.metrics import f1_score
    SK_F1 = True
except Exception:
    SK_F1 = False


# ------------------------------
# Repro
# ------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------
# Dataset (fallback loader)
# ------------------------------
class HARSiteDataset(Dataset):
    """Loads windows for a site by concatenating subjects from sites.json mapping."""

    def __init__(self, site_id: str, data_dir: str, splits_json: str, split: str = "train", val_frac: float = 0.15):
        self.site_id = site_id
        self.data_dir = Path(data_dir)
        self.split = split

        with open(splits_json, "r", encoding="utf-8") as f:
            sites_map = json.load(f)
        subjects = sites_map[site_id]

        X_list, y_list = [], []
        for sid in subjects:
            # Expect files like subject_S01.npz (adjust to your naming if different)
            npz_path = self.data_dir / f"subject_{sid}.npz"
            if not npz_path.exists():
                # Try alternate naming (S01.npz)
                npz_path = self.data_dir / f"{sid}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"Missing NPZ for subject {sid}: {npz_path}")

            with np.load(npz_path, allow_pickle=False) as d:
                X_list.append(d["X"])
                y_list.append(d["y"])

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        # train/val split (per site) — stratified would be nicer but this is simple
        n = len(X)
        idx = np.arange(n)
        rng = np.random.RandomState(42)  # keep deterministic split per site
        rng.shuffle(idx)
        val_n = int(round(val_frac * n))
        val_idx = idx[:val_n]
        tr_idx = idx[val_n:]

        if split == "train":
            self.X = X[tr_idx].astype(np.float32)
            self.y = y[tr_idx].astype(np.int64)
        else:
            self.X = X[val_idx].astype(np.float32)
            self.y = y[val_idx].astype(np.int64)

        # Infer dims
        self.T = self.X.shape[1]
        self.F = self.X.shape[2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def maybe_import_external_loader(site_id: str, data_dir: str, splits_json: str, split: str):
    """If user has a custom datasets.py with a loader, use it; else fallback."""
    try:
        import importlib
        ds_mod = importlib.import_module("site_client.datasets")
        if hasattr(ds_mod, "load_site_dataset"):
            return ds_mod.load_site_dataset(site_id=site_id, data_dir=data_dir, splits_json=splits_json, split=split)
        # Or class SiteDataset(site, split)
        if hasattr(ds_mod, "SiteDataset"):
            return ds_mod.SiteDataset(site_id, split=split)
    except Exception:
        pass
    # Fallback dataset
    return HARSiteDataset(site_id, data_dir, splits_json, split)


# ------------------------------
# Models
# ------------------------------
class TinyHARNet(nn.Module):
    # m1: small CNN + BiGRU head
    def __init__(self, n_feat=6, n_cls=8):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_feat, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.gru = nn.GRU(64, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, n_cls)

    def forward(self, x):  # x: [B, T, F]
        x = x.transpose(1, 2)         # [B, F, T]
        x = self.cnn(x)               # [B, 64, T/4]
        x = x.transpose(1, 2)         # [B, T/4, 64]
        _, h = self.gru(x)            # [2, B, 64]
        h = torch.cat([h[0], h[1]], dim=-1)  # [B, 128]
        return self.fc(h)


class CNNOnly(nn.Module):
    # m2: deeper CNN stack (no RNN)
    def __init__(self, n_feat=6, n_cls=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_feat, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, n_cls)

    def forward(self, x):
        x = x.transpose(1, 2)   # [B, F, T]
        x = self.net(x)         # [B, 128, 1]
        x = x.squeeze(-1)       # [B, 128]
        return self.head(x)


class CNNBiGRU(nn.Module):
    # m3: slightly larger CNN + BiGRU
    def __init__(self, n_feat=6, n_cls=8):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_feat, 64, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 96, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.rnn = nn.GRU(96, 96, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(192, n_cls)

    def forward(self, x):
        x = x.transpose(1, 2)     # [B, F, T]
        x = self.cnn(x)           # [B, 96, T/4]
        x = x.transpose(1, 2)     # [B, T/4, 96]
        _, h = self.rnn(x)        # [2, B, 96]
        h = torch.cat([h[0], h[1]], dim=-1)  # [B, 192]
        return self.fc(h)


def build_model(name: str, n_feat: int, n_cls: int) -> nn.Module:
    name = name.lower()
    if name == "m1":
        return TinyHARNet(n_feat, n_cls)
    if name == "m2":
        return CNNOnly(n_feat, n_cls)
    if name == "m3":
        return CNNBiGRU(n_feat, n_cls)
    raise ValueError(f"Unknown model '{name}'")


# ------------------------------
# Train / Eval
# ------------------------------
def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct += (preds == yb).long().sum().item()
            total += yb.numel()
            loss_sum += loss.item() * yb.size(0)
    acc = correct / max(1, total)
    return {"loss": loss_sum / max(1, total), "acc": acc}


@torch.no_grad()
def evaluate(model, loader, device, n_classes: int):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_y, all_p = [], []
    criterion = nn.CrossEntropyLoss()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).long().sum().item()
        total += yb.numel()
        loss_sum += loss.item() * yb.size(0)
        all_y.append(yb.cpu().numpy())
        all_p.append(preds.cpu().numpy())
    acc = correct / max(1, total)
    metrics = {"val_loss": loss_sum / max(1, total), "val_acc": acc}
    if SK_F1:
        y_true = np.concatenate(all_y) if all_y else np.array([])
        y_pred = np.concatenate(all_p) if all_p else np.array([])
        if y_true.size > 0:
            metrics["val_macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    return metrics


# ------------------------------
# Weight helpers (Torch <-> NumPy)
# ------------------------------
def get_weights(model: nn.Module) -> NDArrays:
    return [p.detach().cpu().numpy().copy() for p in model.state_dict().values()]

def set_weights(model: nn.Module, weights: NDArrays) -> None:
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    if len(keys) != len(weights):
        raise ValueError("Weights length mismatch")
    new_sd = {k: torch.tensor(w) for k, w in zip(keys, weights)}
    model.load_state_dict(new_sd, strict=True)


# ------------------------------
# Flower Client
# ------------------------------
class HARFlowerClient(fl.client.NumPyClient):
    def __init__(self, site_id: str, server: str, model_name: str, data_dir: str, splits_json: str,
                 local_epochs: int, batch: int, lr: float, device: str, seed: int):
        set_seed(seed)
        self.site_id = site_id
        self.model_name = model_name
        self.device = torch.device(device)

        # Datasets (try custom loader first, else fallback)
        train_ds = maybe_import_external_loader(site_id, data_dir, splits_json, split="train")
        val_ds = maybe_import_external_loader(site_id, data_dir, splits_json, split="val")

        self.n_feat = train_ds.X.shape[2]
        self.n_cls = int(np.max(np.concatenate([train_ds.y, val_ds.y])) + 1)

        self.model = build_model(model_name, self.n_feat, self.n_cls).to(self.device)
        self.train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=False)
        self.val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, drop_last=False)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.local_epochs = local_epochs

    # --- Flower hooks ---
    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        if parameters is not None:
            set_weights(self.model, parameters)
        # Local training
        metrics_last = {}
        for _ in range(self.local_epochs):
            metrics_last = train_one_epoch(self.model, self.train_loader, self.device, self.optimizer, self.criterion)
        # Return new weights + metrics
        return get_weights(self.model), len(self.train_loader.dataset), metrics_last

    def evaluate(self, parameters, config):
        if parameters is not None:
            set_weights(self.model, parameters)
        metrics = evaluate(self.model, self.val_loader, self.device, self.n_cls)
        # Flower expects (loss, num_examples, metrics_dict)
        return float(metrics.get("val_loss", 0.0)), len(self.val_loader.dataset), metrics


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=str, required=True, help="Site ID (e.g., A, B, C)")
    ap.add_argument("--server", type=str, required=True, help="Flower server host:port")
    ap.add_argument("--model", type=str, default=os.getenv("MODEL", "m1"), choices=["m1","m2","m3"])
    ap.add_argument("--seed", type=int, default=int(os.getenv("SEED", "1")))
    ap.add_argument("--local-epochs", type=int, default=int(os.getenv("LOCAL_EPOCHS", "1")))
    ap.add_argument("--batch", type=int, default=int(os.getenv("BATCH", "128")))
    ap.add_argument("--lr", type=float, default=float(os.getenv("LR", "1e-3")))
    ap.add_argument("--data-dir", type=str, default=os.getenv("DATA_DIR", "data/processed"))
    ap.add_argument("--splits", type=str, default=os.getenv("SPLITS_JSON", "data/splits/sites.json"))
    ap.add_argument("--device", type=str, default=os.getenv("DEVICE", "cpu"))
    args = ap.parse_args()

    client = HARFlowerClient(
        site_id=args.site,
        server=args.server,
        model_name=args.model,
        data_dir=args.data_dir,
        splits_json=args.splits,
        local_epochs=args.local_epochs,
        batch=args.batch,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )

    print(f"[client] site={args.site} model={args.model} data_dir={args.data_dir} splits={args.splits}")
    fl.client.start_numpy_client(server_address=args.server, client=client)


if __name__ == "__main__":
    main()
