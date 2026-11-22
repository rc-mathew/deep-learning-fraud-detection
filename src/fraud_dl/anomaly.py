"""Autoencoder trained only on legitimate transactions (week 7).

It learns to reconstruct normal transactions; a large reconstruction error flags an
unusual one. No fraud labels are used for training, so it can catch patterns nobody
has labelled yet. Labels are used only to evaluate it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn


def numeric_matrix(df: pd.DataFrame, feature_cols: list[str], fit_rows: np.ndarray) -> np.ndarray:
    """Signed log1p, standardised with `fit_rows` statistics; missing becomes 0 (the mean)."""
    x = df[feature_cols].to_numpy(np.float32)
    x = np.sign(x) * np.log1p(np.abs(x))
    mean = np.nanmean(x[fit_rows], axis=0)
    std = np.nanstd(x[fit_rows], axis=0)
    std[~(std > 1e-6)] = 1.0
    mean = np.nan_to_num(mean)
    x = (x - mean) / std
    return np.clip(np.nan_to_num(x, nan=0.0), -10, 10).astype(np.float32)


class AutoEncoder(nn.Module):
    def __init__(self, n_in: int, hidden: int = 256, bottleneck: int = 32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, bottleneck), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(bottleneck, hidden), nn.ReLU(), nn.Linear(hidden, n_in))

    def forward(self, x):
        return self.dec(self.enc(x))


def train_autoencoder(x: np.ndarray, normal_rows: np.ndarray, device: str, epochs: int = 10,
                      batch_size: int = 1024, lr: float = 1e-3, seed: int = 0) -> tuple[AutoEncoder, list[float]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AutoEncoder(x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.from_numpy(x)
    history = []
    for epoch in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for b in np.array_split(rng.permutation(normal_rows), max(1, len(normal_rows) // batch_size)):
            xb = xt[b].to(device)
            loss = ((model(xb) - xb) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        history.append(tot / n)
        print({"epoch": epoch + 1, "recon_loss": history[-1]})
    return model, history


@torch.no_grad()
def anomaly_score(model: AutoEncoder, x: np.ndarray, rows: np.ndarray, device: str,
                  batch_size: int = 8192) -> np.ndarray:
    """Mean squared reconstruction error per transaction: higher = more unusual."""
    model.eval()
    out = []
    for i in range(0, len(rows), batch_size):
        xb = torch.from_numpy(x[rows[i:i + batch_size]]).to(device)
        out.append(((model(xb) - xb) ** 2).mean(dim=1).cpu().numpy())
    return np.concatenate(out)
