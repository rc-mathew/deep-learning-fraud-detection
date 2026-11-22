"""Entity graph + GraphSAGE (week 7).

Nodes are transactions. An edge runs from an earlier transaction to a later one when both
share an entity: the same card, the same device fingerprint, or the same email domain +
billing address. Each transaction links to at most the previous K that share each entity,
which keeps hubs (one popular device model, gmail.com) from exploding the graph.
Edges only point forward in time, so a transaction's score only uses earlier transactions,
just as it would in production. Labels are never node features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv

from .metrics import fraud_report

PREV_K = 5
# Device fingerprint: model + OS + browser + screen (+ two identity fields when present).
DEVICE_COLS = ["DeviceInfo", "id_30", "id_31", "id_33", "id_19", "id_20"]
ENTITIES = {
    "card": ["uid"],
    "device": DEVICE_COLS,
    "email_addr": ["P_emaildomain", "addr1"],
}


def entity_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """One string per row; missing when the entity's first column is missing."""
    cols = [c for c in cols if c in df.columns]
    key = df[cols[0]].astype(str).astype(object).where(df[cols[0]].notna())
    for c in cols[1:]:
        key = key + "|" + df[c].astype(str).astype(object).where(df[c].notna(), "na")
    return key


def build_edges(df: pd.DataFrame, k: int = PREV_K) -> dict[str, np.ndarray]:
    """Edges (src earlier -> dst later) over row positions of `df`, plus their entity type."""
    t = df["TransactionDT"].to_numpy(np.int64)
    srcs, dsts, types = [], [], []
    for etype, (name, cols) in enumerate(ENTITIES.items()):
        key = entity_key(df, cols)
        rows = np.flatnonzero(key.notna().to_numpy())
        if len(rows) == 0:
            continue
        codes = pd.factorize(key.iloc[rows])[0]
        order = np.lexsort((rows, t[rows], codes))  # by entity, then time
        rows, codes = rows[order], codes[order]
        for lag in range(1, k + 1):
            same = codes[lag:] == codes[:-lag]
            srcs.append(rows[:-lag][same]); dsts.append(rows[lag:][same])
            types.append(np.full(same.sum(), etype, dtype=np.int8))
    src, dst, etype = np.concatenate(srcs), np.concatenate(dsts), np.concatenate(types)
    return {"src": src, "dst": dst, "etype": etype}


def ring_mask(df: pd.DataFrame, edges: dict[str, np.ndarray]) -> np.ndarray:
    """Transactions whose device was used earlier by a different card: the ring-fraud slice."""
    uid = df["uid_code"].to_numpy()
    dev = edges["etype"] == list(ENTITIES).index("device")
    s, d = edges["src"][dev], edges["dst"][dev]
    other = uid[s] != uid[d]
    mask = np.zeros(len(df), dtype=bool)
    mask[d[other]] = True
    return mask


def save_graph(path: Path, df: pd.DataFrame, edges: dict[str, np.ndarray]) -> None:
    """Saved for the Risk Agent's graph lookup tool (rows are TransactionIDs)."""
    np.savez_compressed(path, transaction_id=df["TransactionID"].to_numpy(),
                        entity_types=np.array(list(ENTITIES)), **edges)


class FraudSAGE(nn.Module):
    """Token embedding (as in the transformer) + 2 GraphSAGE layers with residuals."""

    def __init__(self, cardinality: list[int], n_num: int, d: int = 64, dropout: float = 0.2):
        super().__init__()
        self.cat_emb = nn.ModuleList(nn.Embedding(c, d) for c in cardinality)
        self.num_proj = nn.Linear(2 * n_num, d)
        self.convs = nn.ModuleList([SAGEConv(d, d), SAGEConv(d, d)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.head = nn.Linear(d, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, cat, num, isna, edge_index):
        h = sum(emb(cat[:, j]) for j, emb in enumerate(self.cat_emb))
        h = self.norms[0](h + self.num_proj(torch.cat([num, isna], dim=-1)))
        for conv, norm in zip(self.convs, self.norms[1:]):
            h = norm(h + self.drop(F.relu(conv(h, edge_index))))
        return self.head(h).squeeze(-1)


def train_gnn(model: FraudSAGE, x: tuple, edge_index: torch.Tensor, y: np.ndarray,
              train: np.ndarray, val: np.ndarray, epochs: int = 200, lr: float = 3e-3,
              eval_every: int = 5, patience: int = 6, ckpt: Path | None = None) -> list[dict]:
    """Full-batch training: the whole graph fits on one GPU. Early stopping on val PR-AUC."""
    dev = edge_index.device
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    tr = torch.from_numpy(train).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best, bad, history, best_state = -1.0, 0, [], None
    for epoch in range(1, epochs + 1):
        model.train()
        loss = F.binary_cross_entropy_with_logits(model(*x, edge_index)[tr], yt[tr])
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % eval_every:
            continue
        rep = fraud_report(y[val], predict_gnn(model, x, edge_index)[val])
        history.append({"epoch": epoch, "loss": loss.item(), "val_pr_auc": rep["pr_auc"],
                        "val_recall_at_1pct_fpr": rep["recall_at_1pct_fpr"]})
        print(history[-1])
        if rep["pr_auc"] > best:
            best, bad = rep["pr_auc"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    if ckpt:
        torch.save(best_state, ckpt)
    return history


@torch.no_grad()
def predict_gnn(model: FraudSAGE, x: tuple, edge_index: torch.Tensor) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(*x, edge_index)).cpu().numpy()
