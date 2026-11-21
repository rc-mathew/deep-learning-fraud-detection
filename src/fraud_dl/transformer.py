"""Transaction transformer written from scratch, with masked-field pretraining.

Each transaction is one token: its categorical fields are embedded and summed, its
numeric fields projected, plus a learned position. A card's last SEQ_LEN transactions
form the sequence; the fraud score is read from the last token (the transaction being
scored). Pretraining hides random fields and predicts them back, like BERT's masked
words, using no labels.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .metrics import fraud_report
from .sequences import window_indices


class SeqData:
    """The encoded sequence table on one device. `batch(rows)` gathers each row's window."""

    def __init__(self, seq: pd.DataFrame, meta: dict, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.seq_len = meta["seq_len"]
        self.cat_cols, self.num_cols = meta["cat_cols"], meta["num_cols"]
        self.cardinality = [meta["cat_cardinality"][c] for c in self.cat_cols]
        n = len(seq)
        self.pad_row = n  # an all-zero row that window padding points at

        def table(cols, dtype):
            arr = np.zeros((n + 1, len(cols)), dtype=dtype)
            arr[:n] = seq[cols].to_numpy(dtype)
            return torch.from_numpy(arr).to(self.device)

        self.cat = table(self.cat_cols, np.int64)
        self.num = table(self.num_cols, np.float32)
        self.isna = table(meta["isna_cols"], np.float32)
        self.group_start = seq["group_start"].to_numpy(np.int64)
        self.y = seq["isFraud"].to_numpy(np.int64)
        self.split = seq["split"].to_numpy()

    def rows(self, *splits: str) -> np.ndarray:
        return np.flatnonzero(np.isin(self.split, splits))

    def batch(self, rows: np.ndarray):
        idx = window_indices(rows, self.group_start, self.seq_len)
        idx = torch.from_numpy(np.where(idx < 0, self.pad_row, idx)).to(self.device)
        return self.cat[idx], self.num[idx], self.isna[idx], idx == self.pad_row


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad: torch.Tensor):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)  # (B, heads, L, L)
        # Padding is never attended to; the last token is always real, so no row is all -inf.
        scores = scores.masked_fill(pad[:, None, None, :], float("-inf"))
        attn = scores.softmax(dim=-1)
        y = (self.drop(attn) @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(y), attn


class Block(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad):
        a, attn = self.attn(self.ln1(x), pad)
        x = x + self.drop(a)
        return x + self.drop(self.mlp(self.ln2(x))), attn


class TxnTransformer(nn.Module):
    def __init__(self, cardinality: list[int], n_num: int, seq_len: int,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.cardinality = cardinality
        # One extra id per field for [MASK] during pretraining.
        self.cat_emb = nn.ModuleList(nn.Embedding(c + 1, d_model) for c in cardinality)
        self.num_proj = nn.Linear(3 * n_num, d_model)  # value, is-missing flag, is-masked flag
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.ln_in = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, dropout) for _ in range(n_layers))
        self.ln_out = nn.LayerNorm(d_model)
        self.fraud_head = nn.Linear(d_model, 1)
        self.cat_heads = nn.ModuleList(nn.Linear(d_model, c) for c in cardinality)
        self.num_head = nn.Linear(d_model, n_num)

    def encode(self, cat, num, isna, pad, num_masked=None):
        if num_masked is None:
            num_masked = torch.zeros_like(num)
        x = sum(emb(cat[..., j]) for j, emb in enumerate(self.cat_emb))
        x = x + self.num_proj(torch.cat([num, isna, num_masked], dim=-1))
        x = self.ln_in(x + self.pos_emb.weight[None])
        attns = []
        for block in self.blocks:
            x, attn = block(x, pad)
            attns.append(attn)
        return self.ln_out(x), attns

    def forward(self, cat, num, isna, pad):
        """Fraud logit for the last transaction in each window."""
        h, _ = self.encode(cat, num, isna, pad)
        return self.fraud_head(h[:, -1]).squeeze(-1)

    def pretrain_loss(self, cat, num, isna, pad, p: float = 0.15):
        """Hide ~p of the fields of real tokens and predict them from the rest of the sequence."""
        real = ~pad
        m_cat = (torch.rand(cat.shape, device=cat.device) < p) & real[..., None]
        mask_ids = torch.tensor(self.cardinality, device=cat.device)
        cat_in = torch.where(m_cat, mask_ids, cat)
        m_num = (torch.rand(num.shape, device=num.device) < p) & real[..., None] & (isna == 0)
        num_in = num.masked_fill(m_num, 0.0)

        h, _ = self.encode(cat_in, num_in, isna, pad, m_num.float())
        losses = []
        for j, head in enumerate(self.cat_heads):
            sel = m_cat[..., j]
            if sel.any():
                losses.append(F.cross_entropy(head(h[sel]).float(), cat[..., j][sel]))
        cat_loss = torch.stack(losses).mean()
        num_loss = F.mse_loss(self.num_head(h)[m_num].float(), num[m_num]) if m_num.any() else cat_loss * 0
        return cat_loss + num_loss, cat_loss.detach(), num_loss.detach()


def _amp(device: torch.device):
    on = device.type == "cuda"
    return torch.autocast("cuda", dtype=torch.float16, enabled=on), torch.amp.GradScaler("cuda", enabled=on)


def _step(loss, opt, scaler, model):
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()


def pretrain(model: TxnTransformer, data: SeqData, rows: np.ndarray, epochs: int, ckpt: Path,
             batch_size: int = 512, lr: float = 1e-3, seed: int = 0) -> list[dict]:
    """Masked-field pretraining. Saves a checkpoint every epoch and resumes from it."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    autocast, scaler = _amp(data.device)
    history, start = [], 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location=data.device)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
        history, start = state["history"], len(state["history"])
        print(f"resumed pretraining at epoch {start}")
    for epoch in range(start, epochs):
        model.train()
        tot = cat_tot = num_tot = 0.0
        batches = np.array_split(rng.permutation(rows), max(1, len(rows) // batch_size))
        for b in batches:
            with autocast:
                loss, cat_l, num_l = model.pretrain_loss(*data.batch(b))
            _step(loss, opt, scaler, model)
            tot += loss.item(); cat_tot += cat_l.item(); num_tot += num_l.item()
        n = len(batches)
        history.append({"epoch": epoch + 1, "loss": tot / n, "cat_loss": cat_tot / n, "num_loss": num_tot / n})
        print(history[-1])
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "history": history}, ckpt)
    return history


@torch.no_grad()
def predict(model: TxnTransformer, data: SeqData, rows: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    autocast, _ = _amp(data.device)
    out = []
    for i in range(0, len(rows), batch_size):
        with autocast:
            out.append(torch.sigmoid(model(*data.batch(rows[i:i + batch_size])).float()).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def finetune(model: TxnTransformer, data: SeqData, train_rows: np.ndarray, val_rows: np.ndarray,
             epochs: int, ckpt: Path, batch_size: int = 512, lr: float = 5e-4, patience: int = 2,
             seed: int = 0) -> list[dict]:
    """Fraud training with early stopping on validation PR-AUC; keeps the best epoch's weights."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    autocast, scaler = _amp(data.device)
    y = torch.tensor(data.y, dtype=torch.float32).to(data.device)
    best, history, bad = -1.0, [], 0
    for epoch in range(epochs):
        model.train()
        tot = 0.0
        batches = np.array_split(rng.permutation(train_rows), max(1, len(train_rows) // batch_size))
        for b in batches:
            with autocast:
                loss = F.binary_cross_entropy_with_logits(model(*data.batch(b)).float(), y[b])
            _step(loss, opt, scaler, model)
            tot += loss.item()
        val = fraud_report(data.y[val_rows], predict(model, data, val_rows))
        history.append({"epoch": epoch + 1, "train_loss": tot / len(batches),
                        "val_pr_auc": val["pr_auc"], "val_recall_at_1pct_fpr": val["recall_at_1pct_fpr"]})
        print(history[-1])
        if val["pr_auc"] > best:
            best, bad = val["pr_auc"], 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(torch.load(ckpt, map_location=data.device))
    return history
