"""Past-only velocity features per card. Needs sort_by_card_and_time() first."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

_KEY_STRIDE = np.int64(10**10)  # wider than any TransactionDT + window, so cards never overlap


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Counts, amount sums and gaps over each card's earlier transactions.

    Every feature uses only transactions at or before the current timestamp, so
    the same features can be computed at scoring time. The current transaction is
    excluded from counts and sums.
    """
    uid_code = df["uid_code"].to_numpy(np.int64)
    if np.any(np.diff(uid_code) < 0):
        raise ValueError("Call sort_by_card_and_time() before add_velocity_features().")
    t = df["TransactionDT"].to_numpy(np.int64)
    amt = df["TransactionAmt"].to_numpy(np.float64)
    group_start = df["group_start"].to_numpy(np.int64)
    idx = np.arange(len(df), dtype=np.int64)

    key = uid_code * _KEY_STRIDE + t
    amt_cumsum = np.concatenate([[0.0], np.cumsum(amt)])

    for name, window in config.VELOCITY_WINDOWS.items():
        first = np.searchsorted(key, key - window, side="left")
        df[f"n_txn_{name}"] = (idx - first).astype(np.float32)
        df[f"amt_sum_{name}"] = (amt_cumsum[idx] - amt_cumsum[first]).astype(np.float32)

    n_prior = idx - group_start
    df["n_txn_prior"] = n_prior.astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        prior_mean = (amt_cumsum[idx] - amt_cumsum[group_start]) / n_prior
        prior_mean[n_prior == 0] = np.nan
        ratio = amt / prior_mean
    ratio[~np.isfinite(ratio)] = np.nan
    df["amt_prior_mean"] = prior_mean.astype(np.float32)
    df["amt_to_prior_mean"] = ratio.astype(np.float32)

    gap = np.full(len(df), np.nan)
    has_prev = n_prior > 0
    gap[has_prev] = t[has_prev] - t[idx[has_prev] - 1]
    df["secs_since_prev"] = gap.astype(np.float32)
    return df
