"""Per-card transaction sequences for the transformer (weeks 5-6).

Storage is one row per transaction, sorted by card then time, instead of padded
windows: 590k x 32 x ~40 values would be ~3 GB, while this table is ~100 MB. The
week 5 Dataset rebuilds a window with window_indices().

All encoders are fitted on train rows only. A test transaction's window can
include earlier train-period transactions of the same card, which is fine: that
history exists at scoring time. Labels of earlier transactions are never inputs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


@dataclass
class SequenceEncoder:
    cat_cols: list[str]
    log_cols: list[str]
    plain_cols: list[str]
    vocab: dict[str, dict[str, int]] = field(default_factory=dict)
    num_mean: dict[str, float] = field(default_factory=dict)
    num_std: dict[str, float] = field(default_factory=dict)

    @property
    def num_cols(self) -> list[str]:
        return self.log_cols + self.plain_cols

    def fit(self, train: pd.DataFrame, min_count: int = config.MIN_CAT_COUNT) -> "SequenceEncoder":
        for col in self.cat_cols:
            counts = _as_key(train[col]).value_counts()
            kept = sorted(counts[counts >= min_count].index)
            # 0 = padding, 1 = unknown/rare, real categories start at 2. Missing is its own category.
            self.vocab[col] = {k: i + 2 for i, k in enumerate(kept)}
        nums = self._raw_numeric(train)
        for col in self.num_cols:
            vals = nums[col]
            mean, std = float(np.nanmean(vals)), float(np.nanstd(vals))
            self.num_mean[col] = mean
            self.num_std[col] = std if std > 1e-6 else 1.0
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in self.cat_cols:
            codes = _as_key(df[col]).map(self.vocab[col])
            out[f"cat_{col}"] = codes.fillna(config.UNK_ID).astype(np.int32).to_numpy()
        nums = self._raw_numeric(df)
        for col in self.num_cols:
            z = (nums[col] - self.num_mean[col]) / self.num_std[col]
            out[f"num_{col}"] = np.nan_to_num(z, nan=0.0).astype(np.float32)
            out[f"isna_{col}"] = np.isnan(nums[col]).astype(np.int8)
        return pd.DataFrame(out, index=df.index)

    def _raw_numeric(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        nums = {}
        for col in self.log_cols:
            v = df[col].to_numpy(np.float64)
            nums[col] = np.log1p(np.clip(v, 0, None))
        for col in self.plain_cols:
            nums[col] = df[col].to_numpy(np.float64)
        return nums

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "SequenceEncoder":
        return cls(**json.loads(Path(path).read_text()))


def _as_key(s: pd.Series) -> pd.Series:
    """Categories as strings so ints, floats and text share one vocabulary format."""
    if pd.api.types.is_float_dtype(s):
        s = s.astype("Int64")
    s = s.astype(object)
    return s.where(s.notna(), "<NA>").astype(str)


def build_sequence_table(df: pd.DataFrame, encoder: SequenceEncoder) -> pd.DataFrame:
    """One encoded row per transaction, in card-then-time order."""
    if np.any(np.diff(df["uid_code"].to_numpy()) < 0):
        raise ValueError("Call sort_by_card_and_time() first.")
    meta = df[["TransactionID", "TransactionDT", "isFraud", "split",
               "uid_code", "group_start", "pos_in_card"]].reset_index(drop=True)
    enc = encoder.transform(df).reset_index(drop=True)
    return pd.concat([meta, enc], axis=1)


def window_indices(rows: np.ndarray, group_start: np.ndarray, seq_len: int = config.SEQ_LEN) -> np.ndarray:
    """Row indices of each target's last `seq_len` transactions, oldest first.

    Returns an int64 array of shape (len(rows), seq_len), left-padded with -1.
    The target transaction is always the last column.
    """
    rows = np.asarray(rows, dtype=np.int64)
    offsets = np.arange(-seq_len + 1, 1, dtype=np.int64)  # e.g. -31 .. 0
    idx = rows[:, None] + offsets[None, :]
    start = group_start[rows][:, None]
    return np.where(idx >= start, idx, -1)
