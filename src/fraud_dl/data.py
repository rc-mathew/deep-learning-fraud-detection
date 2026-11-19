"""Load IEEE-CIS, derive a card id and split by time."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def load_raw(data_dir: str | Path) -> pd.DataFrame:
    """Read train_transaction + train_identity, left-joined on TransactionID.

    Only the labelled train files are used; Kaggle's test files have no labels.
    Floats are downcast to float32, which roughly halves memory (~1 GB instead of ~2 GB).
    """
    data_dir = _find_data_dir(Path(data_dir))
    txn = pd.read_csv(data_dir / "train_transaction.csv")
    ident = pd.read_csv(data_dir / "train_identity.csv")
    # Kaggle's test identity file spells these id-30; normalise in case of mixed files.
    ident.columns = [c.replace("-", "_") for c in ident.columns]
    df = txn.merge(ident, on="TransactionID", how="left")
    return downcast(df)


def _find_data_dir(data_dir: Path) -> Path:
    """Kaggle's mount path varies (/kaggle/input/<slug> or /kaggle/input/competitions/<slug>)."""
    if (data_dir / "train_transaction.csv").exists():
        return data_dir
    root = data_dir if data_dir.exists() else data_dir.parent
    hit = next(root.rglob("train_transaction.csv"), None)
    if hit is None:
        raise FileNotFoundError(f"train_transaction.csv not found under {root}")
    return hit.parent


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype(np.float32)
    return df


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """TransactionDT is seconds from an unknown start date, so derive day and hour from it."""
    dt = df["TransactionDT"].astype(np.int64)
    df["day"] = (dt // config.SECONDS_PER_DAY).astype(np.int32)
    hour = (dt % config.SECONDS_PER_DAY) / 3_600
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    return df


def add_uid(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate card/account id: card1 + addr1 + the day the card was first seen.

    IEEE-CIS has no card id. D1 is "days since the card's first transaction", so
    day - D1 is constant for one card. This is the key found by the competition's
    winning team; it is an approximation, and rows with missing D1 or addr1 get
    a coarser id.
    """
    first_day = np.floor(df["day"] - df["D1"])
    df["uid"] = _int_key(df["card1"]) + "_" + _int_key(df["addr1"]) + "_" + _int_key(first_day)
    return df


def _int_key(s: pd.Series) -> pd.Series:
    """Integer-like values as text, with missing spelled "na" (same on pandas 2 and 3)."""
    s = s.astype("Int64").astype(object)
    return s.where(s.notna(), "na").astype(str)


def time_split(
    df: pd.DataFrame,
    test_days: int = config.TEST_DAYS,
    val_days: int = config.VAL_DAYS,
    gap_days: int = config.GAP_DAYS,
) -> pd.DataFrame:
    """Label rows train / val / test by day: past trains, the future tests.

    Layout: [train][gap][val][test]. Gap rows are labelled "gap" and never used.
    """
    last_day = int(df["day"].max())
    test_start = last_day - test_days + 1
    val_start = test_start - val_days
    train_end = val_start - gap_days  # exclusive
    split = np.full(len(df), "gap", dtype=object)
    day = df["day"].to_numpy()
    split[day < train_end] = "train"
    split[(day >= val_start) & (day < test_start)] = "val"
    split[day >= test_start] = "test"
    df["split"] = split
    if not (df["split"] == "train").any():
        raise ValueError("No training rows left; reduce test_days, val_days or gap_days.")
    return df


def sort_by_card_and_time(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by card, then time, and add each card's start row and position.

    After this sort a card's history is a contiguous block, so the last k
    transactions of row r are rows max(group_start[r], r-k+1) .. r.
    """
    df = df.sort_values(["uid", "TransactionDT", "TransactionID"], kind="mergesort")
    df = df.reset_index(drop=True)
    uid_code = pd.factorize(df["uid"], sort=False)[0].astype(np.int64)
    # factorize assigns codes in order of appearance, which is sorted here.
    df["uid_code"] = uid_code
    df["group_start"] = np.searchsorted(uid_code, uid_code, side="left").astype(np.int64)
    df["pos_in_card"] = (np.arange(len(df)) - df["group_start"].to_numpy()).astype(np.int32)
    return df
