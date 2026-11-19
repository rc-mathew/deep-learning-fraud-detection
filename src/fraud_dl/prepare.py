"""Week 4 data step: raw CSVs -> time-split feature table + sequence table.

    python -m fraud_dl.prepare --data-dir /kaggle/input/ieee-fraud-detection --out-dir artifacts
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data import add_time_columns, add_uid, load_raw, sort_by_card_and_time, time_split
from .features import add_velocity_features
from .sequences import SequenceEncoder, build_sequence_table


def prepare(df: pd.DataFrame, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "sequences").mkdir(parents=True, exist_ok=True)

    df = add_time_columns(df)
    df = add_uid(df)
    df = time_split(df)
    df = sort_by_card_and_time(df)
    df = add_velocity_features(df)

    df.to_parquet(out_dir / "transactions.parquet", index=False)

    encoder = SequenceEncoder(config.SEQ_CAT_COLS, config.SEQ_LOG_COLS, config.SEQ_PLAIN_COLS)
    encoder.fit(df[df["split"] == "train"])
    seq = build_sequence_table(df, encoder)
    seq.to_parquet(out_dir / "sequences" / "tokens.parquet", index=False)
    encoder.save(out_dir / "sequences" / "encoder.json")
    (out_dir / "sequences" / "meta.json").write_text(json.dumps({
        "seq_len": config.SEQ_LEN,
        "pad_id": config.PAD_ID,
        "unk_id": config.UNK_ID,
        "cat_cols": [f"cat_{c}" for c in encoder.cat_cols],
        "cat_cardinality": {f"cat_{c}": len(v) + 2 for c, v in encoder.vocab.items()},
        "num_cols": [f"num_{c}" for c in encoder.num_cols],
        "isna_cols": [f"isna_{c}" for c in encoder.num_cols],
        "note": "Rows are sorted by card then time; use fraud_dl.sequences.window_indices.",
    }, indent=1))

    report = data_report(df)
    (out_dir / "data_report.json").write_text(json.dumps(report, indent=1))
    return report


def data_report(df: pd.DataFrame) -> dict:
    report: dict = {"n_rows": int(len(df)), "n_cards": int(df["uid_code"].nunique()), "splits": {}}
    train_uids = set(df.loc[df["split"] == "train", "uid_code"])
    for name, part in df.groupby("split"):
        report["splits"][name] = {
            "rows": int(len(part)),
            "fraud_rate": round(float(part["isFraud"].mean()), 4),
            "first_day": int(part["day"].min()),
            "last_day": int(part["day"].max()),
            "share_cards_seen_in_train": round(float(part["uid_code"].isin(train_uids).mean()), 3),
            "median_prior_txns": float(part["pos_in_card"].median()),
            "share_with_no_history": round(float((part["pos_in_card"] == 0).mean()), 3),
        }
    # How often a card that had a fraud earlier keeps producing fraud. IEEE-CIS labels
    # later transactions of a defrauded account as fraud too, which inflates
    # card-history models. Worth knowing before trusting a sequence model's lift.
    fraud_cumsum = np.concatenate([[0], np.cumsum(df["isFraud"].to_numpy())])
    idx = np.arange(len(df))
    prior_frauds = fraud_cumsum[idx] - fraud_cumsum[df["group_start"].to_numpy()]
    after = df[prior_frauds > 0]
    report["fraud_rate_after_card_had_fraud"] = round(float(after["isFraud"].mean()), 3) if len(after) else None
    report["fraud_rate_overall"] = round(float(df["isFraud"].mean()), 4)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="artifacts")
    args = ap.parse_args()
    t0 = time.time()
    df = load_raw(args.data_dir)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} cols in {time.time() - t0:.0f}s")
    report = prepare(df, args.out_dir)
    print(json.dumps(report, indent=1))
    print(f"Done in {time.time() - t0:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
