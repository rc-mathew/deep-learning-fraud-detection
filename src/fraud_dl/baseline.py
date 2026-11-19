"""Week 4 baseline: one XGBoost model on raw + velocity features, time-split.

    python -m fraud_dl.baseline --artifacts artifacts [--gpu]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config
from .metrics import fraud_report

# Engineered columns that belong to the sequence pipeline, not the tabular model.
_SEQ_ONLY = ["uid_code", "group_start", "pos_in_card"]


def feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Model inputs. Text columns become integer codes using train categories only."""
    drop = set(config.NON_FEATURE_COLS) | set(_SEQ_ONLY)
    cols = [c for c in df.columns if c not in drop]
    X = df[cols].copy()
    train_mask = (df["split"] == "train").to_numpy()
    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(X[c])]
    for col in cat_cols:
        cats = pd.Index(X.loc[train_mask, col].dropna().unique())
        codes = pd.Categorical(X[col], categories=cats).codes.astype(np.float32)
        codes[codes < 0] = np.nan  # missing or unseen in train
        X[col] = codes
    return X, cat_cols


def train_baseline(df: pd.DataFrame, use_gpu: bool = False, seed: int = 42) -> tuple[xgb.XGBClassifier, dict]:
    X, _ = feature_frame(df)
    y = df["isFraud"].to_numpy()
    split = df["split"].to_numpy()
    tr, va, te = split == "train", split == "val", split == "test"

    model = xgb.XGBClassifier(
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.5,
        min_child_weight=5,
        tree_method="hist",
        device="cuda" if use_gpu else "cpu",
        eval_metric="aucpr",
        early_stopping_rounds=100,
        random_state=seed,
        n_jobs=-1,
    )
    t0 = time.time()
    model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=200)
    train_secs = time.time() - t0

    results = {"model": "xgboost_baseline", "best_iteration": int(model.best_iteration),
               "train_seconds": round(train_secs, 1), "n_features": int(X.shape[1])}
    for name, mask in [("val", va), ("test", te)]:
        scores = model.predict_proba(X[mask])[:, 1]
        results[name] = fraud_report(y[mask], scores)
    return model, results


def append_results_row(results: dict, path: Path) -> None:
    """The README's results table: one row per model, test-set metrics."""
    t = results["test"]
    row = pd.DataFrame([{
        "model": results["model"],
        "recall_at_1pct_fpr": round(t["recall_at_1pct_fpr"], 4),
        "recall_at_5pct_fpr": round(t["recall_at_5pct_fpr"], 4),
        "pr_auc": round(t["pr_auc"], 4),
        "roc_auc": round(t["roc_auc"], 4),
        "test_rows": t["n"],
    }])
    if path.exists():
        old = pd.read_csv(path)
        row = pd.concat([old[old["model"] != results["model"]], row], ignore_index=True)
    row.to_csv(path, index=False)


def top_features(model: xgb.XGBClassifier, k: int = 25) -> dict[str, float]:
    gain = model.get_booster().get_score(importance_type="total_gain")
    total = sum(gain.values()) or 1.0
    ranked = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return {name: round(v / total, 4) for name, v in ranked}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    art = Path(args.artifacts)
    (art / "models").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(art / "transactions.parquet")
    model, results = train_baseline(df, use_gpu=args.gpu)
    results["top_features_by_gain"] = top_features(model)

    model.save_model(art / "models" / "xgb_baseline.json")
    (art / "xgb_baseline_results.json").write_text(json.dumps(results, indent=1))
    append_results_row(results, art / "results.csv")
    print(json.dumps({k: results[k] for k in ("best_iteration", "train_seconds", "val", "test")}, indent=1))
    print(pd.read_csv(art / "results.csv").to_string(index=False))


if __name__ == "__main__":
    main()
