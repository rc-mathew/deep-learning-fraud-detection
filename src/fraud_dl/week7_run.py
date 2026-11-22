"""Week 7: GraphSAGE on the entity graph and an autoencoder, against XGBoost on the same test set.

    python -m fraud_dl.week7_run --artifacts artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .anomaly import anomaly_score, numeric_matrix, train_autoencoder
from .baseline import append_results_row, feature_frame, train_baseline
from .graph import FraudSAGE, build_edges, predict_gnn, ring_mask, save_graph, train_gnn
from .metrics import card_had_prior_fraud, fraud_report, threshold_at_fpr


def report(y, scores, prior, ring) -> dict:
    """Standard metrics, plus ring-fraud recall at the same 1% FPR threshold used for all rows."""
    rep = fraud_report(y, scores, prior)
    hit = scores > threshold_at_fpr(y, scores, 0.01)
    ring_pos = ring & (y == 1)
    rep["ring_rows"] = int(ring.sum())
    rep["ring_fraud_rate"] = float(y[ring].mean()) if ring.any() else float("nan")
    rep["ring_recall_at_1pct_fpr"] = float(hit[ring_pos].mean()) if ring_pos.any() else float("nan")
    return rep


def run(art: Path, gnn_epochs: int = 300, ae_epochs: int = 10, device: str | None = None,
        use_gpu_xgb: bool | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_gpu_xgb = device == "cuda" if use_gpu_xgb is None else use_gpu_xgb
    df = pd.read_parquet(art / "transactions.parquet")
    seq = pd.read_parquet(art / "sequences" / "tokens.parquet")
    meta = json.loads((art / "sequences" / "meta.json").read_text())
    assert (df["TransactionID"].to_numpy() == seq["TransactionID"].to_numpy()).all(), "tables out of order"

    y = df["isFraud"].to_numpy()
    split = df["split"].to_numpy()
    train, val, test = (np.flatnonzero(split == s) for s in ("train", "val", "test"))
    prior = card_had_prior_fraud(y, df["group_start"].to_numpy())

    edges = build_edges(df)
    (art / "graph").mkdir(exist_ok=True)
    save_graph(art / "graph" / "edges.npz", df, edges)
    ring = ring_mask(df, edges)
    out: dict = {"n_edges": int(len(edges["src"])), "edges_by_type": np.bincount(edges["etype"]).tolist(),
                 "ring_share_of_test": float(ring[test].mean())}
    print(json.dumps(out))

    # XGBoost again, to score it on the ring slice.
    xgb_model, _ = train_baseline(df, use_gpu=use_gpu_xgb)
    X, cat_cols = feature_frame(df)
    xgb_scores = xgb_model.predict_proba(X.iloc[test])[:, 1]
    out["xgboost_baseline"] = {"model": "xgboost_baseline", "test": report(y[test], xgb_scores, prior[test], ring[test])}

    # GraphSAGE, full batch.
    t = lambda cols, dtype: torch.tensor(seq[cols].to_numpy(dtype), device=device)
    x = (t(meta["cat_cols"], np.int64), t(meta["num_cols"], np.float32), t(meta["isna_cols"], np.float32))
    edge_index = torch.tensor(np.stack([edges["src"], edges["dst"]]), dtype=torch.long, device=device)
    torch.manual_seed(0)
    gnn = FraudSAGE([meta["cat_cardinality"][c] for c in meta["cat_cols"]], len(meta["num_cols"])).to(device)
    (art / "models").mkdir(exist_ok=True)
    hist = train_gnn(gnn, x, edge_index, y, train, val, epochs=gnn_epochs, ckpt=art / "models" / "graphsage.pt")
    gnn_scores = predict_gnn(gnn, x, edge_index)
    out["graphsage"] = {"model": "graphsage", "history": hist,
                        "test": report(y[test], gnn_scores[test], prior[test], ring[test])}

    # Autoencoder on legitimate training transactions only.
    num_cols = [c for c in X.columns if c not in cat_cols]
    normal = train[y[train] == 0]
    xm = numeric_matrix(X, num_cols, normal)
    ae, ae_hist = train_autoencoder(xm, normal, device, epochs=ae_epochs)
    torch.save(ae.state_dict(), art / "models" / "autoencoder.pt")
    ae_scores = anomaly_score(ae, xm, test, device)
    out["autoencoder"] = {"model": "autoencoder", "history": ae_hist,
                          "test": report(y[test], ae_scores, prior[test], ring[test])}

    for name in ("xgboost_baseline", "graphsage", "autoencoder"):
        append_results_row(out[name], art / "results.csv")
        print(name, json.dumps(out[name]["test"], indent=1))
    np.savez_compressed(art / "week7_test_scores.npz", row=test, xgboost=xgb_scores,
                        graphsage=gnn_scores[test], autoencoder=ae_scores)  # for the week 8 ensemble
    (art / "week7_results.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()
    run(Path(args.artifacts))
    print(pd.read_csv(Path(args.artifacts) / "results.csv").to_string(index=False))


if __name__ == "__main__":
    main()
