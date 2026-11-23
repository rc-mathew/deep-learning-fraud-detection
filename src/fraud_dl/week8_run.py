"""Week 8: every model in one run, then the stacked ensemble, calibration, cost threshold,
reason codes and fairness, all judged on the same test month.

    python -m fraud_dl.week8_run --artifacts artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .baseline import append_results_row
from .ensemble import (FALSE_POSITIVE_COST, ablation, best_threshold, calibration, cost, fairness,
                       fit_stack, predict_stack)
from .explain import field_attributions, reason_codes
from .graph import build_edges, ring_mask
from .metrics import card_had_prior_fraud, fraud_report, threshold_at_fpr
from .transformer import SeqData, TxnTransformer, finetune, predict
from .week7_run import report, run as run_week7


def run(art: Path, gnn_epochs: int = 300, transformer_epochs: int = 10, device: str | None = None,
        n_explain: int = 200) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    week7 = run_week7(art, gnn_epochs=gnn_epochs, device=device)
    sc = np.load(art / "week7_scores.npz")
    df = pd.read_parquet(art / "transactions.parquet")
    seq = pd.read_parquet(art / "sequences" / "tokens.parquet")
    meta = json.loads((art / "sequences" / "meta.json").read_text())

    # Transformer from scratch (pretraining did not help on the test month in weeks 5-6).
    data = SeqData(seq, meta, device)
    train, val, test = data.rows("train"), data.rows("val"), data.rows("test")
    torch.manual_seed(0)
    tf = TxnTransformer(data.cardinality, len(data.num_cols), data.seq_len).to(device)
    finetune(tf, data, train, val, transformer_epochs, art / "models" / "transformer_scratch.pt")
    all_rows = np.arange(len(seq))

    scores = {"xgboost": sc["xgboost"], "transformer": predict(tf, data, all_rows),
              "graphsage": sc["graphsage"], "autoencoder": sc["autoencoder"]}
    y = data.y
    prior = card_had_prior_fraud(y, data.group_start)
    ring = ring_mask(df, build_edges(df))

    # Stack: fitted on validation, judged on test.
    stack = fit_stack(scores, y, val)
    p_val, p_test = predict_stack(stack, scores, val), predict_stack(stack, scores, test)
    out: dict = {"stack_weights": dict(zip(scores, np.round(stack.coef_[0], 3).tolist())),
                 "ensemble": {"model": "ensemble", "test": report(y[test], p_test, prior[test], ring[test])}}
    ranks = np.mean([pd.Series(s[test]).rank(pct=True).to_numpy() for s in scores.values()], axis=0)
    out["rank_average"] = {"model": "rank_average", "test": report(y[test], ranks, prior[test], ring[test])}
    abl = ablation(scores, y, val, test, prior)
    abl.to_csv(art / "ablation.csv", index=False)

    # Calibration of the ensemble vs raw XGBoost.
    cal, ece, brier = calibration(y[test], p_test)
    cal.to_csv(art / "calibration.csv", index=False)
    _, ece_x, brier_x = calibration(y[test], scores["xgboost"][test])
    out["calibration"] = {"ensemble_ece": ece, "ensemble_brier": brier, "xgboost_ece": ece_x, "xgboost_brier": brier_x}

    # Cost-based threshold, chosen on validation.
    amt = df["TransactionAmt"].to_numpy()
    thr = best_threshold(y[val], p_val, amt[val])
    flag = p_test > thr
    thr_1pct = threshold_at_fpr(y[test], p_test, 0.01)
    out["cost"] = {
        "false_positive_cost": FALSE_POSITIVE_COST, "threshold": thr,
        "test_cost_at_threshold": cost(y[test], p_test, amt[test], thr),
        "test_cost_at_1pct_fpr": cost(y[test], p_test, amt[test], thr_1pct),
        "test_cost_block_nothing": float(amt[test][y[test] == 1].sum()),
        "recall_at_threshold": float(flag[y[test] == 1].mean()),
        "fpr_at_threshold": float(flag[y[test] == 0].mean()),
        "fraud_amount_caught_share": float(amt[test][(y[test] == 1) & flag].sum() / amt[test][y[test] == 1].sum()),
    }

    # Fairness at the chosen threshold.
    fair = fairness(df.iloc[test].reset_index(drop=True), y[test], flag)
    fair.to_csv(art / "fairness.csv", index=False)

    # Reason codes for the highest-scoring test transactions.
    top = test[np.argsort(-p_test)[:n_explain]]
    attr = field_attributions(tf, data, top)
    codes = reason_codes(attr)
    pd.DataFrame({"TransactionID": seq["TransactionID"].to_numpy()[top], "ensemble_score": np.sort(p_test)[::-1][:n_explain],
                  "isFraud": y[top], "reasons": ["; ".join(c) for c in codes]}).to_csv(art / "reason_codes.csv", index=False)
    out["global_field_importance"] = attr.abs().mean().sort_values(ascending=False).head(10).round(4).to_dict()

    for name in ("ensemble", "rank_average"):
        append_results_row(out[name], art / "results.csv")
    out["week7_graph"] = {k: week7[k] for k in ("n_edges", "edges_by_type", "ring_share_of_test")}
    (art / "week8_results.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k not in ("ensemble", "rank_average")}, indent=1, default=float))
    print(abl.round(4).to_string(index=False))
    print(fair.round(4).to_string(index=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()
    run(Path(args.artifacts))
    print(pd.read_csv(Path(args.artifacts) / "results.csv").to_string(index=False))


if __name__ == "__main__":
    main()
