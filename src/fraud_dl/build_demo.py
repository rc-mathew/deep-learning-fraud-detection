"""Build the demo bundle served by the API and the Streamlit app (run on Kaggle after week 8 and 8b).

    python -m fraud_dl.build_demo --artifacts artifacts --out demo

The bundle is a sample of test-month transactions with everything the demo shows:
- each transaction's 16-step card window, so the transformer scores it live with ONNX Runtime;
- the week 8 ensemble score and decision (XGBoost and GraphSAGE need the full feature table and
  graph, so their part is precomputed here);
- Captum reason codes and the 3 most similar past fraud cases.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .ensemble import best_threshold, fit_stack, predict_stack
from .explain import field_attributions, reason_codes
from .optimize import CaseIndex, case_embeddings
from .transformer import SeqData, TxnTransformer, predict


def build(art: Path, out: Path, n_fraud: int = 300, n_legit: int = 700, seed: int = 0) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((art / "sequences" / "meta.json").read_text())
    seq = pd.read_parquet(art / "sequences" / "tokens.parquet")
    data = SeqData(seq, meta, "cpu")
    model = TxnTransformer(data.cardinality, len(data.num_cols), data.seq_len)
    model.load_state_dict(torch.load(art / "models" / "transformer_scratch.pt", map_location="cpu"))
    model.eval()

    # Ensemble scores and the cost threshold, rebuilt exactly as week 8 did.
    sc = np.load(art / "week7_scores.npz")
    every = np.arange(len(seq))
    scores = {"xgboost": sc["xgboost"], "transformer": predict(model, data, every),
              "graphsage": sc["graphsage"], "autoencoder": sc["autoencoder"]}
    val, test = data.rows("val"), data.rows("test")
    stack = fit_stack(scores, data.y, val)
    tx = pd.read_parquet(art / "transactions.parquet",
                         columns=["TransactionID", "TransactionAmt", "ProductCD", "card4", "card6"])
    amt = tx["TransactionAmt"].to_numpy()
    threshold = best_threshold(data.y[val], predict_stack(stack, scores, val), amt[val])

    rng = np.random.default_rng(seed)
    fraud, legit = test[data.y[test] == 1], test[data.y[test] == 0]
    rows = np.sort(np.concatenate([rng.choice(fraud, n_fraud, replace=False),
                                   rng.choice(legit, n_legit, replace=False)]))

    cat, num, isna, pad = (t.numpy() for t in data.batch(rows))
    np.savez_compressed(out / "windows.npz", cat=cat.astype(np.int32), num=num, isna=isna, pad=pad)

    codes = reason_codes(field_attributions(model, data, rows))
    past = data.rows("train", "val")
    past_fraud = past[data.y[past] == 1]
    ids = seq["TransactionID"].to_numpy()
    index = CaseIndex(case_embeddings(model, data, past_fraud), ids[past_fraud])
    sim_ids, sim = index.search(case_embeddings(model, data, rows), k=3)

    ens = predict_stack(stack, scores, rows)
    pd.DataFrame({
        "TransactionID": ids[rows], "TransactionAmt": amt[rows],
        "ProductCD": tx["ProductCD"].to_numpy()[rows], "card4": tx["card4"].astype(str).to_numpy()[rows],
        "card6": tx["card6"].astype(str).to_numpy()[rows], "isFraud": data.y[rows],
        "ensemble_score": ens, "xgboost_score": scores["xgboost"][rows],
        "reasons": ["; ".join(c) for c in codes],
        "similar_cases": [";".join(map(str, r)) for r in sim_ids],
        "similarity": [";".join(f"{v:.3f}" for v in r) for r in sim],
    }).to_csv(out / "transactions.csv", index=False)

    shutil.copy(art / "onnx" / "transformer_fp32.onnx", out / "transformer_fp32.onnx")
    (out / "config.json").write_text(json.dumps({
        "threshold": threshold, "false_positive_cost": 10.0, "seq_len": data.seq_len,
        "note": "Test-month sample. Transformer scored live with ONNX Runtime; ensemble, reasons and "
                "similar cases precomputed from the week 8 run."}, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="demo")
    args = ap.parse_args()
    print(build(Path(args.artifacts), Path(args.out)))


if __name__ == "__main__":
    main()
