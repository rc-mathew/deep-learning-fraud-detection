"""Stage 3: ONNX/INT8 comparison and the FAISS case index, from the week 8 run's artifacts.

    python -m fraud_dl.optimize_run --artifacts artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .optimize import CaseIndex, case_embeddings, compare
from .transformer import SeqData, TxnTransformer


def run(art: Path) -> dict:
    meta = json.loads((art / "sequences" / "meta.json").read_text())
    seq = pd.read_parquet(art / "sequences" / "tokens.parquet")
    data = SeqData(seq, meta, "cpu")
    model = TxnTransformer(data.cardinality, len(data.num_cols), data.seq_len)
    model.load_state_dict(torch.load(art / "models" / "transformer_scratch.pt", map_location="cpu"))
    model.eval()

    test = data.rows("test")
    out = {"onnx": compare(model, data, art / "onnx", test)}

    # Index past confirmed fraud (train + val) and look up the closest cases for test fraud.
    past = data.rows("train", "val")
    past_fraud = past[data.y[past] == 1]
    ids = seq["TransactionID"].to_numpy()
    index = CaseIndex(case_embeddings(model, data, past_fraud), ids[past_fraud])
    index.save(art / "onnx" / "fraud_cases.faiss")
    test_fraud, test_legit = test[data.y[test] == 1], test[data.y[test] == 0][:20000]
    _, sim_f = index.search(case_embeddings(model, data, test_fraud), k=1)
    _, sim_l = index.search(case_embeddings(model, data, test_legit), k=1)
    out["faiss"] = {"indexed_fraud_cases": int(len(past_fraud)),
                    "mean_top1_similarity_test_fraud": float(sim_f.mean()),
                    "mean_top1_similarity_test_legit": float(sim_l.mean())}
    (art / "optimize_results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    run(Path(ap.parse_args().artifacts))


if __name__ == "__main__":
    main()
