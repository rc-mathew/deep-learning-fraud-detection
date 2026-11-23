"""Scoring for the API and the demo. Needs only numpy, pandas and onnxruntime: no PyTorch."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

DEMO_DIR = Path(__file__).resolve().parents[2] / "demo"


class Scorer:
    def __init__(self, demo_dir: Path | None = None):
        demo_dir = Path(demo_dir or DEMO_DIR)
        self.config = json.loads((demo_dir / "config.json").read_text())
        self.tx = pd.read_csv(demo_dir / "transactions.csv")
        w = np.load(demo_dir / "windows.npz")
        self.windows = {k: w[k] for k in ("cat", "num", "isna", "pad")}
        self.windows["cat"] = self.windows["cat"].astype(np.int64)
        self.row = {int(t): i for i, t in enumerate(self.tx["TransactionID"])}
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(demo_dir / "transformer_fp32.onnx"), opts,
                                            providers=["CPUExecutionProvider"])

    def transaction_ids(self) -> list[int]:
        return [int(t) for t in self.tx["TransactionID"]]

    def score(self, transaction_id: int) -> dict:
        """Live transformer score plus the precomputed ensemble result for one transaction."""
        if transaction_id not in self.row:
            raise KeyError(transaction_id)
        i = self.row[transaction_id]
        feed = {k: v[i:i + 1] for k, v in self.windows.items()}
        t0 = time.perf_counter()
        logit = float(self.session.run(None, feed)[0][0])
        latency = (time.perf_counter() - t0) * 1000
        r = self.tx.iloc[i]
        ens = float(r["ensemble_score"])
        return {
            "transaction_id": transaction_id,
            "amount": float(r["TransactionAmt"]),
            "product": str(r["ProductCD"]), "card_network": str(r["card4"]), "card_type": str(r["card6"]),
            "transformer_score": float(1 / (1 + np.exp(-logit))),
            "transformer_latency_ms": round(latency, 3),
            "ensemble_score": ens,
            "decision": "block" if ens > self.config["threshold"] else "allow",
            "threshold": self.config["threshold"],
            "reasons": [s for s in str(r["reasons"]).split("; ") if s and s != "nan"],
            "similar_past_fraud": [
                {"transaction_id": int(t), "similarity": float(s)}
                for t, s in zip(str(r["similar_cases"]).split(";"), str(r["similarity"]).split(";"))],
            "actual_label": "fraud" if int(r["isFraud"]) else "legitimate",
        }
