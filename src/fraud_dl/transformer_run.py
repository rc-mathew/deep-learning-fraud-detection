"""Weeks 5-6: transformer from scratch vs masked-pretrained transformer, same test set as XGBoost.

    python -m fraud_dl.transformer_run --artifacts artifacts
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from .baseline import append_results_row
from .metrics import card_had_prior_fraud, fraud_report
from .transformer import SeqData, TxnTransformer, finetune, predict, pretrain


def run(art: Path, pretrain_epochs: int = 10, finetune_epochs: int = 10, d_model: int = 64,
        n_layers: int = 2, batch_size: int = 512, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    meta = json.loads((art / "sequences" / "meta.json").read_text())
    seq = pd.read_parquet(art / "sequences" / "tokens.parquet")
    data = SeqData(seq, meta, device)
    prior = card_had_prior_fraud(data.y, data.group_start)
    train, val, test = data.rows("train"), data.rows("val"), data.rows("test")
    ckpt = art / "models"
    ckpt.mkdir(parents=True, exist_ok=True)

    def new_model():
        torch.manual_seed(0)
        return TxnTransformer(data.cardinality, len(data.num_cols), data.seq_len,
                              d_model=d_model, n_layers=n_layers).to(device)

    out: dict = {"device": device, "params": sum(p.numel() for p in new_model().parameters())}

    # Pretraining sees train + val transactions without labels; test stays unseen.
    t0 = time.time()
    pre = new_model()
    out["pretrain_history"] = pretrain(pre, data, data.rows("train", "val"), pretrain_epochs,
                                       ckpt / "transformer_pretrain.pt", batch_size)
    out["pretrain_seconds"] = round(time.time() - t0, 1)

    for name, model in [("transformer_scratch", new_model()), ("transformer_pretrained", pre)]:
        t0 = time.time()
        hist = finetune(model, data, train, val, finetune_epochs, ckpt / f"{name}.pt", batch_size)
        res = {"model": name, "finetune_history": hist, "train_seconds": round(time.time() - t0, 1)}
        for split, rows in [("val", val), ("test", test)]:
            res[split] = fraud_report(data.y[rows], predict(model, data, rows), prior[rows])
        out[name] = res
        append_results_row(res, art / "results.csv")
        print(name, json.dumps(res["test"], indent=1))

    (art / "transformer_results.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--pretrain-epochs", type=int, default=10)
    ap.add_argument("--finetune-epochs", type=int, default=10)
    args = ap.parse_args()
    run(Path(args.artifacts), args.pretrain_epochs, args.finetune_epochs)
    print(pd.read_csv(Path(args.artifacts) / "results.csv").to_string(index=False))


if __name__ == "__main__":
    main()
