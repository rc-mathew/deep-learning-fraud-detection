"""Reason codes for the transformer with Captum Integrated Gradients (week 8).

Integrated Gradients measures how much each input moves the fraud score, relative to a
neutral baseline (all-zero embeddings and numbers). Attributions are summed over the
card's history window, giving one value per field: positive pushes towards fraud.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients

from .transformer import SeqData, TxnTransformer


def field_attributions(model: TxnTransformer, data: SeqData, rows: np.ndarray,
                       batch_size: int = 64, steps: int = 32) -> pd.DataFrame:
    """One row per transaction, one column per input field: attribution to the fraud logit."""
    model.eval()
    names = [c.removeprefix("cat_") for c in data.cat_cols] + [c.removeprefix("num_") for c in data.num_cols]
    out = []
    for i in range(0, len(rows), batch_size):
        cat, num, isna, pad = data.batch(rows[i:i + batch_size])
        fe = model.field_embeddings(cat).detach()

        def logit(fe_in, num_in):
            # Captum repeats the batch once per integration step; expand the fixed inputs to match.
            k = fe_in.shape[0] // cat.shape[0]
            h, _ = model.encode(cat.repeat(k, 1, 1), num_in, isna.repeat(k, 1, 1), pad.repeat(k, 1),
                                field_embs=fe_in)
            return model.fraud_head(h[:, -1]).squeeze(-1)

        ig = IntegratedGradients(logit)
        a_fe, a_num = ig.attribute((fe, num), baselines=(torch.zeros_like(fe), torch.zeros_like(num)),
                                   n_steps=steps)
        a = torch.cat([a_fe.sum(dim=(1, 3)), a_num.sum(dim=1)], dim=1)  # (B, fields)
        out.append(a.detach().cpu().numpy())
    return pd.DataFrame(np.concatenate(out), columns=names)


def reason_codes(attr: pd.DataFrame, k: int = 3) -> list[list[str]]:
    """Top-k fields pushing each transaction towards fraud."""
    vals = attr.to_numpy()
    order = np.argsort(-vals, axis=1)[:, :k]
    return [[attr.columns[j] for j in row if vals[i, j] > 0] for i, row in enumerate(order)]
