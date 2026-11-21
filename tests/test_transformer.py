import json

import numpy as np
import pandas as pd
import pytest
import torch

from fraud_dl.baseline import train_baseline
from fraud_dl.data import load_raw
from fraud_dl.metrics import card_had_prior_fraud
from fraud_dl.prepare import prepare
from fraud_dl.transformer import SeqData, TxnTransformer
from fraud_dl.transformer_run import run

from synthetic import write_raw


@pytest.fixture(scope="module")
def art(tmp_path_factory):
    raw = load_raw(write_raw(tmp_path_factory.mktemp("raw"), n_cards=300))
    out = tmp_path_factory.mktemp("art")
    prepare(raw, out)
    return out


@pytest.fixture(scope="module")
def data(art):
    meta = json.loads((art / "sequences" / "meta.json").read_text())
    return SeqData(pd.read_parquet(art / "sequences" / "tokens.parquet"), meta)


def _model(data):
    torch.manual_seed(0)
    return TxnTransformer(data.cardinality, len(data.num_cols), data.seq_len, d_model=16, n_heads=2)


def test_padding_gets_no_attention(data):
    rows = np.flatnonzero(data.group_start == np.arange(len(data.y)))[:8]  # cards' first transactions
    cat, num, isna, pad = data.batch(rows)
    assert pad[:, :-1].all() and not pad[:, -1].any()
    _, attns = _model(data).encode(cat, num, isna, pad)
    assert torch.allclose(attns[0][..., :-1], torch.zeros(1))


def test_score_ignores_padding_values(data):
    model = _model(data).eval()
    rows = np.arange(16)
    batch = list(data.batch(rows))
    before = model(*batch)
    batch[1] = batch[1].masked_fill(batch[3][..., None], 123.0)  # junk in padded slots
    assert torch.allclose(before, model(*batch), atol=1e-5)


def test_score_ignores_future_transactions(data):
    """A row's window holds only its own and earlier transactions of the card."""
    model = _model(data).eval()
    rows = np.arange(len(data.y) - 1)
    rows = rows[data.group_start[rows + 1] == data.group_start[rows]]  # rows with a later txn
    rows = rows[~np.isin(rows, rows + 1)][:20]  # the edited "future" rows are not scored themselves
    before = model(*data.batch(rows))
    data.num[rows + 1] += 50.0
    try:
        assert torch.allclose(before, model(*data.batch(rows)), atol=1e-5)
    finally:
        data.num[rows + 1] -= 50.0


def test_pretrain_loss_learns(data):
    model = _model(data)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    batch = data.batch(data.rows("train")[:256])
    first = None
    for _ in range(60):
        torch.manual_seed(1)  # same mask each step, so the loss must fall
        loss, _, _ = model.pretrain_loss(*batch)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.7 * first


def test_card_had_prior_fraud():
    y = np.array([0, 1, 0, 0, 1, 0])
    gs = np.array([0, 0, 0, 3, 3, 3])
    np.testing.assert_array_equal(card_had_prior_fraud(y, gs), [0, 0, 1, 0, 0, 1])


def test_run_end_to_end(art):
    train_baseline(pd.read_parquet(art / "transactions.parquet"))
    out = run(art, pretrain_epochs=2, finetune_epochs=2, d_model=16, batch_size=128, device="cpu")
    assert len(out["pretrain_history"]) == 2
    assert (art / "models" / "transformer_pretrain.pt").exists()
    for name in ["transformer_scratch", "transformer_pretrained"]:
        t = out[name]["test"]
        assert 0 <= t["recall_at_1pct_fpr"] <= 1 and "first_fraud_recall_at_1pct_fpr" in t
    assert set(pd.read_csv(art / "results.csv")["model"]) == {"transformer_scratch", "transformer_pretrained"}
    # Resume: a rerun with the same epochs loads the checkpoint and trains nothing new.
    again = run(art, pretrain_epochs=2, finetune_epochs=1, d_model=16, batch_size=128, device="cpu")
    assert again["pretrain_history"] == out["pretrain_history"]
