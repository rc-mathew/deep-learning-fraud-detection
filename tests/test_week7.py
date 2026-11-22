import json

import numpy as np
import pandas as pd
import pytest
import torch

from fraud_dl.data import load_raw
from fraud_dl.graph import PREV_K, FraudSAGE, build_edges, entity_key, predict_gnn, ring_mask
from fraud_dl.prepare import prepare
from fraud_dl.week7_run import run

from synthetic import write_raw


@pytest.fixture(scope="module")
def art(tmp_path_factory):
    raw = load_raw(write_raw(tmp_path_factory.mktemp("raw"), n_cards=300))
    out = tmp_path_factory.mktemp("art")
    prepare(raw, out)
    return out


@pytest.fixture(scope="module")
def df(art):
    return pd.read_parquet(art / "transactions.parquet")


def test_edges_point_forward_in_time_and_share_an_entity(df):
    e = build_edges(df)
    t = df["TransactionDT"].to_numpy()
    assert len(e["src"]) > 0
    assert (t[e["src"]] <= t[e["dst"]]).all()
    assert (e["src"] != e["dst"]).all()
    keys = {i: entity_key(df, cols).to_numpy() for i, cols in enumerate(
        [["uid"], ["DeviceInfo", "id_30", "id_31", "id_33", "id_19", "id_20"], ["P_emaildomain", "addr1"]])}
    for et, key in keys.items():
        m = e["etype"] == et
        assert (key[e["src"][m]] == key[e["dst"][m]]).all()


def test_in_degree_capped_per_entity(df):
    e = build_edges(df)
    for et in np.unique(e["etype"]):
        assert np.bincount(e["dst"][e["etype"] == et]).max() <= PREV_K


def test_missing_entity_gets_no_edges(df):
    e = build_edges(df)
    no_device = df["DeviceInfo"].isna().to_numpy()
    dev = e["etype"] == 1
    assert not no_device[e["dst"][dev]].any()


def test_ring_mask_is_cross_card_device_links(df):
    e = build_edges(df)
    ring = ring_mask(df, e)
    uid = df["uid_code"].to_numpy()
    for r in np.flatnonzero(ring)[:20]:
        m = (e["etype"] == 1) & (e["dst"] == r)
        assert (uid[e["src"][m]] != uid[r]).any()


def test_gnn_score_ignores_later_transactions(art, df):
    meta = json.loads((art / "sequences" / "meta.json").read_text())
    seq = pd.read_parquet(art / "sequences" / "tokens.parquet")
    x = [torch.tensor(seq[meta[k]].to_numpy(dt)) for k, dt in
         [("cat_cols", np.int64), ("num_cols", np.float32), ("isna_cols", np.float32)]]
    e = build_edges(df)
    ei = torch.tensor(np.stack([e["src"], e["dst"]]))
    torch.manual_seed(0)
    model = FraudSAGE([meta["cat_cardinality"][c] for c in meta["cat_cols"]], len(meta["num_cols"]), d=16)
    before = predict_gnn(model, tuple(x), ei)
    late = df["TransactionDT"].to_numpy() > np.quantile(df["TransactionDT"], 0.8)
    x[1][torch.from_numpy(late)] += 50.0
    after = predict_gnn(model, tuple(x), ei)
    np.testing.assert_allclose(before[~late], after[~late], atol=1e-5)


def test_run_end_to_end(art):
    out = run(art, gnn_epochs=10, ae_epochs=2, device="cpu", use_gpu_xgb=False)
    for name in ("xgboost_baseline", "graphsage", "autoencoder"):
        t = out[name]["test"]
        assert 0 <= t["recall_at_1pct_fpr"] <= 1 and "ring_recall_at_1pct_fpr" in t
    assert (art / "graph" / "edges.npz").exists() and (art / "week7_test_scores.npz").exists()
    res = pd.read_csv(art / "results.csv")
    assert {"graphsage", "autoencoder"} <= set(res["model"]) and "ring_recall_at_1pct_fpr" in res
