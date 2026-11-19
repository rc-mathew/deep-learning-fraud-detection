import json

import numpy as np
import pandas as pd
import pytest

from fraud_dl import config
from fraud_dl.baseline import append_results_row, feature_frame, train_baseline
from fraud_dl.data import add_time_columns, add_uid, load_raw, sort_by_card_and_time, time_split
from fraud_dl.features import add_velocity_features
from fraud_dl.metrics import recall_at_fpr, threshold_at_fpr
from fraud_dl.prepare import prepare
from fraud_dl.sequences import window_indices

from synthetic import write_raw


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    d = write_raw(tmp_path_factory.mktemp("raw"), n_cards=400)
    return load_raw(d)


@pytest.fixture(scope="module")
def prepared(raw, tmp_path_factory):
    out = tmp_path_factory.mktemp("art")
    report = prepare(raw.copy(), out)
    return out, report


def _pipeline(df):
    df = add_time_columns(df.copy())
    df = add_uid(df)
    df = time_split(df)
    df = sort_by_card_and_time(df)
    return add_velocity_features(df)


def test_identity_columns_renamed(raw):
    assert {"id_30", "id_31", "id_33"} <= set(raw.columns)


def test_time_split_is_ordered(raw):
    df = time_split(add_time_columns(raw.copy()), test_days=30, val_days=30, gap_days=7)
    days = {s: (g["day"].min(), g["day"].max()) for s, g in df.groupby("split")}
    assert days["train"][1] < days["gap"][0] <= days["gap"][1] < days["val"][0]
    assert days["val"][1] < days["test"][0]
    assert days["test"][1] - days["test"][0] == 29


def test_uid_is_stable_within_a_card(raw):
    df = add_uid(add_time_columns(raw.copy()))
    # The synthetic generator gives each card a fixed card1/addr1/first day.
    assert df["uid"].nunique() <= 400
    assert df.groupby("uid")["card1"].nunique().max() == 1


def test_velocity_uses_only_the_past(raw):
    df = _pipeline(raw)
    # Inflate every amount in the last 10 days; earlier features must not change.
    later = raw.copy()
    cutoff = later["TransactionDT"].max() - 10 * 86_400
    later.loc[later["TransactionDT"] > cutoff, "TransactionAmt"] *= 100
    df2 = _pipeline(later)
    early = df["TransactionDT"] <= cutoff
    cols = [c for c in df.columns if c.startswith(("n_txn", "amt_sum", "amt_prior", "secs_since"))]
    pd.testing.assert_frame_equal(df.loc[early, cols], df2.loc[early, cols])


def test_velocity_matches_brute_force(raw):
    df = _pipeline(raw)
    rng = np.random.default_rng(1)
    for i in rng.choice(len(df), 50, replace=False):
        r = df.iloc[i]
        same = df[(df["uid"] == r["uid"]) & (df.index < i)]
        in_24h = same[same["TransactionDT"] >= r["TransactionDT"] - 86_400]
        assert r["n_txn_24h"] == len(in_24h)
        assert r["amt_sum_24h"] == pytest.approx(in_24h["TransactionAmt"].sum(), rel=1e-4)
        assert r["n_txn_prior"] == len(same)


def test_window_indices():
    group_start = np.array([0, 0, 0, 3, 3, 5])
    w = window_indices(np.array([2, 4, 5]), group_start, seq_len=3)
    np.testing.assert_array_equal(w, [[0, 1, 2], [-1, 3, 4], [-1, -1, 5]])


def test_windows_never_cross_cards(prepared):
    out, _ = prepared
    seq = pd.read_parquet(out / "sequences" / "tokens.parquet")
    w = window_indices(np.arange(len(seq)), seq["group_start"].to_numpy(), config.SEQ_LEN)
    uid = seq["uid_code"].to_numpy()
    valid = w >= 0
    assert (uid[w[valid]] == np.repeat(uid, valid.sum(1))).all()
    assert (w[:, -1] == np.arange(len(seq))).all()


def test_sequence_table_has_no_label_leak(prepared):
    out, _ = prepared
    meta = json.loads((out / "sequences" / "meta.json").read_text())
    inputs = meta["cat_cols"] + meta["num_cols"] + meta["isna_cols"]
    assert not any("isFraud" in c for c in inputs)
    seq = pd.read_parquet(out / "sequences" / "tokens.parquet", columns=meta["cat_cols"])
    for col, card in meta["cat_cardinality"].items():
        assert seq[col].between(1, card - 1).all()


def test_features_exclude_ids_and_time(prepared):
    out, _ = prepared
    df = pd.read_parquet(out / "transactions.parquet")
    X, cat_cols = feature_frame(df)
    for banned in ["TransactionID", "TransactionDT", "isFraud", "uid", "day", "uid_code"]:
        assert banned not in X.columns
    assert "ProductCD" in cat_cols and X["ProductCD"].dtype == np.float32


def test_recall_at_fpr():
    y = np.array([0] * 100 + [1] * 10)
    s = np.concatenate([np.linspace(0, 0.9, 100), np.linspace(0.5, 1.0, 10)])
    thr = threshold_at_fpr(y, s, 0.01)
    assert (s[y == 0] > thr).sum() <= 1
    assert recall_at_fpr(y, s, 0.01) == pytest.approx(np.mean(s[y == 1] > thr))


def test_baseline_end_to_end(prepared, tmp_path):
    out, report = prepared
    assert set(report["splits"]) == {"train", "val", "test"}
    df = pd.read_parquet(out / "transactions.parquet")
    _, results = train_baseline(df)
    assert 0 <= results["test"]["recall_at_1pct_fpr"] <= 1
    # The synthetic fraud is learnable, so the baseline should beat random ranking.
    assert results["test"]["roc_auc"] > 0.6
    path = tmp_path / "results.csv"
    append_results_row(results, path)
    append_results_row(results, path)
    assert len(pd.read_csv(path)) == 1


def test_load_raw_finds_nested_kaggle_mount(tmp_path):
    write_raw(tmp_path / "input" / "competitions" / "ieee-fraud-detection", n_cards=5)
    df = load_raw(tmp_path / "input" / "ieee-fraud-detection")  # the old, missing path
    assert len(df) > 0
