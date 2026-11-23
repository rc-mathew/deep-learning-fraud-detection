import json

import numpy as np
import pandas as pd
import pytest
import torch

from fraud_dl.data import load_raw
from fraud_dl.optimize import CaseIndex, case_embeddings, compare, export_onnx, onnx_scores, session
from fraud_dl.prepare import prepare
from fraud_dl.transformer import SeqData, TxnTransformer

from synthetic import write_raw


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("opt")
    prepare(load_raw(write_raw(d / "raw", n_cards=200)), d / "art")
    meta = json.loads((d / "art" / "sequences" / "meta.json").read_text())
    data = SeqData(pd.read_parquet(d / "art" / "sequences" / "tokens.parquet"), meta)
    torch.manual_seed(0)
    model = TxnTransformer(data.cardinality, len(data.num_cols), data.seq_len, d_model=16, n_heads=2).eval()
    return d, data, model


def test_onnx_matches_pytorch(setup):
    d, data, model = setup
    path = export_onnx(model, data, d / "m.onnx")
    rows = np.arange(64)
    with torch.no_grad():
        ref = torch.sigmoid(model(*data.batch(rows))).numpy()
    np.testing.assert_allclose(onnx_scores(session(path), data, rows), ref, atol=1e-5)


def test_int8_is_smaller_and_close(setup):
    d, data, _ = setup
    torch.manual_seed(0)  # default size: tiny tensors are left unquantised by ONNX Runtime
    model = TxnTransformer(data.cardinality, len(data.num_cols), data.seq_len).eval()
    res = compare(model, data, d / "onnx", data.rows("test"))
    assert res["onnx_int8"]["size_mb"] < res["onnx_fp32"]["size_mb"]
    # The largest single change varies by CPU; the average change is the stable check.
    assert res["int8_mean_score_diff"] < 0.02
    assert res["onnx_int8"]["p95_ms"] > 0


def test_case_index_finds_itself(setup, tmp_path):
    _, data, model = setup
    rows = np.arange(50)
    emb = case_embeddings(model, data, rows)
    idx = CaseIndex(emb, rows + 1000)
    ids, sim = idx.search(emb[:5], k=3)
    assert (ids[:, 0] == rows[:5] + 1000).all() and np.allclose(sim[:, 0], 1, atol=1e-5)
    idx.save(tmp_path / "cases.faiss")
    ids2, _ = CaseIndex.load(tmp_path / "cases.faiss").search(emb[:5], k=3)
    assert (ids2 == ids).all()


def test_optimize_run_end_to_end(setup):
    from fraud_dl.optimize_run import run
    d, data, _ = setup
    torch.manual_seed(0)
    model = TxnTransformer(data.cardinality, len(data.num_cols), data.seq_len)
    (d / "art" / "models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), d / "art" / "models" / "transformer_scratch.pt")
    out = run(d / "art")
    assert out["faiss"]["indexed_fraud_cases"] > 0
    assert (d / "art" / "onnx" / "transformer_int8.onnx").exists()
