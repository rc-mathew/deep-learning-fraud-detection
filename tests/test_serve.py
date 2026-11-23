import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraud_dl.build_demo import build
from fraud_dl.data import load_raw
from fraud_dl.optimize_run import run as run_optimize
from fraud_dl.prepare import prepare
from fraud_dl.serve import Scorer
from fraud_dl.week8_run import run as run_week8

from synthetic import write_raw


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    d = tmp_path_factory.mktemp("demo")
    prepare(load_raw(write_raw(d / "raw", n_cards=300)), d / "art")
    run_week8(d / "art", gnn_epochs=5, transformer_epochs=1, device="cpu", n_explain=4)
    run_optimize(d / "art")
    return build(d / "art", d / "demo", n_fraud=10, n_legit=20)


def test_live_onnx_score_matches_bundle_shapes(demo):
    s = Scorer(demo)
    assert len(s.transaction_ids()) == 30
    r = s.score(s.transaction_ids()[0])
    assert 0 <= r["transformer_score"] <= 1 and r["decision"] in ("block", "allow")
    assert len(r["similar_past_fraud"]) == 3 and r["transformer_latency_ms"] > 0


def test_decision_follows_threshold(demo):
    s = Scorer(demo)
    for t in s.transaction_ids():
        r = s.score(t)
        assert (r["decision"] == "block") == (r["ensemble_score"] > r["threshold"])


def test_api(demo, monkeypatch):
    import fraud_dl.serve as serve
    monkeypatch.setattr(serve, "DEMO_DIR", demo)
    import importlib, sys
    sys.modules.pop("app.api", None)
    api = importlib.import_module("app.api")
    c = TestClient(api.app)
    assert c.get("/health").json()["transactions"] == 30
    tid = c.get("/transactions?limit=1").json()[0]
    assert c.post("/score", json={"transaction_id": tid}).json()["transaction_id"] == tid
    assert c.post("/score", json={"transaction_id": -1}).status_code == 404
    assert c.post("/score", json={"transaction_id": "x"}).status_code == 422
