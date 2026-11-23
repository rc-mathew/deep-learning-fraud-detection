import numpy as np
import pandas as pd
import pytest

from fraud_dl.data import load_raw
from fraud_dl.ensemble import best_threshold, calibration, cost, fit_stack, predict_stack
from fraud_dl.prepare import prepare
from fraud_dl.week8_run import run

from synthetic import write_raw


def test_stack_beats_noise_and_learns_weights():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.1).astype(int)
    good = np.clip(0.5 * y + rng.random(4000) * 0.5, 0, 1)
    noise = rng.random(4000)
    scores = {"good": good, "noise": noise}
    m = fit_stack(scores, y, np.arange(2000))
    assert abs(m.coef_[0][0]) > 5 * abs(m.coef_[0][1])
    p = predict_stack(m, scores, np.arange(2000, 4000))
    assert ((p > 0.5) == y[2000:]).mean() > 0.9


def test_calibration_of_perfectly_calibrated_scores():
    rng = np.random.default_rng(1)
    p = rng.random(50_000)
    y = (rng.random(50_000) < p).astype(int)
    _, ece, _ = calibration(y, p)
    assert ece < 0.02


def test_cost_threshold_prefers_cheaper_choice():
    y = np.array([1, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.1, 0.1])
    amt = np.array([1000.0, 5, 5, 5])
    # Blocking the 0.8 good customer costs 10, missing the fraud costs 1000.
    assert cost(y, p, amt, 0.85) == 0.0 and cost(y, p, amt, 0.95) == 1000.0
    assert best_threshold(y, p, amt) < 0.9


def test_run_end_to_end(tmp_path):
    raw = load_raw(write_raw(tmp_path / "raw", n_cards=300))
    prepare(raw, tmp_path / "art")
    out = run(tmp_path / "art", gnn_epochs=10, transformer_epochs=1, device="cpu", n_explain=8)
    assert set(out["stack_weights"]) == {"xgboost", "transformer", "graphsage", "autoencoder"}
    assert out["cost"]["test_cost_at_threshold"] <= out["cost"]["test_cost_block_nothing"] + 1e-6
    rc = pd.read_csv(tmp_path / "art" / "reason_codes.csv")
    assert len(rc) == 8
    for f in ("ablation.csv", "fairness.csv", "calibration.csv", "week8_results.json"):
        assert (tmp_path / "art" / f).exists()
