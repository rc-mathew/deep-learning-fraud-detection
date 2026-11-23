"""Week 8: stack the models, check calibration, pick a cost-based threshold, check fairness."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .metrics import fraud_report

# Assumed cost of wrongly blocking a good customer (support call, lost sale). A missed fraud
# costs its transaction amount. Change it and the chosen threshold moves.
FALSE_POSITIVE_COST = 10.0


def stack_features(scores: dict[str, np.ndarray]) -> np.ndarray:
    """Probabilities as logits, the autoencoder's error as log; one column per model."""
    cols = []
    for name, s in scores.items():
        s = np.asarray(s, dtype=np.float64)
        if name == "autoencoder":
            cols.append(np.log(s + 1e-9))
        else:
            p = np.clip(s, 1e-6, 1 - 1e-6)
            cols.append(np.log(p / (1 - p)))
    return np.column_stack(cols)


def fit_stack(scores: dict[str, np.ndarray], y: np.ndarray, fit_rows: np.ndarray) -> LogisticRegression:
    """Logistic regression on validation scores: learns the blend weights and calibrates in one step."""
    x = stack_features(scores)
    mean, std = x[fit_rows].mean(0), x[fit_rows].std(0) + 1e-9
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit((x[fit_rows] - mean) / std, y[fit_rows])
    model.scale_ = (mean, std)
    return model


def predict_stack(model: LogisticRegression, scores: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    mean, std = model.scale_
    return model.predict_proba((stack_features(scores)[rows] - mean) / std)[:, 1]


def ablation(scores: dict[str, np.ndarray], y, val, test, prior) -> pd.DataFrame:
    """Test metrics of the stack with each model left out: what each model adds."""
    rows = []
    for drop in [None, *scores]:
        sub = {k: v for k, v in scores.items() if k != drop}
        rep = fraud_report(y[test], predict_stack(fit_stack(sub, y, val), sub, test), prior[test])
        rows.append({"left_out": drop or "(none)", "recall_at_1pct_fpr": rep["recall_at_1pct_fpr"],
                     "pr_auc": rep["pr_auc"], "first_fraud_recall_at_1pct_fpr": rep["first_fraud_recall_at_1pct_fpr"]})
    return pd.DataFrame(rows)


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[pd.DataFrame, float, float]:
    """Reliability table by predicted-probability decile, expected calibration error and Brier score."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    table = pd.DataFrame({"bin": idx, "p": p, "y": y}).groupby("bin").agg(
        n=("y", "size"), mean_predicted=("p", "mean"), fraud_rate=("y", "mean"))
    ece = float((table["n"] * (table["mean_predicted"] - table["fraud_rate"]).abs()).sum() / len(y))
    return table.reset_index(), ece, float(np.mean((p - y) ** 2))


def cost(y, p, amount, thr, fp_cost: float = FALSE_POSITIVE_COST) -> float:
    flag = p > thr
    return float(amount[(y == 1) & ~flag].sum() + fp_cost * ((y == 0) & flag).sum())


def best_threshold(y, p, amount, fp_cost: float = FALSE_POSITIVE_COST) -> float:
    """Threshold with the lowest total cost, searched over score quantiles."""
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.9999, 400)))
    costs = [cost(y, p, amount, t, fp_cost) for t in grid]
    return float(grid[int(np.argmin(costs))])


def fairness(df: pd.DataFrame, y: np.ndarray, flag: np.ndarray) -> pd.DataFrame:
    """Recall and false-positive rate by card network, card type, product and country code."""
    region = df["addr2"].astype("Int64").astype(str)
    top = region.value_counts().index[:3]
    groups = {"card4": df["card4"].astype(str), "card6": df["card6"].astype(str),
              "ProductCD": df["ProductCD"].astype(str),
              "addr2 (country)": region.where(region.isin(top), "other")}
    overall_fpr = flag[y == 0].mean()
    out = []
    for col, vals in groups.items():
        for g in pd.unique(vals):
            m = (vals == g).to_numpy()
            if m.sum() < 200:
                continue
            fraud, legit = m & (y == 1), m & (y == 0)
            fpr = flag[legit].mean() if legit.any() else np.nan
            out.append({"attribute": col, "group": g, "n": int(m.sum()), "fraud_rate": y[m].mean(),
                        "recall": flag[fraud].mean() if fraud.any() else np.nan, "fpr": fpr,
                        "fpr_vs_overall": fpr / overall_fpr if overall_fpr else np.nan})
    return pd.DataFrame(out)
