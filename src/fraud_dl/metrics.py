"""Fraud metrics shared by every model, so the results table compares like with like."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr: float = 0.01) -> float:
    """Score above which at most `fpr` of legitimate transactions are flagged."""
    y_true = np.asarray(y_true)
    neg = np.sort(np.asarray(scores)[y_true == 0])[::-1]
    if len(neg) == 0:
        raise ValueError("No negatives to set a threshold on.")
    k = int(np.floor(fpr * len(neg)))
    return float(neg[min(k, len(neg) - 1)])


def recall_at_fpr(y_true: np.ndarray, scores: np.ndarray, fpr: float = 0.01) -> float:
    """Share of fraud caught when flagging score > threshold_at_fpr()."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    thr = threshold_at_fpr(y_true, scores, fpr)
    return float(np.mean(scores[y_true == 1] > thr))


def fraud_report(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    return {
        "n": int(len(y_true)),
        "fraud_rate": float(y_true.mean()),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, scores, 0.01),
        "recall_at_5pct_fpr": recall_at_fpr(y_true, scores, 0.05),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "threshold_at_1pct_fpr": threshold_at_fpr(y_true, scores, 0.01),
    }
