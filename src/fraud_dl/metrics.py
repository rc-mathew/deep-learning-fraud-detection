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


def card_had_prior_fraud(is_fraud: np.ndarray, group_start: np.ndarray) -> np.ndarray:
    """True where an earlier transaction of the same card was fraud. Rows sorted by card, then time.

    IEEE-CIS labels an account's later transactions as fraud too, so these rows are easy
    for any model that sees card history. Evaluation only: never a model input.
    """
    cs = np.concatenate([[0], np.cumsum(np.asarray(is_fraud))])
    return (cs[np.arange(len(is_fraud))] - cs[np.asarray(group_start)]) > 0


def fraud_report(y_true: np.ndarray, scores: np.ndarray, prior_fraud: np.ndarray | None = None) -> dict[str, float]:
    """Metrics on all rows; with `prior_fraud`, also recall on cards with no earlier fraud."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    report = _report(y_true, scores)
    if prior_fraud is not None:
        keep = ~np.asarray(prior_fraud)
        report["first_fraud_recall_at_1pct_fpr"] = recall_at_fpr(y_true[keep], scores[keep], 0.01)
        report["first_fraud_pr_auc"] = float(average_precision_score(y_true[keep], scores[keep]))
    return report


def _report(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y_true)),
        "fraud_rate": float(y_true.mean()),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, scores, 0.01),
        "recall_at_5pct_fpr": recall_at_fpr(y_true, scores, 0.05),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "threshold_at_1pct_fpr": threshold_at_fpr(y_true, scores, 0.01),
    }
