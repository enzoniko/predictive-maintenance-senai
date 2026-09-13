"""Standard classification metrics, plus probability calibration checks.

Calibration matters here specifically because conformal.py and the API's
abstention logic (serving/api.py) both lean on predicted probabilities
meaning what they claim to mean; a model that is "confidently wrong" is
worse for a maintenance team than one that says "I don't know".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def confusion_matrix_df(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(
        cm,
        index=[f"true_{lbl}" for lbl in labels],
        columns=[f"pred_{lbl}" for lbl in labels],
    )


def full_classification_report(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    return classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)


@dataclass
class CalibrationResult:
    bin_confidence: np.ndarray
    bin_accuracy: np.ndarray
    bin_count: np.ndarray
    expected_calibration_error: float


def evaluate_calibration(
    y_true: np.ndarray, proba: np.ndarray, classes: list[str], n_bins: int = 10
) -> CalibrationResult:
    """Expected Calibration Error for a multi-class model: bin predictions by
    the confidence of the *predicted* class, compare to empirical accuracy
    in each bin."""
    pred_idx = np.argmax(proba, axis=1)
    confidence = proba[np.arange(len(proba)), pred_idx]
    correct = np.array([classes[p] for p in pred_idx]) == y_true

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confidence = np.zeros(n_bins)
    bin_accuracy = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (confidence >= lo) & (confidence < hi if b < n_bins - 1 else confidence <= hi)
        bin_count[b] = mask.sum()
        if mask.sum() > 0:
            bin_confidence[b] = confidence[mask].mean()
            bin_accuracy[b] = correct[mask].mean()

    total = bin_count.sum()
    if total > 0:
        ece = float(np.sum(bin_count * np.abs(bin_accuracy - bin_confidence)) / total)
    else:
        ece = float("nan")
    return CalibrationResult(bin_confidence, bin_accuracy, bin_count, ece)
