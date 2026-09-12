from __future__ import annotations

import numpy as np

from pdm.evaluation.metrics import confusion_matrix_df, evaluate_calibration, full_classification_report

CLASSES = ["Classe A", "Classe B"]


def test_confusion_matrix_df_shape_and_labels() -> None:
    y_true = np.array(["Classe A", "Classe A", "Classe B", "Classe B"])
    y_pred = np.array(["Classe A", "Classe B", "Classe B", "Classe B"])
    df = confusion_matrix_df(y_true, y_pred, CLASSES)
    assert df.shape == (2, 2)
    assert df.loc["true_Classe A", "pred_Classe A"] == 1
    assert df.loc["true_Classe B", "pred_Classe B"] == 2


def test_classification_report_contains_expected_keys() -> None:
    y_true = np.array(["Classe A", "Classe B"])
    y_pred = np.array(["Classe A", "Classe A"])
    report = full_classification_report(y_true, y_pred, CLASSES)
    assert "Classe A" in report
    assert "macro avg" in report


def test_perfectly_calibrated_model_has_near_zero_ece() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    # Predicted confidence equals true accuracy at every confidence level by
    # construction: draw confidence, then make the prediction correct with
    # exactly that probability.
    confidences = rng.uniform(0.5, 1.0, size=n)
    is_correct = rng.uniform(size=n) < confidences
    y_true = np.array(["Classe A"] * n)
    proba = np.zeros((n, 2))
    proba[:, 0] = np.where(is_correct, confidences, 1 - confidences)
    proba[:, 1] = 1 - proba[:, 0]

    result = evaluate_calibration(y_true, proba, CLASSES, n_bins=10)
    assert result.expected_calibration_error < 0.05


def test_overconfident_model_has_high_ece() -> None:
    n = 500
    y_true = np.array(["Classe A"] * (n // 2) + ["Classe B"] * (n // 2))
    # Always 99% confident in "Classe A" regardless of truth -> ~50% accuracy
    # at ~99% confidence is badly miscalibrated.
    proba = np.tile([0.99, 0.01], (n, 1))
    result = evaluate_calibration(y_true, proba, CLASSES, n_bins=10)
    assert result.expected_calibration_error > 0.3
