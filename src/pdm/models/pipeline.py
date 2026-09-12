"""End-to-end training pipeline: audit -> features -> split -> model
selection -> conformal calibration -> evaluation -> a saved ModelBundle.

This is what ``python -m pdm.cli train`` and ``evaluate`` run, and what
notebooks/03 and notebooks/04 call into directly so the notebooks and the
CLI can never drift apart. See docs/03_arquitetura.md for how this fits
into the rest of the system, and docs/04_cronograma_9_meses.md for how the
one-shot audit -> features -> train sequence here maps onto that
schedule's phases 1-3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pdm.config import Config
from pdm.data.audit import AuditReport, run_audit
from pdm.data.loader import SensorDataset, load_sensor_dataset
from pdm.evaluation.conformal import (
    ConformalCalibrationSummary,
    SplitConformalClassifier,
    average_set_size,
    empirical_coverage,
)
from pdm.evaluation.imbalance import ImbalanceEvaluation, evaluate_under_prior
from pdm.evaluation.metrics import CalibrationResult, evaluate_calibration, full_classification_report
from pdm.evaluation.robustness import (
    DegradationResult,
    feature_group_ablation_test,
    label_shuffle_control,
    noise_sensor_control,
    sensor_dropout_test,
)
from pdm.features.builder import build_features_from_raw, fit_cleaners
from pdm.models.bundle import ModelBundle
from pdm.models.train import CVResult, cross_validate_models, feature_columns, fit_final_model, get_candidate_models


@dataclass
class TrainingArtifacts:
    bundle: ModelBundle
    audit_report: AuditReport
    cv_results: list[CVResult]
    best_model_name: str
    test_classification_report: dict
    test_calibration: CalibrationResult
    conformal_summary: ConformalCalibrationSummary
    conformal_coverage: float
    conformal_avg_set_size: float
    imbalance_evaluation: ImbalanceEvaluation
    sensor_dropout: list[DegradationResult]
    feature_group_ablation: list[DegradationResult]
    label_shuffle_control_f1: float
    noise_sensor_control_f1: float
    extra: dict[str, Any] = field(default_factory=dict)


def _stratified_three_way_split_indices(
    y: np.ndarray, split_cfg: dict, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split at the level of raw-window indices, *before* any feature is
    computed -- see features/builder.py's module docstring for why cleaning
    thresholds must be fit after this split, not before it."""
    train_frac, calib_frac, test_frac = split_cfg["train"], split_cfg["calibration"], split_cfg["test"]
    assert abs(train_frac + calib_frac + test_frac - 1.0) < 1e-6

    all_idx = np.arange(len(y))
    idx_train, idx_rest = train_test_split(
        all_idx, train_size=train_frac, stratify=y, random_state=seed
    )
    relative_calib = calib_frac / (calib_frac + test_frac)
    idx_calib, idx_test = train_test_split(
        idx_rest, train_size=relative_calib, stratify=y[idx_rest], random_state=seed
    )
    return idx_train, idx_calib, idx_test


def _build_split_features(
    dataset: SensorDataset,
    cleaners: dict,
    idx: np.ndarray,
    include_wavelet: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Feature table (pure feature columns only, is_silent already merged
    in) plus the corresponding label array, for one index subset."""
    raw = {name: dataset.sensors[name][idx] for name in cleaners}
    table, is_silent = build_features_from_raw(raw, cleaners, dataset.sample_rate_hz, include_wavelet)
    table["is_silent"] = is_silent
    return table, dataset.labels[idx]


def run_training_pipeline(config: Config, dataset: SensorDataset | None = None) -> TrainingArtifacts:
    dataset = dataset or load_sensor_dataset(config)
    audit_report = run_audit(dataset, config)
    seed = config.random_seed
    classes = sorted(set(dataset.labels.tolist()))

    idx_train, idx_calib, idx_test = _stratified_three_way_split_indices(dataset.labels, config.split, seed)

    # Cleaning thresholds (saturation, silence) are fit on the training
    # indices only, then reused unchanged for calibration and test -- see
    # features/builder.py's module docstring.
    cleaners = fit_cleaners(dataset, config, audit_report.recommended_sensors, row_indices=idx_train)

    train_table, y_train = _build_split_features(dataset, cleaners, idx_train)
    calib_table, y_calib = _build_split_features(dataset, cleaners, idx_calib)
    test_table, y_test = _build_split_features(dataset, cleaners, idx_test)

    cols = feature_columns(train_table.assign(label="_"))
    X_train, X_calib, X_test = train_table[cols], calib_table[cols], test_table[cols]

    models = get_candidate_models(seed)
    cv_results = cross_validate_models(X_train, y_train, models, n_splits=config.cross_validation["n_splits"], seed=seed)
    best = max(cv_results, key=lambda r: r.mean_f1_macro)
    best_model = fit_final_model(models[best.model_name], X_train, y_train, model_name=best.model_name)

    y_pred_test = best_model.predict(X_test)
    test_report = full_classification_report(y_test, y_pred_test, classes)
    proba_test = best_model.predict_proba(X_test)
    calibration = evaluate_calibration(y_test, proba_test, classes)

    conformal = SplitConformalClassifier(classes, method=config.conformal["method"])
    proba_calib = best_model.predict_proba(X_calib)
    conformal_summary = conformal.calibrate(proba_calib, y_calib, config.conformal["target_coverage"])
    conformal_sets = conformal.predict_sets(proba_test)
    coverage = empirical_coverage(conformal_sets, y_test, classes)
    avg_set_size = average_set_size(conformal_sets)

    imbalance_eval = evaluate_under_prior(
        best_model, X_test, y_test, config.imbalance["simulated_prior"], classes, seed=seed
    )

    dropout_results = sensor_dropout_test(best_model, X_test, y_test, classes, audit_report.recommended_sensors)
    ablation_results = feature_group_ablation_test(best_model, X_test, y_test, classes)

    shuffle_score = label_shuffle_control(
        lambda: get_candidate_models(seed)[best.model_name], X_train, y_train, n_splits=3, seed=seed
    )

    excluded_sensors = [s for s in dataset.sensor_names if s not in audit_report.recommended_sensors]
    noise_score = float("nan")
    if excluded_sensors:
        # Same train-only-fitting discipline as the retained sensors, even
        # though this control never touches the held-out test set itself.
        noise_cleaners = fit_cleaners(dataset, config, excluded_sensors[:1], row_indices=idx_train)
        noise_table, y_noise = _build_split_features(dataset, noise_cleaners, idx_train, include_wavelet=False)
        noise_cols = feature_columns(noise_table.assign(label="_"))
        noise_score = noise_sensor_control(
            lambda: get_candidate_models(seed)[best.model_name],
            noise_table[noise_cols], y_noise, n_splits=3, seed=seed,
        )

    bundle = ModelBundle(
        model=best_model,
        model_name=best.model_name,
        feature_columns=cols,
        classes=classes,
        sensor_names=audit_report.recommended_sensors,
        sample_rate_hz=dataset.sample_rate_hz,
        window_len=dataset.window_len,
        conformal=conformal,
        cleaners=cleaners,
        metrics={
            "cv_f1_macro_mean": best.mean_f1_macro,
            "cv_f1_macro_std": best.std_f1_macro,
            "test_f1_macro": test_report["macro avg"]["f1-score"],
            "test_ece": calibration.expected_calibration_error,
            "conformal_coverage": coverage,
            "conformal_avg_set_size": avg_set_size,
        },
    )

    return TrainingArtifacts(
        bundle=bundle,
        audit_report=audit_report,
        cv_results=cv_results,
        best_model_name=best.model_name,
        test_classification_report=test_report,
        test_calibration=calibration,
        conformal_summary=conformal_summary,
        conformal_coverage=coverage,
        conformal_avg_set_size=avg_set_size,
        imbalance_evaluation=imbalance_eval,
        sensor_dropout=dropout_results,
        feature_group_ablation=ablation_results,
        label_shuffle_control_f1=shuffle_score,
        noise_sensor_control_f1=noise_score,
    )
