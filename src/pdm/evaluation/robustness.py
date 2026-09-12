"""Robustness and sanity-control tests for a trained model.

Two kinds of check live here:

* **Degradation tests** answer "how much worse does the model get if a
  sensor drops out, a feature group is unavailable, or the signal is
  noisier than training?" -- the honest answer a director needs before
  betting a maintenance decision on this system, not just the best-case
  cross-validated score.
* **Controls** answer "does the evaluation pipeline itself work?" -- a
  model that still scores well after its labels are shuffled would mean
  the whole measurement setup is leaking information, not that the model
  is unreasonably good.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


@dataclass
class DegradationResult:
    scenario: str
    f1_macro: float
    f1_macro_baseline: float

    @property
    def relative_drop(self) -> float:
        if self.f1_macro_baseline == 0:
            return float("nan")
        return (self.f1_macro_baseline - self.f1_macro) / self.f1_macro_baseline


def _score(model, X: pd.DataFrame, y: np.ndarray, classes: list[str]) -> float:
    y_pred = model.predict(X)
    return float(f1_score(y, y_pred, labels=classes, average="macro", zero_division=0))


def sensor_dropout_test(
    model, X_test: pd.DataFrame, y_test: np.ndarray, classes: list[str], sensor_prefixes: list[str]
) -> list[DegradationResult]:
    """Zero out every feature belonging to one sensor at a time (simulating
    that sensor going offline) and measure the resulting drop. The model is
    *not* retrained -- this measures how much the already-deployed model
    leans on each channel, which is what matters operationally."""
    baseline = _score(model, X_test, y_test, classes)
    results = [DegradationResult("baseline (all sensors)", baseline, baseline)]
    for prefix in sensor_prefixes:
        cols = [c for c in X_test.columns if c.startswith(f"{prefix}_")]
        X_dropped = X_test.copy()
        X_dropped[cols] = 0.0
        score = _score(model, X_dropped, y_test, classes)
        results.append(DegradationResult(f"drop_sensor:{prefix}", score, baseline))
    return results


def feature_group_ablation_test(
    model, X_test: pd.DataFrame, y_test: np.ndarray, classes: list[str]
) -> list[DegradationResult]:
    """Zero out an entire feature *family* (time/frequency/wavelet) across
    all sensors, to see which domain the model actually relies on."""
    baseline = _score(model, X_test, y_test, classes)
    group_markers = {
        "time_domain": ["_rms", "_crest_factor", "_kurtosis", "_skewness", "_shape_factor",
                        "_impulse_factor", "_clearance_factor", "_zero_crossing_rate",
                        "_peak", "_std", "_mean"],
        "frequency_domain": ["_band_", "_spectral_", "_peak1_", "_peak2_"],
        "wavelet_domain": ["_cwt_", "_ssq_"],
    }
    results = [DegradationResult("baseline (all groups)", baseline, baseline)]
    for group, markers in group_markers.items():
        cols = [c for c in X_test.columns if any(m in c for m in markers)]
        X_dropped = X_test.copy()
        X_dropped[cols] = 0.0
        score = _score(model, X_dropped, y_test, classes)
        results.append(DegradationResult(f"drop_group:{group}", score, baseline))
    return results


def gaussian_noise_injection_test(
    model, X_test: pd.DataFrame, y_test: np.ndarray, classes: list[str],
    noise_levels: list[float] = (0.1, 0.25, 0.5, 1.0), seed: int = 42,
) -> list[DegradationResult]:
    """Perturb every feature by Gaussian noise proportional to its own
    training-set std (``noise_level`` is that proportion). This is a proxy
    for "the sensor got noisier" applied at the feature level -- a stricter
    test would inject noise into the raw waveform and re-run feature
    extraction, which notebooks/04 does for the retained scenario used in
    the final report; this fast version is what runs in CI/CD-style checks."""
    rng = np.random.default_rng(seed)
    baseline = _score(model, X_test, y_test, classes)
    feature_std = X_test.std(axis=0)
    results = [DegradationResult("baseline (no noise)", baseline, baseline)]
    for level in noise_levels:
        noise = rng.normal(0, feature_std.values * level, size=X_test.shape)
        X_noisy = X_test + noise
        score = _score(model, X_noisy, y_test, classes)
        results.append(DegradationResult(f"gaussian_noise:{level}", score, baseline))
    return results


def label_shuffle_control(
    model_factory, X: pd.DataFrame, y: np.ndarray, n_splits: int = 5, seed: int = 42
) -> float:
    """A model trained on shuffled labels should score at chance level. If it
    doesn't, something in the pipeline (feature leakage, a column derived
    from the label, a data-splitting bug) is inflating every other result in
    this report and needs to be found before anything else is trusted."""
    rng = np.random.default_rng(seed)
    y_shuffled = rng.permutation(y)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(model_factory(), X, y_shuffled, cv=cv, scoring="f1_macro", n_jobs=1)
    return float(scores.mean())


def noise_sensor_control(
    model_factory, X_noise_sensor_features: pd.DataFrame, y: np.ndarray,
    n_splits: int = 5, seed: int = 42,
) -> float:
    """A model trained on nothing but the audit-excluded noise sensor's
    features should also score at chance -- if it doesn't, the audit's
    exclusion decision (data/audit.py) needs revisiting, not this test."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(
        model_factory(), X_noise_sensor_features, y, cv=cv, scoring="f1_macro", n_jobs=1
    )
    return float(scores.mean())
