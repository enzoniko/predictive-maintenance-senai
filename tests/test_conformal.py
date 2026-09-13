from __future__ import annotations

import numpy as np
import pytest

from pdm.evaluation.conformal import (
    SplitConformalClassifier,
    abstention_rate,
    average_set_size,
    coverage_by_class,
    empirical_coverage,
)

CLASSES = ["Classe A", "Classe B", "Classe C"]


def _synthetic_calibrated_probs(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities that are, by construction, well-calibrated: draw a true
    class uniformly, then generate a probability vector peaked on it."""
    rng = np.random.default_rng(seed)
    y_idx = rng.integers(0, 3, size=n)
    proba = rng.dirichlet(alpha=[0.5, 0.5, 0.5], size=n)
    # Bias each row so its argmax matches the drawn true class most of the time.
    for i in range(n):
        proba[i, y_idx[i]] += 1.5
    proba = proba / proba.sum(axis=1, keepdims=True)
    y = np.array([CLASSES[i] for i in y_idx])
    return proba, y


@pytest.mark.parametrize("method", ["aps", "lac"])
def test_conformal_achieves_approximately_target_coverage(method: str) -> None:
    proba_calib, y_calib = _synthetic_calibrated_probs(3000, seed=0)
    proba_test, y_test = _synthetic_calibrated_probs(3000, seed=1)

    clf = SplitConformalClassifier(CLASSES, method=method)
    clf.calibrate(proba_calib, y_calib, target_coverage=0.9)
    sets = clf.predict_sets(proba_test)

    coverage = empirical_coverage(sets, y_test, CLASSES)
    assert 0.85 <= coverage <= 0.97, f"{method} coverage {coverage} far from target 0.90"


def test_aps_abstention_rate_is_in_line_with_alpha() -> None:
    # Randomized APS *can* produce an empty set (see conformal.py's module
    # docstring): an empty set is always a non-coverage event, so its rate
    # should stay roughly within the alpha = 1; target_coverage budget,
    # not dominate it.
    proba_calib, y_calib = _synthetic_calibrated_probs(1000, seed=0)
    proba_test, _ = _synthetic_calibrated_probs(500, seed=1)
    clf = SplitConformalClassifier(CLASSES, method="aps")
    clf.calibrate(proba_calib, y_calib, target_coverage=0.9)
    sets = clf.predict_sets(proba_test)
    assert abstention_rate(sets) < 0.15


def test_higher_confidence_predictions_get_smaller_sets() -> None:
    proba_calib, y_calib = _synthetic_calibrated_probs(2000, seed=0)
    clf = SplitConformalClassifier(CLASSES, method="aps")
    clf.calibrate(proba_calib, y_calib, target_coverage=0.9)

    confident = np.array([[0.97, 0.02, 0.01]])
    ambiguous = np.array([[0.4, 0.35, 0.25]])
    set_confident = clf.predict_sets(confident)
    set_ambiguous = clf.predict_sets(ambiguous)
    assert set_confident.sum() <= set_ambiguous.sum()


def test_coverage_by_class_and_average_set_size_are_well_formed() -> None:
    proba_calib, y_calib = _synthetic_calibrated_probs(2000, seed=0)
    proba_test, y_test = _synthetic_calibrated_probs(1000, seed=1)
    clf = SplitConformalClassifier(CLASSES, method="aps")
    clf.calibrate(proba_calib, y_calib, target_coverage=0.9)
    sets = clf.predict_sets(proba_test)

    per_class = coverage_by_class(sets, y_test, CLASSES)
    assert set(per_class) == set(CLASSES)
    # Randomized APS can occasionally abstain (empty set), so the average
    # set size is not bounded below by 1 the way the deterministic variant
    # would be; only that it stays within a sane range overall.
    assert 0.5 <= average_set_size(sets) <= 3.0


def test_invalid_method_raises() -> None:
    with pytest.raises(ValueError):
        SplitConformalClassifier(CLASSES, method="bogus")
