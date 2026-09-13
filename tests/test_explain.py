from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pdm.evaluation.explain import (
    SHAP_AVAILABLE,
    _aggregate_shap_values,
    lime_explain_instance,
    local_shap_values_for_class,
    permutation_importance_report,
    shap_global_importance,
)

CLASSES = ["Classe A", "Classe B"]


def _make_dataset(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 300
    informative = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, n)
    y = np.where(informative > 0, "Classe A", "Classe B")
    X = pd.DataFrame({"informative": informative, "noise": noise})
    return X, y


def test_permutation_importance_ranks_informative_feature_first() -> None:
    X, y = _make_dataset()
    model = RandomForestClassifier(n_estimators=100, random_state=0).fit(X, y)
    report = permutation_importance_report(model, X, y, n_repeats=10, seed=0)
    assert report.iloc[0]["feature"] == "informative"


def test_shap_global_importance_matches_permutation_ranking_when_available() -> None:
    X, y = _make_dataset()
    model = RandomForestClassifier(n_estimators=100, random_state=0).fit(X, y)
    report = shap_global_importance(model, X, sample_size=200, seed=0)
    if not SHAP_AVAILABLE:
        assert report is None
        return
    assert report.iloc[0]["feature"] == "informative"


def test_lime_explains_a_single_instance() -> None:
    X, y = _make_dataset()
    model = RandomForestClassifier(n_estimators=100, random_state=0).fit(X, y)
    explanation = lime_explain_instance(model, X, X.iloc[0], class_names=CLASSES, num_features=2)
    as_list = explanation.as_list(label=explanation.available_labels()[0])
    assert len(as_list) == 2


def test_aggregate_shap_values_handles_list_of_per_class_arrays() -> None:
    # Older SHAP API for multi-class TreeExplainer: one (n_samples,
    # n_features) array per class.
    rng = np.random.default_rng(0)
    per_class = [rng.normal(size=(50, 4)) for _ in range(3)]
    result = _aggregate_shap_values(per_class)
    assert result.shape == (4,)
    expected = np.mean([np.abs(a) for a in per_class], axis=(0, 1))
    assert np.allclose(result, expected)


def test_aggregate_shap_values_handles_3d_array() -> None:
    # Newer SHAP API for multi-class TreeExplainer: a single
    # (n_samples, n_features, n_classes) array; this shape is exactly
    # what broke shap_global_importance during the real explainability run
    # (see docs/03_arquitetura.md, section 3.6).
    rng = np.random.default_rng(0)
    values = rng.normal(size=(50, 4, 3))
    result = _aggregate_shap_values(values)
    assert result.shape == (4,)
    assert np.allclose(result, np.abs(values).mean(axis=(0, 2)))


def test_aggregate_shap_values_handles_2d_array() -> None:
    # Binary classification / regression: a single (n_samples, n_features) array.
    rng = np.random.default_rng(0)
    values = rng.normal(size=(50, 4))
    result = _aggregate_shap_values(values)
    assert result.shape == (4,)
    assert np.allclose(result, np.abs(values).mean(axis=0))


def test_local_shap_values_for_class_handles_list_of_per_class_arrays() -> None:
    rng = np.random.default_rng(0)
    per_class = [rng.normal(size=(1, 4)) for _ in range(3)]
    result = local_shap_values_for_class(per_class, class_idx=1)
    assert result.shape == (4,)
    assert np.allclose(result, per_class[1][0])


def test_local_shap_values_for_class_handles_3d_array() -> None:
    # This is exactly the shape a single-row TreeExplainer call returns on
    # current SHAP: (n_samples=1, n_features, n_classes). Code that only
    # handled the list-of-2D-arrays shape (e.g. `shap_values[0]`) silently
    # sliced off the first feature instead of indexing the one instance,
    # returning a per-class vector instead of a per-feature one -- this is
    # the bug that broke POST /explain's SHAP branch in production, caught
    # only once SHAP was genuinely installed and exercised end to end.
    rng = np.random.default_rng(0)
    values = rng.normal(size=(1, 4, 3))
    result = local_shap_values_for_class(values, class_idx=2)
    assert result.shape == (4,)
    assert np.allclose(result, values[0, :, 2])


def test_local_shap_values_for_class_handles_2d_array() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(1, 4))
    result = local_shap_values_for_class(values, class_idx=0)
    assert result.shape == (4,)
    assert np.allclose(result, values[0])
