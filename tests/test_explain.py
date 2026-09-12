from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pdm.evaluation.explain import (
    SHAP_AVAILABLE,
    lime_explain_instance,
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
