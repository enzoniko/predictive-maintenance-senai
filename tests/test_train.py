from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pdm.models.train import (
    cross_validate_models,
    feature_columns,
    fit_final_model,
    get_candidate_models,
    label_mapping,
    split_features_labels,
)


def _make_separable_table(n_per_class: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    classes = ["Classe A", "Classe B", "Classe C"]
    rows = []
    for i, c in enumerate(classes):
        X = rng.normal(loc=i * 5.0, scale=0.5, size=(n_per_class, 4))
        for x in X:
            rows.append({"f0": x[0], "f1": x[1], "f2": x[2], "f3": x[3],
                         "is_silent": False, "label": c})
    return pd.DataFrame(rows)


def test_feature_columns_excludes_label_and_silent_flag() -> None:
    table = _make_separable_table()
    cols = feature_columns(table)
    assert "label" not in cols
    assert "is_silent" not in cols
    assert set(cols) == {"f0", "f1", "f2", "f3"}


def test_label_mapping_is_deterministic_and_alphabetical() -> None:
    y = np.array(["Classe B", "Classe A", "Classe C", "Classe A"])
    mapping = label_mapping(y)
    assert mapping == {"Classe A": 0, "Classe B": 1, "Classe C": 2}


def test_cross_validate_models_recovers_separable_classes() -> None:
    table = _make_separable_table()
    X, y = split_features_labels(table)
    models = get_candidate_models(seed=0)
    results = cross_validate_models(X, y, models, n_splits=3, seed=0)

    assert len(results) == len(models)
    for r in results:
        assert r.mean_f1_macro > 0.9, f"{r.model_name} underperformed on a trivially separable task"


def test_fit_final_model_predicts_training_labels_well() -> None:
    table = _make_separable_table()
    X, y = split_features_labels(table)
    model = get_candidate_models(seed=0)["random_forest"]
    fitted = fit_final_model(model, X, y, model_name="random_forest")
    preds = fitted.predict(X)
    accuracy = (preds == y).mean()
    assert accuracy > 0.95
