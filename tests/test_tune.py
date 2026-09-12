from __future__ import annotations

import numpy as np
import pandas as pd

from pdm.models.tune import tune_hist_gradient_boosting, tune_xgboost


def _make_table(n_per_class: int = 60, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = ["Classe A", "Classe B"]
    rows, labels = [], []
    for i, c in enumerate(classes):
        X = rng.normal(loc=i * 4.0, scale=0.7, size=(n_per_class, 3))
        rows.append(X)
        labels += [c] * n_per_class
    X = pd.DataFrame(np.vstack(rows), columns=["f0", "f1", "f2"])
    return X, np.array(labels)


def test_tune_hist_gradient_boosting_returns_valid_result() -> None:
    X, y = _make_table()
    result = tune_hist_gradient_boosting(X, y, n_trials=3, subsample_frac=1.0, n_splits=2)
    assert result.model_name == "hist_gradient_boosting"
    assert 0.0 <= result.best_value <= 1.0
    assert result.n_trials == 3
    assert "max_iter" in result.best_params


def test_tune_xgboost_returns_valid_result_or_none_if_unavailable() -> None:
    X, y = _make_table()
    result = tune_xgboost(X, y, n_trials=3, subsample_frac=1.0, n_splits=2)
    if result is not None:
        assert result.model_name == "xgboost"
        assert 0.0 <= result.best_value <= 1.0
