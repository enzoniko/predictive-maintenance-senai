from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from pdm.evaluation.imbalance import evaluate_under_prior, resample_to_prior

CLASSES = ["Classe A", "Classe B", "Classe C"]


def _balanced_dataset(n_per_class: int = 1000, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for i, c in enumerate(CLASSES):
        X = rng.normal(loc=i * 3.0, scale=1.0, size=(n_per_class, 2))
        rows.append(X)
        labels += [c] * n_per_class
    return pd.DataFrame(np.vstack(rows), columns=["f0", "f1"]), np.array(labels)


def test_resample_to_prior_matches_target_proportions() -> None:
    X, y = _balanced_dataset()
    prior = {"Classe A": 0.7, "Classe B": 0.2, "Classe C": 0.1}
    X_r, y_r = resample_to_prior(X, y, prior, seed=0)

    counts = pd.Series(y_r).value_counts(normalize=True)
    for c, p in prior.items():
        assert abs(counts[c] - p) < 0.02
    assert len(X_r) == len(y_r)


def test_resample_to_prior_rejects_prior_not_summing_to_one() -> None:
    X, y = _balanced_dataset()
    with pytest.raises(ValueError, match="sum to 1.0"):
        resample_to_prior(X, y, {"Classe A": 0.5, "Classe B": 0.2, "Classe C": 0.2}, seed=0)


def test_evaluate_under_prior_returns_sane_metrics() -> None:
    X, y = _balanced_dataset()
    model = LogisticRegression().fit(X, y)
    prior = {"Classe A": 0.8, "Classe B": 0.1, "Classe C": 0.1}
    result = evaluate_under_prior(model, X, y, prior, CLASSES, seed=0)

    assert result.n_samples > 0
    assert set(result.recall_per_class) == set(CLASSES)
    assert 0.0 <= result.f1_macro <= 1.0
    for v in result.pr_auc_per_class.values():
        assert np.isnan(v) or 0.0 <= v <= 1.0
