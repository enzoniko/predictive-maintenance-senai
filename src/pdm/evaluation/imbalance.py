"""Re-evaluate a model under a realistic class prior.

The case dataset is perfectly balanced (10,000 windows per class -- see
docs/data_audit findings), which the audit flags as almost certainly
curated rather than representative of a real factory floor, where healthy
operation dominates and specific faults are comparatively rare. Reporting
metrics only on the balanced test set would overstate how the model will
behave in production; this module resamples (without replacement) a held-out
balanced test set into one matching a configurable target prior
(``configs/default.yaml``'s ``imbalance.simulated_prior``) and re-computes
metrics there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, recall_score


def resample_to_prior(
    X: pd.DataFrame, y: np.ndarray, prior: dict[str, float], seed: int = 42
) -> tuple[pd.DataFrame, np.ndarray]:
    """Subsample (X, y) without replacement so class proportions match
    ``prior``. The total output size is capped by whichever class has the
    least headroom, so no class is ever oversampled with repeats -- this
    changes *proportions*, not the reliability of any individual class's
    represented sample."""
    if not np.isclose(sum(prior.values()), 1.0, atol=1e-6):
        raise ValueError(f"prior must sum to 1.0, got {sum(prior.values())}")

    rng = np.random.default_rng(seed)
    class_indices = {c: np.where(y == c)[0] for c in prior}
    available = {c: len(idx) for c, idx in class_indices.items()}

    # Largest total N such that every class's required count <= what's available.
    max_total = min(available[c] / p for c, p in prior.items() if p > 0)
    total = int(np.floor(max_total))

    selected = []
    for c, p in prior.items():
        n = int(round(total * p))
        n = min(n, available[c])
        chosen = rng.choice(class_indices[c], size=n, replace=False)
        selected.append(chosen)
    idx = np.concatenate(selected)
    rng.shuffle(idx)
    return X.iloc[idx].reset_index(drop=True), y[idx]


@dataclass
class ImbalanceEvaluation:
    prior: dict[str, float]
    n_samples: int
    recall_per_class: dict[str, float]
    f1_macro: float
    pr_auc_per_class: dict[str, float]


def evaluate_under_prior(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    prior: dict[str, float],
    classes: list[str],
    seed: int = 42,
) -> ImbalanceEvaluation:
    X_resampled, y_resampled = resample_to_prior(X_test, y_test, prior, seed)
    y_pred = model.predict(X_resampled)
    proba = model.predict_proba(X_resampled) if hasattr(model, "predict_proba") else None

    recall = recall_score(y_resampled, y_pred, labels=classes, average=None, zero_division=0)
    recall_per_class = dict(zip(classes, recall.tolist()))
    f1_macro = float(f1_score(y_resampled, y_pred, labels=classes, average="macro", zero_division=0))

    pr_auc_per_class = {}
    if proba is not None:
        # Do not assume `classes` and `model.classes_` share column order --
        # look each one up explicitly (bitten once already by an analogous
        # assumption with XGBoost's integer label encoding; see
        # models/train.py's label_mapping docstring).
        model_classes = list(model.classes_)
        for c in classes:
            y_binary = (y_resampled == c).astype(int)
            if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
                pr_auc_per_class[c] = float("nan")
            else:
                col = model_classes.index(c)
                pr_auc_per_class[c] = float(average_precision_score(y_binary, proba[:, col]))

    return ImbalanceEvaluation(
        prior=prior,
        n_samples=len(y_resampled),
        recall_per_class=recall_per_class,
        f1_macro=f1_macro,
        pr_auc_per_class=pr_auc_per_class,
    )
