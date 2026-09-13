"""Candidate models, cross-validation and final-fit helpers.

Model choice is deliberately conservative: a decision tree (fully legible
rules, for stakeholders who need to see *why*), a random forest and a
histogram gradient boosting classifier (both scikit-learn) and XGBoost.
No deep-learning baseline is
included in the preview: at 50k windows x ~150 engineered features this is
not a data-scarce, representation-learning problem, and the interpretable
models below already reach very high macro-F1 (see
notebooks/03_feasibility_modeling.ipynb); a heavier model would trade
away exactly the auditability this project's research track (see
docs/06_track_pesquisa.md) argues for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

LABEL_COLUMN = "label"
NON_FEATURE_COLUMNS = {LABEL_COLUMN, "is_silent"}


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c not in NON_FEATURE_COLUMNS]


def split_features_labels(table: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    cols = feature_columns(table)
    return table[cols], table[LABEL_COLUMN].to_numpy()


def get_candidate_models(seed: int) -> dict[str, object]:
    models: dict[str, object] = {
        "decision_tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, random_state=seed
        ),
    }
    if XGBOOST_AVAILABLE:
        # `objective`/`eval_metric` are left unset so XGBoost auto-selects
        # between binary and multi-class variants from the encoded labels'
        # cardinality; hardcoding "multi:softprob" broke outright on a
        # 2-class label set during development (XGBoostError: num_class
        # must be >= 1), and a future project phase with a coarser
        # healthy/faulty label is exactly the kind of relabeling that would
        # hit this again.
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            n_jobs=-1,
            random_state=seed,
        )
    return models


@dataclass
class CVResult:
    model_name: str
    fold_f1_macro: list[float]
    fold_balanced_accuracy: list[float]
    fit_time_s: list[float]

    @property
    def mean_f1_macro(self) -> float:
        return float(np.mean(self.fold_f1_macro))

    @property
    def std_f1_macro(self) -> float:
        return float(np.std(self.fold_f1_macro))


def cross_validate_models(
    X: pd.DataFrame,
    y: np.ndarray,
    models: dict[str, object],
    n_splits: int = 5,
    seed: int = 42,
) -> list[CVResult]:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = []
    for name, model in models.items():
        # XGBoost's sklearn wrapper needs integer-encoded labels for
        # multi:softprob; every other model here is fine with raw strings.
        y_ = _encode_if_needed(name, y)
        scores = cross_validate(
            model, X, y_, cv=cv,
            scoring={"f1_macro": "f1_macro", "balanced_accuracy": "balanced_accuracy"},
            n_jobs=1,  # models already parallelize internally (n_jobs=-1)
        )
        results.append(
            CVResult(
                model_name=name,
                fold_f1_macro=scores["test_f1_macro"].tolist(),
                fold_balanced_accuracy=scores["test_balanced_accuracy"].tolist(),
                fit_time_s=scores["fit_time"].tolist(),
            )
        )
    return results


def label_mapping(y: np.ndarray) -> dict[str, int]:
    """Explicit, deterministic class -> integer mapping (alphabetical), used
    instead of re-fitting a fresh sklearn LabelEncoder on every call; that
    would only coincidentally stay consistent across calls (it does, here,
    because "Classe A".."Classe E" sort the same alphabetically as they are
    conceptually ordered, but relying on that coincidence silently would be
    a landmine for any future class relabeling)."""
    return {c: i for i, c in enumerate(sorted(set(y.tolist())))}


def _encode_if_needed(model_name: str, y: np.ndarray) -> np.ndarray:
    if model_name != "xgboost":
        return y
    mapping = label_mapping(y)
    return np.array([mapping[label] for label in y])


class EncodedLabelClassifier:
    """Adapts an estimator that needs integer-encoded labels (XGBoost) to
    the plain string-label interface (``predict``, ``predict_proba``,
    ``classes_``) every other model in this module already exposes.
    Without this, every downstream consumer (evaluation/imbalance.py,
    evaluation/robustness.py, evaluation/conformal.py, serving/api.py) would
    need its own special case for "this one model returns integers"; the
    wrapper contains that concern in one place instead.
    """

    def __init__(self, estimator: object, mapping: dict[str, int]) -> None:
        self.estimator = estimator
        self.mapping = mapping
        self.inverse_mapping = {v: k for k, v in mapping.items()}
        # predict_proba's column order follows the encoded integers 0..k-1,
        # which is exactly what this reconstructs.
        self.classes_ = np.array([self.inverse_mapping[i] for i in range(len(mapping))])

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> EncodedLabelClassifier:
        y_encoded = np.array([self.mapping[label] for label in y])
        self.estimator.fit(X, y_encoded)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        encoded = self.estimator.predict(X)
        return np.array([self.inverse_mapping[p] for p in encoded])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X)


def fit_final_model(model: object, X: pd.DataFrame, y: np.ndarray, model_name: str = "") -> object:
    """Fit ``model`` on string labels directly, wrapping it first if it is
    one of the models that needs integer-encoded labels internally. Returns
    the (possibly wrapped) fitted model; callers should keep using the
    returned object, not the original."""
    if model_name == "xgboost":
        model = EncodedLabelClassifier(model, label_mapping(y))
    model.fit(X, y)
    return model
