"""Bayesian hyperparameter search (Optuna, TPE sampler) for the two
strongest candidates from train.py's cross-validation (HistGradientBoosting
always; XGBoost when available). Runs on a stratified subsample by default
-- full 5-fold CV per trial at 50k rows x ~150 features is not worth the
wall-clock time this search needs to be useful within the case's timeline;
the winning configuration is re-validated on the full data in
evaluation/metrics.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split

from pdm.models.train import XGBOOST_AVAILABLE, label_mapping

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class TuningResult:
    model_name: str
    best_params: dict
    best_value: float
    n_trials: int


def _subsample(X: pd.DataFrame, y: np.ndarray, frac: float, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    if frac >= 1.0:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=frac, stratify=y, random_state=seed
    )
    return X_sub, y_sub


def _hgb_objective(trial: optuna.Trial, X: pd.DataFrame, y: np.ndarray, cv: StratifiedKFold) -> float:
    params = {
        "max_iter": trial.suggest_int("max_iter", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 1.0, log=True),
    }
    model = HistGradientBoostingClassifier(random_state=0, **params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=1)
    return float(scores.mean())


def _xgb_objective(trial: optuna.Trial, X: pd.DataFrame, y_encoded: np.ndarray, cv: StratifiedKFold) -> float:
    from xgboost import XGBClassifier

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    # See train.get_candidate_models for why objective/eval_metric are left
    # to XGBoost's auto-detection rather than hardcoded.
    model = XGBClassifier(n_jobs=-1, random_state=0, **params)
    scores = cross_val_score(model, X, y_encoded, cv=cv, scoring="f1_macro", n_jobs=1)
    return float(scores.mean())


def tune_hist_gradient_boosting(
    X: pd.DataFrame, y: np.ndarray, n_trials: int = 30, subsample_frac: float = 0.3,
    n_splits: int = 3, seed: int = 42,
) -> TuningResult:
    X_sub, y_sub = _subsample(X, y, subsample_frac, seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda t: _hgb_objective(t, X_sub, y_sub, cv), n_trials=n_trials, show_progress_bar=False)
    return TuningResult("hist_gradient_boosting", study.best_params, study.best_value, n_trials)


def tune_xgboost(
    X: pd.DataFrame, y: np.ndarray, n_trials: int = 30, subsample_frac: float = 0.3,
    n_splits: int = 3, seed: int = 42,
) -> TuningResult | None:
    if not XGBOOST_AVAILABLE:
        return None
    X_sub, y_sub = _subsample(X, y, subsample_frac, seed)
    mapping = label_mapping(y_sub)
    y_encoded = np.array([mapping[label] for label in y_sub])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda t: _xgb_objective(t, X_sub, y_encoded, cv), n_trials=n_trials, show_progress_bar=False)
    return TuningResult("xgboost", study.best_params, study.best_value, n_trials)
