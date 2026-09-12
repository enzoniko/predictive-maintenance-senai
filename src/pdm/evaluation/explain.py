"""Model explainability: permutation importance (always available), SHAP
and LIME (optional).

``shap`` needs a C/C++ toolchain to build from source on Windows ARM64
(same root cause as mlflow/streamlit/ssqueezepy -- no prebuilt wheel
exists there; confirmed importable on the Linux x86-64 test server, see
docs/03_arquitetura.md). ``lime`` is pure Python and installs everywhere.
Every function here degrades gracefully rather than raising when SHAP is
unavailable, so the rest of the evaluation suite is never blocked by it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def permutation_importance_report(
    model, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 10, seed: int = 42
) -> pd.DataFrame:
    """Global feature importance that works for any fitted estimator,
    used as the baseline explanation and as a cross-check for SHAP's
    global ranking when SHAP is available."""
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=seed, scoring="f1_macro", n_jobs=1
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def _aggregate_shap_values(shap_values) -> np.ndarray:
    """Reduce a TreeExplainer's ``shap_values`` output to one mean-|value|
    per feature, regardless of which of the three shapes SHAP hands back --
    kept as its own pure function (no shap import needed to exercise it) so
    it can be unit-tested without the library installed, which matters
    because it was never exercised at all on the one platform where SHAP is
    actually importable (see docs/03_arquitetura.md, section 3.6: shap
    fails to build on this project's Windows ARM64 dev machine). Running
    for real on the Linux x86-64 test server surfaced the bug this function
    fixes: multi-class TreeExplainer output changed, across SHAP versions,
    from a list of one (n_samples, n_features) array per class to a single
    (n_samples, n_features, n_classes) array -- code written only against
    the list form silently produced a 2-D "mean" (features x classes)
    instead of a 1-D per-feature vector, which crashed one level up when
    that got zipped into a DataFrame column.

    - list of 2-D arrays (older multi-class API): one array per class.
    - a single 3-D array, shape (n_samples, n_features, n_classes): newer
      multi-class API.
    - a single 2-D array, shape (n_samples, n_features): binary/regression.
    """
    if isinstance(shap_values, list):
        stacked = np.stack([np.abs(v) for v in shap_values], axis=0)
        return stacked.mean(axis=(0, 1))
    values = np.asarray(shap_values)
    if values.ndim == 3:
        return np.abs(values).mean(axis=(0, 2))
    return np.abs(values).mean(axis=0)


def shap_global_importance(
    model, X: pd.DataFrame, sample_size: int = 2000, seed: int = 42
) -> pd.DataFrame | None:
    """Mean |SHAP value| per feature, using TreeExplainer (fast, exact for
    tree ensembles). Returns None if shap is not importable."""
    if not SHAP_AVAILABLE:
        return None
    X_sample = X.sample(n=min(sample_size, len(X)), random_state=seed)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    mean_abs = _aggregate_shap_values(shap_values)
    return (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def lime_explain_instance(
    model, X_train: pd.DataFrame, instance: pd.Series, class_names: list[str], num_features: int = 10
):
    """Local explanation for a single prediction via LIME's tabular
    explainer -- always available (pure Python), used both for the
    technical presentation and as a sanity cross-check against SHAP's local
    attributions when both are present."""
    from lime.lime_tabular import LimeTabularExplainer

    explainer = LimeTabularExplainer(
        X_train.to_numpy(),
        feature_names=list(X_train.columns),
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
    )
    predict_fn = model.predict_proba
    return explainer.explain_instance(
        instance.to_numpy(), predict_fn, num_features=num_features, top_labels=1
    )
