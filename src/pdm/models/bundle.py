"""A self-contained, serializable bundle of everything inference needs.

Bundling the fitted model together with its feature-column order, the class
list, the sensors it expects, and (optionally) a calibrated conformal
predictor means ``serving/api.py`` never has to guess these things or risk
drifting out of sync with how the model was trained; load one file, get a
consistent, versioned unit.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import sklearn

from pdm.evaluation.conformal import SplitConformalClassifier
from pdm.preprocessing.cleaning import SensorCleaner

BUNDLE_FILENAME = "model_bundle.joblib"


def _current_environment() -> dict[str, str]:
    try:
        import xgboost

        xgboost_version = xgboost.__version__
    except ImportError:
        xgboost_version = "unavailable"
    return {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost_version,
        "platform": platform.platform(),
    }


@dataclass
class ModelBundle:
    model: Any
    model_name: str
    feature_columns: list[str]
    classes: list[str]
    sensor_names: list[str]
    sample_rate_hz: int
    window_len: int
    conformal: SplitConformalClassifier | None
    # Fitted on training-split rows only (see features/builder.py) and
    # reused verbatim at inference time; serving/api.py must not re-fit
    # these against production traffic.
    cleaners: dict[str, SensorCleaner] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    # Recorded at save() time so a failed load(); see the note there about
    # cross-version pickle incompatibility; can at least be diagnosed
    # after the fact by comparing against a *successfully* loaded sibling
    # bundle, even though the failing load itself can't read this field
    # (the unpickling dies on the model object before reaching it).
    environment: dict[str, str] = field(default_factory=_current_environment)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Uncompressed, this HistGradientBoostingClassifier-based bundle
        # weighed in at 5.8 MB; on its own already over the case
        # submission's 5 MB-per-attachment limit once the repo is zipped.
        # joblib's zlib compression (level 9, the max) brings a real bundle
        # from this project down to ~2.9 MB with no change to what gets
        # loaded back.
        joblib.dump(self, path, compress=9)
        return path

    @staticmethod
    def load(path: Path) -> ModelBundle:
        try:
            bundle = joblib.load(path)
        except (ModuleNotFoundError, AttributeError, ImportError) as exc:
            # Concretely reproduced against a bundle trained under a
            # different scikit-learn release: it fails with "No module
            # named '_loss'" when loaded under a version where
            # HistGradientBoosting's internal loss module has moved;
            # see docs/03_arquitetura.md, section 3.6.
            here = _current_environment()
            raise RuntimeError(
                f"Could not load the model bundle at {path}: {exc}\n"
                "This usually means it was pickled under a different "
                "scikit-learn/Python combination than this environment "
                f"provides. This environment: Python {here['python_version']}, "
                f"scikit-learn {here['sklearn_version']}. Retrain a bundle "
                "that matches this environment with `python -m pdm.cli train`."
            ) from exc
        if not isinstance(bundle, ModelBundle):
            raise TypeError(f"{path} does not contain a ModelBundle")
        return bundle
