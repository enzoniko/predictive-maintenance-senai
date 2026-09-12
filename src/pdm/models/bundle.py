"""A self-contained, serializable bundle of everything inference needs.

Bundling the fitted model together with its feature-column order, the class
list, the sensors it expects, and (optionally) a calibrated conformal
predictor means ``serving/api.py`` never has to guess these things or risk
drifting out of sync with how the model was trained -- load one file, get a
consistent, versioned unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from pdm.evaluation.conformal import SplitConformalClassifier
from pdm.preprocessing.cleaning import SensorCleaner

BUNDLE_FILENAME = "model_bundle.joblib"


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
    # reused verbatim at inference time -- serving/api.py must not re-fit
    # these against production traffic.
    cleaners: dict[str, SensorCleaner] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> "ModelBundle":
        bundle = joblib.load(path)
        if not isinstance(bundle, ModelBundle):
            raise TypeError(f"{path} does not contain a ModelBundle")
        return bundle
