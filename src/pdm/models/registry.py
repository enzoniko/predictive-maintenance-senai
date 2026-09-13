"""Experiment tracking, with a local fallback when MLflow is unavailable.

``mlflow`` needs ``pyarrow``, which requires a C/C++ toolchain to build
from source and is not available in every environment (see
docs/03_arquitetura.md, section 3.6). Rather than make MLflow a hard
requirement for running this pipeline at all, ``get_tracker()`` returns an
MLflow-backed tracker when importable and a small local JSON+joblib
tracker otherwise, with the same three-method interface. Every run is
reproducible and inspectable either way; only the UI differs (``mlflow ui``
vs. reading ``models/runs/<run_id>/``).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import joblib

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class Tracker(Protocol):
    def start_run(self, run_name: str) -> str: ...
    def log_params(self, run_id: str, params: dict[str, Any]) -> None: ...
    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None: ...
    def log_model(self, run_id: str, model: Any, filename: str = "model.joblib") -> Path: ...


class MlflowTracker:
    def __init__(self, experiment_name: str = "predictive-maintenance") -> None:
        mlflow.set_experiment(experiment_name)
        self._active: dict[str, Any] = {}

    def start_run(self, run_name: str) -> str:
        run = mlflow.start_run(run_name=run_name)
        self._active[run.info.run_id] = run
        mlflow.end_run()
        return run.info.run_id

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_params(params)

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)

    def log_model(self, run_id: str, model: Any, filename: str = "model.joblib") -> Path:
        with mlflow.start_run(run_id=run_id):
            mlflow.sklearn.log_model(model, "model")
        return Path("mlruns")  # MLflow manages its own storage layout.


@dataclass
class _LocalRun:
    run_id: str
    run_name: str
    started_at: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


class LocalTracker:
    """Zero-dependency tracker: one directory per run under ``models/runs/``."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._runs: dict[str, _LocalRun] = {}

    def _run_dir(self, run_id: str) -> Path:
        d = self.base_dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save(self, run: _LocalRun) -> None:
        path = self._run_dir(run.run_id) / "run.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(run.__dict__, fh, indent=2, ensure_ascii=False)

    def start_run(self, run_name: str) -> str:
        run_id = uuid.uuid4().hex[:12]
        run = _LocalRun(run_id=run_id, run_name=run_name,
                         started_at=datetime.now(timezone.utc).isoformat())
        self._runs[run_id] = run
        self._save(run)
        return run_id

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        run = self._runs[run_id]
        run.params.update(params)
        self._save(run)

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        run = self._runs[run_id]
        run.metrics.update(metrics)
        self._save(run)

    def log_model(self, run_id: str, model: Any, filename: str = "model.joblib") -> Path:
        path = self._run_dir(run_id) / filename
        joblib.dump(model, path)
        return path


def get_tracker(base_dir: Path, experiment_name: str = "predictive-maintenance") -> Tracker:
    if MLFLOW_AVAILABLE:
        return MlflowTracker(experiment_name)
    return LocalTracker(base_dir)
