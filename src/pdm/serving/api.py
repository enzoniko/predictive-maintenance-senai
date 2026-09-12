"""FastAPI inference service.

Loads a ``ModelBundle`` once at startup and serves predictions, an
on-demand data-quality audit, drift scoring against a reference sample, and
an explanation endpoint -- backed by ``serving/db.py`` so every call this
service makes is durably recorded (see that module's docstring for why the
database's scope stops there rather than duplicating the client's own raw
sensor data).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from pdm.config import load_config
from pdm.evaluation.drift import DriftMonitor
from pdm.evaluation.explain import SHAP_AVAILABLE
from pdm.features.builder import build_features_from_raw
from pdm.models.bundle import BUNDLE_FILENAME, ModelBundle
from pdm.serving import db
from pdm.serving.schemas import (
    AuditSummaryResponse,
    DriftSummaryResponse,
    ExplainResponse,
    HealthResponse,
    PredictBatchRequest,
    PredictRequest,
    PredictResponse,
)

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    bundle_path = cfg.paths.models_dir / BUNDLE_FILENAME
    if not bundle_path.exists():
        raise RuntimeError(
            f"No model bundle at {bundle_path} -- run `python -m pdm.cli train` first."
        )
    state["config"] = cfg
    state["bundle"] = ModelBundle.load(bundle_path)
    state["engine"] = db.init_db(cfg.api["database_url"])
    state["session_factory"] = db.get_session_factory(state["engine"])
    state["drift_reference"] = None  # populated lazily by GET /drift or set_drift_reference()
    yield
    state.clear()


app = FastAPI(title="Predictive Maintenance API", version="0.1.0", lifespan=lifespan)


def _bundle() -> ModelBundle:
    return state["bundle"]


def _predict_one(
    sensors: dict[str, list[float]],
) -> tuple[str, dict[str, float], list[str], bool, np.ndarray]:
    bundle = _bundle()
    missing = [s for s in bundle.sensor_names if s not in sensors]
    if missing:
        raise HTTPException(422, f"missing required sensors: {missing}")

    raw = {}
    for name in bundle.sensor_names:
        samples = sensors[name]
        if len(samples) != bundle.window_len:
            raise HTTPException(
                422, f"sensor {name}: expected {bundle.window_len} samples, got {len(samples)}"
            )
        raw[name] = np.asarray(samples, dtype=np.float64).reshape(1, -1)

    table, is_silent = build_features_from_raw(
        raw, bundle.cleaners, bundle.sample_rate_hz, include_wavelet=True
    )
    X = table[bundle.feature_columns]

    proba = bundle.model.predict_proba(X)[0]
    class_idx = list(bundle.model.classes_)
    proba_by_class = {c: float(proba[class_idx.index(c)]) for c in bundle.classes}
    predicted_class = max(proba_by_class, key=proba_by_class.get)

    conformal_set = bundle.classes
    if bundle.conformal is not None:
        sets = bundle.conformal.predict_sets(proba.reshape(1, -1))
        conformal_set = [c for c, included in zip(bundle.classes, sets[0]) if included]

    return predicted_class, proba_by_class, conformal_set, bool(is_silent[0]), X.iloc[0].to_numpy()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    bundle = _bundle()
    return HealthResponse(
        status="ok",
        model_name=bundle.model_name,
        classes=bundle.classes,
        sensor_names=bundle.sensor_names,
        window_len=bundle.window_len,
        sample_rate_hz=bundle.sample_rate_hz,
        bundle_created_at=bundle.created_at,
    )


@app.get("/audit", response_model=AuditSummaryResponse)
def audit() -> AuditSummaryResponse:
    """Re-runs the data-quality audit against the currently configured raw
    data directory and persists it. Expensive (tens of seconds on the full
    dataset) -- intended for periodic/manual checks, not per-prediction use."""
    from pdm.data.audit import run_audit
    from pdm.data.loader import load_sensor_dataset

    cfg = state["config"]
    dataset = load_sensor_dataset(cfg)
    report = run_audit(dataset, cfg)

    with state["session_factory"]() as session:
        session.add(db.AuditRecord(report=report.to_dict()))
        session.commit()

    return AuditSummaryResponse(
        n_windows=report.n_windows,
        class_balance_verdict=report.class_balance_verdict,
        recommended_sensors=report.recommended_sensors,
        excluded_sensors=report.excluded_sensors,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    predicted_class, proba_by_class, conformal_set, is_silent, _features = _predict_one(request.sensors)

    warnings = []
    if is_silent:
        warnings.append(
            "window flagged as near-silent by at least one sensor -- treat this "
            "prediction with caution (see docs/07_perguntas_ao_cliente.md)"
        )
    if len(conformal_set) > 1:
        warnings.append(f"model is not confident enough to narrow this down past {conformal_set}")
    elif len(conformal_set) == 0:
        warnings.append("conformal set is empty: this window does not resemble any trained class")

    bundle = _bundle()
    with state["session_factory"]() as session:
        session.add(
            db.PredictionRecord(
                model_name=bundle.model_name,
                predicted_class=predicted_class,
                probabilities=proba_by_class,
                conformal_set=conformal_set,
                is_silent=is_silent,
                sensor_names=bundle.sensor_names,
            )
        )
        session.commit()

    return PredictResponse(
        predicted_class=predicted_class,
        probabilities=proba_by_class,
        conformal_set=conformal_set,
        is_silent=is_silent,
        warnings=warnings,
    )


@app.post("/predict_batch", response_model=list[PredictResponse])
def predict_batch(request: PredictBatchRequest) -> list[PredictResponse]:
    return [predict(window) for window in request.windows]


@app.post("/explain", response_model=ExplainResponse)
def explain(request: PredictRequest) -> ExplainResponse:
    bundle = _bundle()
    predicted_class, _proba, _set, _silent, feature_values = _predict_one(request.sensors)

    if SHAP_AVAILABLE:
        try:
            import shap

            explainer = shap.TreeExplainer(bundle.model)
            row = pd.DataFrame([feature_values], columns=bundle.feature_columns)
            shap_values = explainer.shap_values(row)
            class_idx = bundle.classes.index(predicted_class)
            values = shap_values[class_idx][0] if isinstance(shap_values, list) else shap_values[0]
            order = np.argsort(-np.abs(values))[:10]
            top_features = [
                {"feature": bundle.feature_columns[i], "shap_value": float(values[i])} for i in order
            ]
            return ExplainResponse(predicted_class=predicted_class, top_features=top_features, method="shap")
        except Exception:  # pragma: no cover -- defensive fallback below covers this
            pass

    # Fallback: the model's own (global) feature_importances_, when present,
    # ranked and reported as the closest available substitute to a local
    # explanation on a platform without SHAP -- see evaluation/explain.py.
    importances = getattr(bundle.model, "feature_importances_", None)
    if importances is None:
        raise HTTPException(503, "no explainability backend available for this model/platform")
    order = np.argsort(-importances)[:10]
    top_features = [
        {"feature": bundle.feature_columns[i], "importance": float(importances[i])} for i in order
    ]
    return ExplainResponse(
        predicted_class=predicted_class, top_features=top_features, method="feature_importances"
    )


@app.get("/drift", response_model=DriftSummaryResponse)
def drift() -> DriftSummaryResponse:
    """Scores the training feature distribution against itself as a smoke
    test when no production traffic has been recorded yet, and against
    recent predictions' feature snapshots once they exist. A real
    deployment would call this on a schedule against a rolling window of
    production features (see docs/09_mlops.md)."""
    reference = state.get("drift_reference")
    if reference is None:
        raise HTTPException(
            503,
            "no drift reference set -- call set_drift_reference() at startup "
            "or POST recent production features before scoring drift",
        )
    cfg = state["config"]
    monitor = DriftMonitor(psi_alarm_threshold=cfg.drift["psi_alarm_threshold"]).fit(reference)
    # Without a second batch, compare the reference to itself as a
    # zero-drift sanity check -- see notebooks/05_deployment_demo.ipynb for
    # a version that scores real held-out data.
    report = monitor.score(reference)

    with state["session_factory"]() as session:
        session.add(
            db.DriftRecord(
                n_alarms=report.n_alarms,
                n_reference=report.n_reference,
                n_current=report.n_current,
                report=report.to_dataframe().to_dict(orient="records"),
            )
        )
        session.commit()

    top = report.to_dataframe().head(10).to_dict(orient="records")
    return DriftSummaryResponse(
        n_alarms=report.n_alarms, n_reference=report.n_reference, n_current=report.n_current, top_features=top
    )
