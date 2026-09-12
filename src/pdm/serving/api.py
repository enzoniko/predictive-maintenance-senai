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
    state["drift_reference"] = _load_cached_reference(cfg, state["bundle"])
    yield
    state.clear()


def _load_cached_reference(cfg, bundle: ModelBundle):
    """Best-effort initial drift reference from the cached feature table
    (``python -m pdm.cli features``), restricted to this bundle's feature
    columns. Returns None (drift scoring then requires an explicit
    ``POST /drift/reference`` call) if no cache exists or its columns don't
    match -- this is a convenience default, not a hard dependency."""
    from pdm.data.io import load_feature_table

    try:
        table = load_feature_table(cfg.paths.processed_dir)
        return table[bundle.feature_columns]
    except (FileNotFoundError, KeyError):
        return None


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


def _predict_and_build_record(sensors: dict[str, list[float]]) -> tuple[PredictResponse, db.PredictionRecord]:
    bundle = _bundle()
    predicted_class, proba_by_class, conformal_set, is_silent, feature_values = _predict_one(sensors)

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

    response = PredictResponse(
        predicted_class=predicted_class,
        probabilities=proba_by_class,
        conformal_set=conformal_set,
        is_silent=is_silent,
        warnings=warnings,
    )
    record = db.PredictionRecord(
        model_name=bundle.model_name,
        predicted_class=predicted_class,
        probabilities=proba_by_class,
        conformal_set=conformal_set,
        is_silent=is_silent,
        sensor_names=bundle.sensor_names,
        features=dict(zip(bundle.feature_columns, (float(v) for v in feature_values))),
    )
    return response, record


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    response, record = _predict_and_build_record(request.sensors)
    with state["session_factory"]() as session:
        session.add(record)
        session.commit()
    return response


@app.post("/predict_batch", response_model=list[PredictResponse])
def predict_batch(request: PredictBatchRequest) -> list[PredictResponse]:
    """Scores every window and persists all of them in a single transaction.

    Calling POST /predict in a loop from a client -- as notebooks/03's demo
    originally did to generate a batch of production traffic for GET /drift
    -- means one HTTP round trip *and* one fsync'd DB commit per window;
    at a couple hundred windows that took over 10 minutes end to end and
    is what actually motivated this endpoint's batching to be real instead
    of just a thin loop over ``predict()``, which is what it was before.
    """
    responses, records = [], []
    for window in request.windows:
        response, record = _predict_and_build_record(window.sensors)
        responses.append(response)
        records.append(record)

    with state["session_factory"]() as session:
        session.add_all(records)
        session.commit()

    return responses


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

    # Fallback 2: LIME, a local (per-instance) explanation like SHAP -- and,
    # unlike the feature_importances_ fallback this replaced, one that
    # actually exists for every model here: HistGradientBoostingClassifier
    # (the model this bundle's own training run selected) has no
    # feature_importances_ attribute at all, so on a platform without SHAP
    # (this Windows ARM64 dev machine, see docs/03_arquitetura.md sec. 3.6)
    # the old fallback always raised 503 -- caught by notebooks/03's demo
    # run, not a hypothetical. LIME needs a background sample to perturb
    # around; the cached drift reference (already loaded at startup) serves
    # that purpose without requiring the full training set at inference time.
    reference = state.get("drift_reference")
    if reference is not None:
        try:
            from pdm.evaluation.explain import lime_explain_instance

            row = pd.Series(feature_values, index=bundle.feature_columns)
            lime_exp = lime_explain_instance(
                bundle.model, reference, row, class_names=bundle.classes, num_features=10
            )
            label = lime_exp.available_labels()[0]
            top_features = [
                {"feature": f, "weight": float(w)} for f, w in lime_exp.as_list(label=label)
            ]
            return ExplainResponse(predicted_class=predicted_class, top_features=top_features, method="lime")
        except Exception:  # pragma: no cover -- defensive fallback below covers this
            pass

    # Fallback 3: the model's own (global) feature_importances_, when present.
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


# PSI (evaluation/drift.py) bins the reference into deciles by default;
# with too few current-batch samples per bin, sampling noise alone produces
# large PSI values and floods the report with false alarms -- observed
# directly running notebooks/03_api_demo.ipynb with ~40 samples drawn from
# the *same* distribution as the reference (no real drift possible) and
# still getting 143/176 features flagged. 200 keeps roughly 20+ samples per
# decile bin, which is a more defensible floor; still a coarse rule of
# thumb, not a substitute for reviewing a real production drift report.
MIN_PRODUCTION_SAMPLES_FOR_DRIFT = 200


@app.post("/drift/reference/reload")
def reload_drift_reference() -> dict:
    """Re-reads the cached feature table (``python -m pdm.cli features``) as
    the drift reference -- call this after retraining against new data."""
    cfg = state["config"]
    reference = _load_cached_reference(cfg, _bundle())
    state["drift_reference"] = reference
    n_reference = 0 if reference is None else len(reference)
    return {"reference_loaded": reference is not None, "n_reference": n_reference}


@app.get("/drift", response_model=DriftSummaryResponse)
def drift(n: int = 200) -> DriftSummaryResponse:
    """Scores the ``n`` most recent real predictions' feature vectors
    (persisted by POST /predict) against the training reference. Requires
    at least ``MIN_PRODUCTION_SAMPLES_FOR_DRIFT`` recorded predictions --
    early in a deployment's life, before that much traffic exists, this
    correctly refuses to report a drift score rather than comparing the
    reference against itself and calling that a measurement. A real
    deployment calls this on a schedule (see docs/09_mlops.md)."""
    reference = state.get("drift_reference")
    if reference is None:
        raise HTTPException(
            503,
            "no drift reference available -- run `python -m pdm.cli features` "
            "then POST /drift/reference/reload, or retrain",
        )

    with state["session_factory"]() as session:
        rows = (
            session.query(db.PredictionRecord.features)
            .order_by(db.PredictionRecord.id.desc())
            .limit(n)
            .all()
        )
    if len(rows) < MIN_PRODUCTION_SAMPLES_FOR_DRIFT:
        raise HTTPException(
            503,
            f"only {len(rows)} recorded predictions available, need at least "
            f"{MIN_PRODUCTION_SAMPLES_FOR_DRIFT} to score drift meaningfully -- "
            "call POST /predict more (or lower n) before checking drift",
        )

    current = pd.DataFrame([r.features for r in rows])
    cfg = state["config"]
    monitor = DriftMonitor(psi_alarm_threshold=cfg.drift["psi_alarm_threshold"]).fit(reference)
    report = monitor.score(current)

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
