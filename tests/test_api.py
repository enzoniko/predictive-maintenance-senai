from __future__ import annotations

import numpy as np
import pytest

from pdm.config import Config
from pdm.models.bundle import BUNDLE_FILENAME
from pdm.models.pipeline import run_training_pipeline


@pytest.fixture(scope="module")
def synthetic_config_module(tmp_path_factory) -> Config:
    # A module-scoped rebuild of the function-scoped `synthetic_config`
    # fixture: training the pipeline once per test module (instead of once
    # per test function) is the difference between ~15s and ~2min here.
    from tests.conftest import synthetic_config as _build_config
    from tests.conftest import synthetic_raw_dir as _build_raw_dir

    tmp_path = tmp_path_factory.mktemp("api_module")
    raw_dir = _build_raw_dir.__wrapped__(tmp_path)
    return _build_config.__wrapped__(raw_dir, tmp_path)


@pytest.fixture(scope="module")
def api_client(synthetic_config_module: Config):
    """A TestClient wired to a bundle trained on the synthetic fixture, so
    the whole request -> feature-building -> model -> conformal -> DB path
    runs for real, without touching the actual ~380 MB dataset."""
    artifacts = run_training_pipeline(synthetic_config_module)
    bundle_path = synthetic_config_module.paths.models_dir / BUNDLE_FILENAME
    artifacts.bundle.save(bundle_path)

    # Also cache a feature table, matching what `python -m pdm.cli features`
    # would produce, so the API's startup-time and POST /drift/reference/reload
    # drift-reference loading (serving/api.py::_load_cached_reference) has
    # something real to find instead of silently staying unset.
    from pdm.data.io import save_feature_table
    from pdm.data.loader import load_sensor_dataset
    from pdm.features.builder import build_feature_table

    dataset = load_sensor_dataset(synthetic_config_module)
    feature_table = build_feature_table(
        dataset, synthetic_config_module, sensors=artifacts.bundle.sensor_names
    )
    save_feature_table(feature_table, synthetic_config_module.paths.processed_dir)

    import pdm.serving.api as api_module

    mp = pytest.MonkeyPatch()
    mp.setattr(api_module, "load_config", lambda: synthetic_config_module)

    from fastapi.testclient import TestClient

    with TestClient(api_module.app) as client:
        yield client, artifacts
    mp.undo()


def _sample_window(config: Config, sensor_names: list[str], seed: int = 0) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    window_len = config.acquisition["window_length_samples"]
    return {name: rng.normal(0, 0.2, size=window_len).tolist() for name in sensor_names}


def test_reference_falls_back_to_bundled_examples_without_a_feature_cache(api_client) -> None:
    # serving/api.py::_load_cached_reference needs *something* to hand LIME
    # as a background sample when the cached feature table isn't there,
    # which is the API container's actual situation (it never ships the
    # ~380 MB raw dataset or its derived cache); this is the fallback that
    # covers that, built from the small bundled configs/sample_windows.json
    # instead, and it must work with a real (not synthetic) window shape.
    from pdm.serving.api import _build_reference_from_bundled_examples

    _client, artifacts = api_client
    reference = _build_reference_from_bundled_examples(artifacts.bundle)
    assert list(reference.columns) == artifacts.bundle.feature_columns
    assert len(reference) == 5  # one real example per class


def test_health_reports_bundle_metadata(api_client) -> None:
    client, artifacts = api_client
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_name"] == artifacts.best_model_name
    assert set(body["sensor_names"]) == set(artifacts.bundle.sensor_names)


def test_predict_returns_a_class_in_the_known_set(api_client, synthetic_config_module: Config) -> None:
    client, artifacts = api_client
    window = _sample_window(synthetic_config_module, artifacts.bundle.sensor_names)

    resp = client.post("/predict", json={"sensors": window})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] in artifacts.bundle.classes
    assert set(body["probabilities"]) == set(artifacts.bundle.classes)
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6
    assert body["predicted_class"] in body["conformal_set"] or len(body["conformal_set"]) == 0


def test_predict_rejects_missing_sensor(api_client, synthetic_config_module: Config) -> None:
    client, artifacts = api_client
    window = _sample_window(synthetic_config_module, artifacts.bundle.sensor_names)
    window.pop(artifacts.bundle.sensor_names[0])

    resp = client.post("/predict", json={"sensors": window})
    assert resp.status_code == 422


def test_predict_rejects_wrong_window_length(api_client, synthetic_config_module: Config) -> None:
    client, artifacts = api_client
    window = _sample_window(synthetic_config_module, artifacts.bundle.sensor_names)
    window[artifacts.bundle.sensor_names[0]] = window[artifacts.bundle.sensor_names[0]][:50]

    resp = client.post("/predict", json={"sensors": window})
    assert resp.status_code == 422


def test_predict_batch_returns_one_response_per_window(api_client, synthetic_config_module: Config) -> None:
    client, artifacts = api_client
    windows = [_sample_window(synthetic_config_module, artifacts.bundle.sensor_names) for _ in range(3)]

    resp = client.post("/predict_batch", json={"windows": [{"sensors": w} for w in windows]})
    assert resp.status_code == 200
    assert len(resp.json()) == 3


def test_predictions_are_persisted_to_the_database(api_client, synthetic_config_module: Config) -> None:
    # The api_client fixture (and its database) is shared across this
    # module's tests, so this only asserts growth, not an absolute count.
    client, artifacts = api_client
    import pdm.serving.api as api_module
    from pdm.serving.db import PredictionRecord

    with api_module.state["session_factory"]() as session:
        before = session.query(PredictionRecord).count()

    window = _sample_window(synthetic_config_module, artifacts.bundle.sensor_names)
    client.post("/predict", json={"sensors": window})

    with api_module.state["session_factory"]() as session:
        rows = session.query(PredictionRecord).all()
    assert len(rows) == before + 1
    assert rows[-1].predicted_class in artifacts.bundle.classes


def test_recent_predictions_reflects_a_just_made_prediction(
    api_client, synthetic_config_module: Config
) -> None:
    client, artifacts = api_client
    window = _sample_window(synthetic_config_module, artifacts.bundle.sensor_names)
    predict_resp = client.post("/predict", json={"sensors": window})
    predicted_class = predict_resp.json()["predicted_class"]

    resp = client.get("/predictions/recent?n=1")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 1
    assert body["predictions"][0]["predicted_class"] == predicted_class


def test_explain_returns_top_features(api_client, synthetic_config_module: Config) -> None:
    client, artifacts = api_client
    window = _sample_window(synthetic_config_module, artifacts.bundle.sensor_names)

    resp = client.post("/explain", json={"sensors": window})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] in artifacts.bundle.classes
    assert len(body["top_features"]) > 0
    assert body["method"] in ("shap", "lime", "feature_importances")


def test_audit_endpoint_runs_and_persists(api_client) -> None:
    client, _artifacts = api_client
    resp = client.get("/audit")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_windows"] == 600
    assert set(body["recommended_sensors"]) <= {"Dados_1", "Dados_2", "Dados_3", "Dados_4", "Dados_5"}


def test_audit_endpoint_returns_actionable_error_without_raw_data(
    api_client, synthetic_config_module: Config
) -> None:
    # Reproduces this API's actual situation in the Docker image, which
    # never ships the ~380 MB raw dataset: a missing Classes.npy must
    # surface as a clear 503, not the bare 500 it did before this was
    # caught by actually running the container end to end.
    client, _artifacts = api_client
    real_labels_path = synthetic_config_module.paths.labels_path
    backup_path = real_labels_path.with_suffix(".npy.bak")
    real_labels_path.rename(backup_path)
    try:
        resp = client.get("/audit")
        assert resp.status_code == 503
        assert "raw sensor data not found" in resp.json()["detail"]
    finally:
        backup_path.rename(real_labels_path)


def test_drift_requires_minimum_production_samples(api_client, synthetic_config_module: Config) -> None:
    # A fresh module-scoped client may already have a few predictions from
    # earlier tests, but not the minimum this endpoint requires.
    client, artifacts = api_client
    import pdm.serving.api as api_module

    with api_module.state["session_factory"]() as session:
        from pdm.serving.db import PredictionRecord

        n_existing = session.query(PredictionRecord).count()

    if n_existing >= api_module.MIN_PRODUCTION_SAMPLES_FOR_DRIFT:
        pytest.skip("enough predictions already recorded by earlier tests to satisfy the minimum")

    resp = client.get("/drift")
    assert resp.status_code == 503


def test_drift_succeeds_once_enough_predictions_are_recorded(
    api_client, synthetic_config_module: Config
) -> None:
    client, artifacts = api_client
    import pdm.serving.api as api_module

    reload_resp = client.post("/drift/reference/reload")
    assert reload_resp.status_code == 200

    for i in range(api_module.MIN_PRODUCTION_SAMPLES_FOR_DRIFT + 5):
        window = _sample_window(synthetic_config_module, artifacts.bundle.sensor_names, seed=i)
        client.post("/predict", json={"sensors": window})

    resp = client.get("/drift")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_current"] >= api_module.MIN_PRODUCTION_SAMPLES_FOR_DRIFT
    assert isinstance(body["n_alarms"], int)
    assert len(body["top_features"]) > 0
