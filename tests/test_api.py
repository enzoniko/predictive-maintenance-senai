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

    import pdm.serving.api as api_module

    mp = pytest.MonkeyPatch()
    mp.setattr(api_module, "load_config", lambda: synthetic_config_module)

    from fastapi.testclient import TestClient

    with TestClient(api_module.app) as client:
        yield client, artifacts
    mp.undo()


def _sample_window(config: Config, sensor_names: list[str]) -> dict[str, list[float]]:
    rng = np.random.default_rng(0)
    window_len = config.acquisition["window_length_samples"]
    return {name: rng.normal(0, 0.2, size=window_len).tolist() for name in sensor_names}


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


def test_explain_returns_top_features(api_client, synthetic_config_module: Config) -> None:
    client, artifacts = api_client
    window = _sample_window(synthetic_config_module, artifacts.bundle.sensor_names)

    resp = client.post("/explain", json={"sensors": window})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] in artifacts.bundle.classes
    assert len(body["top_features"]) > 0
    assert body["method"] in ("shap", "feature_importances")


def test_audit_endpoint_runs_and_persists(api_client) -> None:
    client, _artifacts = api_client
    resp = client.get("/audit")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_windows"] == 600
    assert set(body["recommended_sensors"]) <= {"Dados_1", "Dados_2", "Dados_3", "Dados_4", "Dados_5"}
