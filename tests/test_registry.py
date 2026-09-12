from __future__ import annotations

from pathlib import Path

from sklearn.dummy import DummyClassifier

from pdm.models.registry import LocalTracker, get_tracker


def test_local_tracker_round_trips_params_metrics_and_model(tmp_path: Path) -> None:
    tracker = LocalTracker(tmp_path)
    run_id = tracker.start_run("unit-test-run")
    tracker.log_params(run_id, {"max_depth": 5})
    tracker.log_metrics(run_id, {"f1_macro": 0.87})

    model = DummyClassifier(strategy="most_frequent")
    model_path = tracker.log_model(run_id, model)

    import json

    run_json = json.loads((tmp_path / run_id / "run.json").read_text(encoding="utf-8"))
    assert run_json["params"] == {"max_depth": 5}
    assert run_json["metrics"] == {"f1_macro": 0.87}
    assert model_path.exists()


def test_get_tracker_returns_a_usable_tracker(tmp_path: Path) -> None:
    tracker = get_tracker(tmp_path)
    run_id = tracker.start_run("smoke-test")
    tracker.log_params(run_id, {"a": 1})
    tracker.log_metrics(run_id, {"acc": 1.0})
    assert isinstance(run_id, str) and len(run_id) > 0
