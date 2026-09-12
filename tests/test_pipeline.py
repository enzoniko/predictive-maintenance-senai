from __future__ import annotations

from pdm.config import Config
from pdm.models.pipeline import run_training_pipeline


def test_training_pipeline_runs_end_to_end_on_synthetic_data(synthetic_config: Config) -> None:
    artifacts = run_training_pipeline(synthetic_config)

    expected_models = {"decision_tree", "random_forest", "hist_gradient_boosting", "xgboost"}
    assert artifacts.best_model_name in expected_models
    assert artifacts.bundle.classes == ["Classe A", "Classe B", "Classe C", "Classe D", "Classe E"]
    assert set(artifacts.bundle.sensor_names) == {"Dados_1", "Dados_2", "Dados_3"}

    # The synthetic sensors are constructed to be genuinely class-informative
    # (see tests/conftest.py), so the selected model should do much better
    # than chance (0.2 for 5 balanced classes).
    assert artifacts.test_classification_report["macro avg"]["f1-score"] > 0.5

    assert 0.0 <= artifacts.conformal_coverage <= 1.0
    assert artifacts.conformal_avg_set_size >= 0.0

    assert artifacts.label_shuffle_control_f1 < 0.45  # chance is 0.2, allow slack
    # The excluded (stuck/noise) sensor's control should also stay near chance.
    import math

    assert math.isnan(artifacts.noise_sensor_control_f1) or artifacts.noise_sensor_control_f1 < 0.45

    assert len(artifacts.sensor_dropout) == 1 + len(artifacts.bundle.sensor_names)
    assert len(artifacts.feature_group_ablation) == 4


def test_training_pipeline_bundle_round_trips(tmp_path, synthetic_config: Config) -> None:
    artifacts = run_training_pipeline(synthetic_config)
    path = artifacts.bundle.save(tmp_path / "bundle.joblib")

    from pdm.models.bundle import ModelBundle

    loaded = ModelBundle.load(path)
    assert loaded.model_name == artifacts.bundle.model_name
    assert loaded.classes == artifacts.bundle.classes

    import pandas as pd

    sample = pd.DataFrame(
        [[0.0] * len(loaded.feature_columns)], columns=loaded.feature_columns
    )
    pred = loaded.model.predict(sample)
    assert pred[0] in loaded.classes
