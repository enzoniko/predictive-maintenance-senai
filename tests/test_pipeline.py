from __future__ import annotations

from pdm.config import Config
from pdm.models.pipeline import evaluate_bundle_on_holdout, run_training_pipeline


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


def test_evaluate_bundle_on_holdout_reproduces_test_metrics(synthetic_config: Config) -> None:
    from sklearn.metrics import f1_score

    artifacts = run_training_pipeline(synthetic_config)
    result = evaluate_bundle_on_holdout(synthetic_config, artifacts.bundle)

    assert len(result["y_test"]) == len(result["y_pred"]) == len(result["proba"])
    assert result["X_test"].shape[1] == len(artifacts.bundle.feature_columns)

    classes = artifacts.bundle.classes
    f1 = f1_score(result["y_test"], result["y_pred"], labels=classes, average="macro", zero_division=0)
    # Reconstructing the same held-out split independently should reproduce
    # (not just approximate) the F1 the training pipeline itself measured.
    assert abs(f1 - artifacts.test_classification_report["macro avg"]["f1-score"]) < 1e-9

    if result["conformal_sets"] is not None:
        assert result["conformal_sets"].shape == (len(result["y_test"]), len(classes))
