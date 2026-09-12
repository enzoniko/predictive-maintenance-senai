from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from pdm.evaluation.robustness import (
    feature_group_ablation_test,
    gaussian_noise_injection_test,
    label_shuffle_control,
    sensor_dropout_test,
)

CLASSES = ["Classe A", "Classe B"]


def _make_two_sensor_dataset(n_per_class: int = 150, seed: int = 0):
    """Sensor 1's rms feature is fully informative; sensor 2's is pure noise."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for i, c in enumerate(CLASSES):
        n1_rms = rng.normal(loc=i * 5.0, scale=0.3, size=n_per_class)
        n2_rms = rng.normal(loc=0.0, scale=1.0, size=n_per_class)  # uninformative
        for a, b in zip(n1_rms, n2_rms):
            rows.append({"Dados_1_rms": a, "Dados_2_rms": b, "label": c})
        labels += [c] * n_per_class
    df = pd.DataFrame(rows)
    X = df[["Dados_1_rms", "Dados_2_rms"]]
    y = df["label"].to_numpy()
    return X, y


def test_sensor_dropout_hurts_when_informative_sensor_removed() -> None:
    X, y = _make_two_sensor_dataset()
    model = RandomForestClassifier(n_estimators=100, random_state=0).fit(X, y)
    results = sensor_dropout_test(model, X, y, CLASSES, sensor_prefixes=["Dados_1", "Dados_2"])

    by_scenario = {r.scenario: r for r in results}
    baseline_f1 = by_scenario["baseline (all sensors)"].f1_macro
    assert baseline_f1 > 0.9

    drop_informative = by_scenario["drop_sensor:Dados_1"].f1_macro
    drop_noise = by_scenario["drop_sensor:Dados_2"].f1_macro
    assert drop_informative < drop_noise


def test_feature_group_ablation_runs_and_returns_baseline_first() -> None:
    X, y = _make_two_sensor_dataset()
    model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
    results = feature_group_ablation_test(model, X, y, CLASSES)
    assert results[0].scenario.startswith("baseline")
    assert len(results) == 4  # baseline + time/frequency/wavelet


def test_gaussian_noise_injection_degrades_monotonically_on_average() -> None:
    X, y = _make_two_sensor_dataset()
    model = RandomForestClassifier(n_estimators=100, random_state=0).fit(X, y)
    results = gaussian_noise_injection_test(model, X, y, CLASSES, noise_levels=[0.1, 2.0], seed=0)
    by_scenario = {r.scenario: r for r in results}
    small_noise = by_scenario["gaussian_noise:0.1"].f1_macro
    large_noise = by_scenario["gaussian_noise:2.0"].f1_macro
    assert large_noise <= small_noise + 1e-9


def test_label_shuffle_control_is_near_chance() -> None:
    X, y = _make_two_sensor_dataset(n_per_class=100)
    score = label_shuffle_control(
        lambda: RandomForestClassifier(n_estimators=50, random_state=0), X, y, n_splits=3, seed=0
    )
    assert score < 0.65  # chance for 2 balanced classes is 0.5; allow some slack
