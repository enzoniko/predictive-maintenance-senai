from __future__ import annotations

import numpy as np
import pandas as pd

from pdm.evaluation.drift import DriftMonitor


def test_no_drift_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"f0": rng.normal(0, 1, 5000), "f1": rng.normal(5, 2, 5000)})
    cur = pd.DataFrame({"f0": rng.normal(0, 1, 5000), "f1": rng.normal(5, 2, 5000)})

    monitor = DriftMonitor(psi_alarm_threshold=0.2).fit(ref)
    report = monitor.score(cur)

    assert report.n_alarms == 0
    for f in report.features:
        assert f.psi < 0.1


def test_shifted_distribution_triggers_alarm() -> None:
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"f0": rng.normal(0, 1, 5000)})
    cur = pd.DataFrame({"f0": rng.normal(4, 1, 5000)})  # large mean shift

    monitor = DriftMonitor(psi_alarm_threshold=0.2).fit(ref)
    report = monitor.score(cur)

    assert report.n_alarms == 1
    assert report.features[0].alarm
    assert report.features[0].ks_p_value < 0.01


def test_report_to_dataframe_is_sorted_by_psi_descending() -> None:
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"stable": rng.normal(0, 1, 2000), "shifted": rng.normal(0, 1, 2000)})
    cur = pd.DataFrame({"stable": rng.normal(0, 1, 2000), "shifted": rng.normal(3, 1, 2000)})

    monitor = DriftMonitor().fit(ref)
    df = monitor.score(cur).to_dataframe()
    assert df.iloc[0]["feature"] == "shifted"
