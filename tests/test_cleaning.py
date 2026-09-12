from __future__ import annotations

import numpy as np

from pdm.preprocessing.cleaning import SensorCleaner


def _make_matrix(n_rows: int = 50, window_len: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.1, size=(n_rows, window_len))


def test_saturation_spike_is_clipped() -> None:
    X = _make_matrix()
    X[3, 50] = 5.0  # saturation spike, well outside the local neighborhood
    cleaner = SensorCleaner(saturation_abs_threshold=5.0, hampel_window=7, hampel_n_sigmas=3.0,
                             silent_window_std_percentile=2.0)
    cleaned = cleaner.fit_transform(X)

    assert abs(cleaned[3, 50]) < 1.0
    assert cleaner.report_.saturation_rows_clipped == 1
    # Rows untouched by saturation are left numerically unchanged.
    assert np.allclose(cleaned[0], X[0])


def test_nan_rows_are_interpolated_not_dropped() -> None:
    X = _make_matrix()
    X[5, 10:15] = np.nan
    cleaner = SensorCleaner()
    cleaned = cleaner.fit_transform(X)

    assert not np.isnan(cleaned).any()
    assert cleaned.shape == X.shape
    assert cleaner.report_.nan_rows_imputed == 1


def test_all_nan_row_falls_back_to_zeros_without_crashing() -> None:
    X = _make_matrix()
    X[0] = np.nan
    cleaner = SensorCleaner()
    cleaned = cleaner.fit_transform(X)
    assert np.all(cleaned[0] == 0.0)


def test_silent_windows_are_flagged() -> None:
    X = _make_matrix()
    X[7] = np.full(200, 1e-6)  # essentially silent
    cleaner = SensorCleaner(silent_window_std_percentile=5.0)
    cleaner.fit_transform(X)

    assert cleaner.silent_mask_[7]
    assert cleaner.report_.silent_windows >= 1
