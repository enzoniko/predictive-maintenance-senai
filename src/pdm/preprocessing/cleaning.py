"""Per-sensor cleaning: saturation removal, NaN handling, silent-window flags.

The ghost column (see ``pdm.data.loader``) is already gone by the time data
reaches this module -- what is handled here is everything the OBS note in
the case statement warns about: "sensors ... podem captar sinais de ruidos e
ate mesmo entrarem em falha." Concretely, for the retained sensors
(Dados_1-3):

* A handful of samples sit exactly at +/-5.0 in every file -- the ADC's
  apparent full-scale range. These are clipped with a Hampel filter (a
  rolling median + MAD outlier detector) rather than dropped, so a window
  keeps its length and its non-saturated context.
* No NaNs were found in the retained sensors on the real dataset, but the
  transformer still imputes them (row-wise linear interpolation, falling
  back to the row median for an all-NaN row) so it does not silently break
  if a future batch does contain gaps.
* ~1-2% of windows are near-silent (very low RMS) in every real sensor and
  never occur in every class equally (see docs/07_perguntas_ao_cliente.md).
  These are flagged, not discarded or imputed -- a model should be allowed
  to abstain on them (see evaluation/conformal.py) rather than being forced
  to guess from what may be a dropout rather than a genuine idle state.

All thresholds are fit on training data only (via ``fit``) and re-applied at
``transform`` time, so a held-out/production batch cannot leak into them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def _hampel_filter(row: np.ndarray, window: int, n_sigmas: float) -> np.ndarray:
    """Rolling-median/MAD outlier replacement, one window (1-D array) at a time."""
    n = len(row)
    half = window // 2
    out = row.copy()
    k = 1.4826  # MAD -> std scale factor for a Gaussian
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        neighborhood = row[lo:hi]
        med = np.median(neighborhood)
        mad = k * np.median(np.abs(neighborhood - med))
        if mad > 0 and abs(row[i] - med) > n_sigmas * mad:
            out[i] = med
    return out


def _hampel_filter_matrix(matrix: np.ndarray, window: int, n_sigmas: float, threshold: float) -> np.ndarray:
    """Apply the Hampel filter only to rows that actually touch the saturation
    threshold -- this is a small fraction of rows (see docs/07), so it keeps
    the (otherwise O(n * window) per row) filter affordable at 50k rows."""
    out = matrix.copy()
    hits = np.where(np.any(np.abs(matrix) >= threshold, axis=1))[0]
    for i in hits:
        out[i] = _hampel_filter(matrix[i], window, n_sigmas)
    return out


def _interpolate_nans(row: np.ndarray) -> np.ndarray:
    mask = np.isnan(row)
    if not mask.any():
        return row
    if mask.all():
        return np.zeros_like(row)
    idx = np.arange(len(row))
    row = row.copy()
    row[mask] = np.interp(idx[mask], idx[~mask], row[~mask])
    return row


@dataclass
class CleaningReport:
    n_rows: int
    saturation_rows_clipped: int
    nan_rows_imputed: int
    silent_window_threshold: float
    silent_windows: int


class SensorCleaner(BaseEstimator, TransformerMixin):
    """Cleans one sensor's (n_windows, window_len) matrix.

    Parameters mirror configs/default.yaml's ``cleaning`` section.
    """

    def __init__(
        self,
        saturation_abs_threshold: float = 5.0,
        hampel_window: int = 7,
        hampel_n_sigmas: float = 3.0,
        silent_window_std_percentile: float = 2.0,
    ) -> None:
        self.saturation_abs_threshold = saturation_abs_threshold
        self.hampel_window = hampel_window
        self.hampel_n_sigmas = hampel_n_sigmas
        self.silent_window_std_percentile = silent_window_std_percentile

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SensorCleaner":
        X = np.asarray(X, dtype=np.float64)
        filled = np.where(np.isnan(X), np.nanmedian(X), X)
        row_std = filled.std(axis=1)
        self.silent_threshold_ = float(np.percentile(row_std, self.silent_window_std_percentile))
        self.report_ = None
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        n_nan_rows = int(np.any(np.isnan(X), axis=1).sum())
        cleaned = np.apply_along_axis(_interpolate_nans, 1, X)

        sat_rows = int(np.any(np.abs(cleaned) >= self.saturation_abs_threshold, axis=1).sum())
        cleaned = _hampel_filter_matrix(
            cleaned, self.hampel_window, self.hampel_n_sigmas, self.saturation_abs_threshold
        )

        row_std = cleaned.std(axis=1)
        silent = row_std <= self.silent_threshold_
        self.report_ = CleaningReport(
            n_rows=X.shape[0],
            saturation_rows_clipped=sat_rows,
            nan_rows_imputed=n_nan_rows,
            silent_window_threshold=self.silent_threshold_,
            silent_windows=int(silent.sum()),
        )
        self.silent_mask_ = silent
        return cleaned

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs) -> np.ndarray:
        return self.fit(X, y).transform(X)
