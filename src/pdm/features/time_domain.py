"""Time-domain statistical features, computed row-wise over a (N, window_len) matrix.

These are the standard condition-monitoring indicators used across the
vibration- and current-analysis literature (see docs/08_revisao_literatura.md)
because they are cheap, robust and; unlike a raw spectrum; directly
interpretable to a maintenance engineer: "crest factor jumped" means
something on the shop floor in a way that "feature_183" does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def time_domain_features(X: np.ndarray, prefix: str) -> pd.DataFrame:
    """Compute one row of features per input row.

    Parameters
    ----------
    X : (n_windows, window_len) array, already cleaned.
    prefix : column-name prefix, typically the sensor name.
    """
    rms = np.sqrt(np.mean(X**2, axis=1))
    peak = np.max(np.abs(X), axis=1)
    mean_abs = np.mean(np.abs(X), axis=1)
    std = X.std(axis=1)
    mean = X.mean(axis=1)

    centered = X - mean[:, None]
    m2 = np.mean(centered**2, axis=1)
    m3 = np.mean(centered**3, axis=1)
    m4 = np.mean(centered**4, axis=1)
    skewness = m3 / (m2**1.5 + EPS)
    kurtosis = m4 / (m2**2 + EPS)  # Pearson kurtosis (Gaussian white noise -> ~3)

    crest_factor = peak / (rms + EPS)
    shape_factor = rms / (mean_abs + EPS)
    impulse_factor = peak / (mean_abs + EPS)
    clearance_factor = peak / (np.mean(np.sqrt(np.abs(X)), axis=1) ** 2 + EPS)

    sign_changes = np.sum(np.diff(np.sign(X), axis=1) != 0, axis=1)
    zero_crossing_rate = sign_changes / X.shape[1]

    peak_to_peak = np.max(X, axis=1) - np.min(X, axis=1)

    return pd.DataFrame(
        {
            f"{prefix}_rms": rms,
            f"{prefix}_std": std,
            f"{prefix}_mean": mean,
            f"{prefix}_peak": peak,
            f"{prefix}_peak_to_peak": peak_to_peak,
            f"{prefix}_skewness": skewness,
            f"{prefix}_kurtosis": kurtosis,
            f"{prefix}_crest_factor": crest_factor,
            f"{prefix}_shape_factor": shape_factor,
            f"{prefix}_impulse_factor": impulse_factor,
            f"{prefix}_clearance_factor": clearance_factor,
            f"{prefix}_zero_crossing_rate": zero_crossing_rate,
        }
    )
