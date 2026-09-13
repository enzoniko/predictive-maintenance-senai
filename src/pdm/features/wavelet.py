"""Wavelet-domain features: continuous-wavelet-transform (CWT) scale-energy,
plus synchrosqueezed-CWT ridge features when ``ssqueezepy`` is available.

Two implementations, on purpose:

* ``cwt_scale_energy_features`` uses PyWavelets (pure C extension, ships
  prebuilt wheels everywhere this project targets) for a Morlet CWT energy
  distribution across scales; always available, always the baseline.
* ``ssq_ridge_features`` uses ``ssqueezepy`` for a synchrosqueezed CWT and a
  simple ridge extraction, giving a much sharper time-frequency
  localization (useful for both features and visualization; see
  notebooks/02_signal_analysis.ipynb). ``ssqueezepy`` pulls in
  numba/llvmlite, which need a C/C++ toolchain to build from source, so
  this path is wrapped in a guarded optional import
  (``SSQUEEZEPY_AVAILABLE``, see docs/03_arquitetura.md, section 3.6).
  Callers must not assume the ridge columns are always present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pywt

EPS = 1e-12
DEFAULT_N_SCALES = 24
DEFAULT_MIN_FREQ_HZ = 100
DEFAULT_MAX_FREQ_HZ = 4500

try:
    import ssqueezepy  # noqa: F401

    SSQUEEZEPY_AVAILABLE = True
except ImportError:
    SSQUEEZEPY_AVAILABLE = False


def _scales_for_frequency_range(fs: int, min_freq: float, max_freq: float, n_scales: int) -> np.ndarray:
    """PyWavelets scales spaced log-uniformly over [min_freq, max_freq]."""
    freqs = np.geomspace(max_freq, min_freq, n_scales)
    central_freq = pywt.central_frequency("morl")
    return central_freq * fs / freqs


def cwt_scale_energy_features(
    X: np.ndarray,
    prefix: str,
    fs: int,
    n_scales: int = DEFAULT_N_SCALES,
    min_freq_hz: float = DEFAULT_MIN_FREQ_HZ,
    max_freq_hz: float = DEFAULT_MAX_FREQ_HZ,
) -> pd.DataFrame:
    """Per-window energy fraction in each wavelet scale (~frequency band),
    plus a wavelet-domain entropy analogous to the spectral entropy in
    frequency_domain.py."""
    scales = _scales_for_frequency_range(fs, min_freq_hz, max_freq_hz, n_scales)
    n_rows = X.shape[0]
    energy = np.empty((n_rows, n_scales))
    for i in range(n_rows):
        coeffs, _ = pywt.cwt(X[i], scales, "morl", sampling_period=1.0 / fs)
        energy[i] = np.sum(np.abs(coeffs) ** 2, axis=1)

    total = energy.sum(axis=1, keepdims=True) + EPS
    energy_norm = energy / total

    cols = {
        f"{prefix}_cwt_scale{j}_energy_frac": energy_norm[:, j] for j in range(n_scales)
    }
    p = np.clip(energy_norm, EPS, None)
    cols[f"{prefix}_cwt_entropy"] = -(p * np.log(p)).sum(axis=1) / np.log(n_scales)
    cols[f"{prefix}_cwt_energy_log"] = np.log(total.ravel())
    dominant = np.argmax(energy_norm, axis=1)
    cols[f"{prefix}_cwt_dominant_scale_idx"] = dominant.astype(float)
    return pd.DataFrame(cols)


def ssq_ridge_features(X: np.ndarray, prefix: str, fs: int) -> pd.DataFrame | None:
    """Synchrosqueezed-CWT ridge frequency (mean/std) per window.

    Returns None if ssqueezepy is not importable in the current
    environment; callers (features/builder.py) treat that as "these
    columns are unavailable here", not as an error.
    """
    if not SSQUEEZEPY_AVAILABLE:
        return None

    from ssqueezepy import ssq_cwt

    n_rows = X.shape[0]
    ridge_mean = np.empty(n_rows)
    ridge_std = np.empty(n_rows)
    ridge_power_concentration = np.empty(n_rows)

    for i in range(n_rows):
        Tx, _, ssq_freqs, *_ = ssq_cwt(X[i], fs=fs)
        power = np.abs(Tx) ** 2
        ridge_idx = np.argmax(power, axis=0)
        ridge_freqs = ssq_freqs[ridge_idx]
        ridge_mean[i] = ridge_freqs.mean()
        ridge_std[i] = ridge_freqs.std()
        peak_power = power[ridge_idx, np.arange(power.shape[1])]
        ridge_power_concentration[i] = peak_power.sum() / (power.sum() + EPS)

    return pd.DataFrame(
        {
            f"{prefix}_ssq_ridge_freq_mean_hz": ridge_mean,
            f"{prefix}_ssq_ridge_freq_std_hz": ridge_std,
            f"{prefix}_ssq_ridge_power_concentration": ridge_power_concentration,
        }
    )


def wavelet_features(X: np.ndarray, prefix: str, fs: int) -> pd.DataFrame:
    """CWT scale-energy features, plus ridge features when available."""
    df = cwt_scale_energy_features(X, prefix, fs)
    ridge_df = ssq_ridge_features(X, prefix, fs)
    if ridge_df is not None:
        df = pd.concat([df, ridge_df], axis=1)
    return df
