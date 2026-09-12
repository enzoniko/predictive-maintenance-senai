"""Frequency-domain features from the (Hann-windowed, detrended) FFT of each window.

At 10 kHz / 200 samples the frequency resolution is 50 Hz (see
preprocessing/dsp.py's module docstring for what that limitation rules out).
Band edges below are deliberately round multiples of that resolution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pdm.preprocessing.dsp import detrend, hann_window

EPS = 1e-12

DEFAULT_BAND_EDGES_HZ = [0, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]


def frequency_domain_features(
    X: np.ndarray, prefix: str, fs: int, band_edges_hz: list[float] | None = None
) -> pd.DataFrame:
    band_edges_hz = band_edges_hz or DEFAULT_BAND_EDGES_HZ
    conditioned = np.apply_along_axis(lambda r: hann_window(detrend(r)), 1, X)
    spectrum = np.fft.rfft(conditioned, axis=1)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(X.shape[1], d=1.0 / fs)

    total_power = power.sum(axis=1) + EPS
    power_norm = power / total_power[:, None]

    cols: dict[str, np.ndarray] = {}

    for lo, hi in zip(band_edges_hz[:-1], band_edges_hz[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        band_name = f"{prefix}_band_{int(lo)}_{int(hi)}hz"
        cols[band_name] = power_norm[:, mask].sum(axis=1)

    cols[f"{prefix}_spectral_centroid_hz"] = (power_norm * freqs[None, :]).sum(axis=1)
    cols[f"{prefix}_spectral_energy_log"] = np.log(total_power)

    # Spectral entropy: how "spread out" the spectrum is (white noise -> high,
    # a few dominant tonal components -> low).
    p = np.clip(power_norm, EPS, None)
    entropy = -(p * np.log(p)).sum(axis=1) / np.log(power.shape[1])
    cols[f"{prefix}_spectral_entropy"] = entropy

    # Spectral flatness: geometric mean / arithmetic mean of the power
    # spectrum (1.0 for white noise, near 0 for a strongly tonal signal).
    log_p = np.log(p)
    geo_mean = np.exp(log_p.mean(axis=1))
    arith_mean = p.mean(axis=1)
    cols[f"{prefix}_spectral_flatness"] = geo_mean / (arith_mean + EPS)

    # Top-2 spectral peaks (frequency + relative power), useful both as
    # features and, later, as a human-readable explanation ("dominant
    # component at 450 Hz").
    order = np.argsort(-power, axis=1)
    cols[f"{prefix}_peak1_freq_hz"] = freqs[order[:, 0]]
    cols[f"{prefix}_peak1_power_frac"] = power_norm[np.arange(len(X)), order[:, 0]]
    cols[f"{prefix}_peak2_freq_hz"] = freqs[order[:, 1]]
    cols[f"{prefix}_peak2_power_frac"] = power_norm[np.arange(len(X)), order[:, 1]]

    return pd.DataFrame(cols)
