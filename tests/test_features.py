from __future__ import annotations

import numpy as np
import pandas as pd

from pdm.features.frequency_domain import frequency_domain_features
from pdm.features.time_domain import time_domain_features
from pdm.features.wavelet import SSQUEEZEPY_AVAILABLE, cwt_scale_energy_features, wavelet_features

FS = 10_000
N = 200
T = np.arange(N) / FS


def _tone_batch(freqs_hz: list[float]) -> np.ndarray:
    return np.stack([np.sin(2 * np.pi * f * T) for f in freqs_hz])


def test_time_domain_features_shape_and_sanity() -> None:
    # 100 Hz and 500 Hz both divide evenly into quarter-periods at this
    # sample rate, so the discrete samples land exactly on the true peak;
    # an arbitrary frequency can under-sample the peak and is not a fair
    # check of the crest-factor formula itself.
    X = _tone_batch([100, 500])
    df = time_domain_features(X, prefix="s1")
    assert len(df) == 2
    assert (df["s1_rms"] > 0).all()
    # A pure sine's crest factor is sqrt(2) ~= 1.414.
    assert np.allclose(df["s1_crest_factor"], np.sqrt(2), atol=0.01)


def test_time_domain_features_are_deterministic() -> None:
    X = np.random.default_rng(0).normal(size=(5, N))
    df1 = time_domain_features(X, prefix="s1")
    df2 = time_domain_features(X, prefix="s1")
    pd.testing.assert_frame_equal(df1, df2)


def test_frequency_domain_features_locate_dominant_tone() -> None:
    X = _tone_batch([500, 1500])
    df = frequency_domain_features(X, prefix="s1", fs=FS)
    assert np.isclose(df["s1_peak1_freq_hz"].iloc[0], 500, atol=50)
    assert np.isclose(df["s1_peak1_freq_hz"].iloc[1], 1500, atol=50)
    # Bands should sum (approximately) to 1; they partition the spectrum.
    band_cols = [c for c in df.columns if "_band_" in c]
    assert np.allclose(df[band_cols].sum(axis=1), 1.0, atol=1e-6)


def test_frequency_entropy_higher_for_noise_than_pure_tone() -> None:
    rng = np.random.default_rng(0)
    tone = _tone_batch([1000])
    noise = rng.normal(size=(1, N))
    df_tone = frequency_domain_features(tone, prefix="s1", fs=FS)
    df_noise = frequency_domain_features(noise, prefix="s1", fs=FS)
    assert df_noise["s1_spectral_entropy"].iloc[0] > df_tone["s1_spectral_entropy"].iloc[0]


def test_cwt_scale_energy_features_shape() -> None:
    X = _tone_batch([500, 2000])
    df = cwt_scale_energy_features(X, prefix="s1", fs=FS, n_scales=8)
    assert len(df) == 2
    energy_cols = [c for c in df.columns if "energy_frac" in c]
    assert len(energy_cols) == 8
    assert np.allclose(df[energy_cols].sum(axis=1), 1.0, atol=1e-6)


def test_wavelet_features_degrade_gracefully_without_ssqueezepy() -> None:
    X = _tone_batch([500])
    df = wavelet_features(X, prefix="s1", fs=FS)
    ridge_cols = [c for c in df.columns if "ssq_ridge" in c]
    if SSQUEEZEPY_AVAILABLE:
        assert len(ridge_cols) == 3
    else:
        assert len(ridge_cols) == 0
    # Either way, the CWT scale-energy columns must always be present.
    assert any("cwt_scale" in c for c in df.columns)
