from __future__ import annotations

import numpy as np

from pdm.preprocessing.dsp import (
    UnitConverter,
    anti_aliasing_energy_fraction,
    bandpass,
    detrend,
    hilbert_envelope,
)

FS = 10_000
N = 200
T = np.arange(N) / FS


def test_detrend_removes_dc_offset() -> None:
    window = np.full(N, 3.0) + 0.01 * np.sin(2 * np.pi * 500 * T)
    out = detrend(window)
    assert abs(out.mean()) < 1e-9


def test_bandpass_attenuates_out_of_band_tone() -> None:
    in_band = np.sin(2 * np.pi * 500 * T)
    out_of_band = np.sin(2 * np.pi * 3000 * T)
    window = in_band + out_of_band
    filtered = bandpass(window, FS, low_hz=300, high_hz=700)

    # Compare energy near 500 Hz vs 3000 Hz after filtering.
    spectrum = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(N, 1 / FS)
    energy_in_band = spectrum[(freqs > 400) & (freqs < 600)].sum()
    energy_out_of_band = spectrum[(freqs > 2900) & (freqs < 3100)].sum()
    assert energy_in_band > 5 * energy_out_of_band


def test_hilbert_envelope_is_non_negative_and_tracks_amplitude() -> None:
    carrier = np.sin(2 * np.pi * 1000 * T)
    modulated = (1 + 0.5 * np.sin(2 * np.pi * 50 * T)) * carrier
    env = hilbert_envelope(modulated)
    assert np.all(env >= 0)
    assert env.std() > 0


def test_anti_aliasing_fraction_higher_for_near_nyquist_noise() -> None:
    low_freq = np.sin(2 * np.pi * 200 * T)
    near_nyquist = np.sin(2 * np.pi * 4900 * T)
    frac_low = anti_aliasing_energy_fraction(low_freq, FS)
    frac_high = anti_aliasing_energy_fraction(near_nyquist, FS)
    assert frac_high > frac_low


def test_unit_converter_is_identity_when_sensitivity_unknown() -> None:
    converter = UnitConverter({"Dados_1": {"quantity": "unknown", "sensitivity": None}})
    window = np.array([1.0, 2.0, 3.0])
    converted, info = converter.convert(window, "Dados_1")
    assert np.allclose(converted, window)
    assert info.unit == "V"


def test_unit_converter_applies_known_sensitivity() -> None:
    # 100 mV/g accelerometer: 0.1 V -> 1 g.
    converter = UnitConverter(
        {"Dados_1": {"quantity": "acceleration", "sensitivity": 100.0,
                     "sensitivity_unit": "mV/g", "conditioner_gain": 1.0}}
    )
    converted, info = converter.convert(np.array([0.1]), "Dados_1")
    assert np.isclose(converted[0], 1.0)
    assert info.unit == "g"
