"""Digital signal-processing conditioning steps.

These are the checks and transforms a vibration/current-analysis engineer
would reach for before trusting a spectrum, kept separate from
``cleaning.py`` (which handles data-quality defects, not signal physics).
Every function operates on one window (1-D array of length ``window_len``)
or a batch of them and is unit-agnostic: see ``UnitConverter`` and
configs/sensors.yaml for why this pipeline stays in raw volts until the
client supplies transducer sensitivities.

Two structural limits apply to every function here and are worth restating
because they are easy to lose track of once the numbers are in a table:
* 200 samples at 10 kHz is a 20 ms window -> 50 Hz frequency resolution.
  Bearing fault characteristic frequencies (BPFO/BPFI/BSF/FTF) and their
  sidebands routinely sit within a few Hz of each other and of running
  speed harmonics; 50 Hz resolution cannot separate them. Order tracking
  additionally requires a shaft RPM signal, which is not provided.
* The mains line frequency (50/60 Hz) is likewise not resolvable from noise
  at this resolution, so no notch filter is applied by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


def detrend(window: np.ndarray) -> np.ndarray:
    """Remove the DC offset (and any linear trend) from a window."""
    return signal.detrend(window, type="linear")


def hann_window(window: np.ndarray) -> np.ndarray:
    """Apply a Hann taper -- reduces spectral leakage before an FFT."""
    return window * np.hanning(len(window))


def anti_aliasing_energy_fraction(window: np.ndarray, fs: int, band_fraction: float = 0.1) -> float:
    """Fraction of spectral energy in the top ``band_fraction`` of the Nyquist
    band. A healthy anti-aliasing filter should leave this near zero; a
    non-trivial fraction suggests aliased content folded in from above
    Nyquist, which no amount of downstream feature engineering can undo."""
    spectrum = np.abs(np.fft.rfft(hann_window(detrend(window))))
    power = spectrum**2
    n = len(power)
    cutoff = int(n * (1 - band_fraction))
    total = power.sum()
    if total <= 0:
        return 0.0
    return float(power[cutoff:].sum() / total)


def bandpass(window: np.ndarray, fs: int, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter."""
    nyq = fs / 2
    low_hz = max(low_hz, 1e-3)
    high_hz = min(high_hz, nyq * 0.99)
    sos = signal.butter(order, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")
    return signal.sosfiltfilt(sos, window)


def hilbert_envelope(window: np.ndarray) -> np.ndarray:
    """Amplitude envelope via the analytic signal -- the standard first step
    of envelope-demodulation analysis for rolling-element bearing faults
    (typically applied after band-passing around a resonance, then this
    envelope's own spectrum is inspected for BPFO/BPFI lines)."""
    return np.abs(signal.hilbert(window))


@dataclass
class UnitConversion:
    quantity: str
    scale: float  # multiply raw volts by this to get physical units
    unit: str


class UnitConverter:
    """Converts raw-volt windows to physical units using configs/sensors.yaml.

    With every sensor's ``sensitivity`` still ``null`` (unknown transducer,
    see configs/sensors.yaml), this is an identity operation today -- calling
    it makes that assumption explicit and auditable rather than silently
    treating volts as if they were already the physical quantity of
    interest. Once the client supplies sensitivities this class starts doing
    real unit conversion (e.g. mV/g -> g for an IEPE accelerometer) without
    any caller needing to change.
    """

    def __init__(self, sensor_metadata: dict[str, dict]) -> None:
        self.sensor_metadata = sensor_metadata

    def get(self, sensor_name: str) -> UnitConversion:
        meta = self.sensor_metadata.get(sensor_name, {})
        sensitivity = meta.get("sensitivity")
        gain = meta.get("conditioner_gain", 1.0)
        if sensitivity is None:
            return UnitConversion(quantity="voltage (raw, unconverted)", scale=1.0, unit="V")
        # sensitivity is given in mV per physical unit.
        scale = 1000.0 / (sensitivity * gain)
        return UnitConversion(quantity=meta.get("quantity", "unknown"), scale=scale,
                               unit=meta.get("sensitivity_unit", "?").split("/")[-1])

    def convert(self, window: np.ndarray, sensor_name: str) -> tuple[np.ndarray, UnitConversion]:
        conversion = self.get(sensor_name)
        return window * conversion.scale, conversion
