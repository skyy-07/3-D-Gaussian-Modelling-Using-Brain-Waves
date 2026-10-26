"""Spectral conditioning: Welch PSD -> band power -> cognitive salience scalars.

Implements Stage 1 of the blueprint with three deliberate deviations, each of which
matters more than the headline ratio:

1. **Per-session calibration, not per-epoch min/max.** A raw gamma/theta ratio is
   meaningless across subjects and sessions (skull thickness, impedance, amplifier
   gain). We z-score log-ratio against a session baseline with a robust (median/MAD)
   estimator, then squash with a logistic into the documented [0.1, 1.0] range.

2. **Hemifield lateralisation.** O1/O2 are roughly contralateral to the visual
   hemifield. Left/right gamma asymmetry therefore gives a crude horizontal
   attention prior. This is a *prior*, not a decoded coordinate - see docs/SCIENCE.md.

3. **Spectral entropy as certainty.** The gamma/theta ratio measures arousal/vividness.
   It does not measure confidence. We use the concentration of the 4-45 Hz spectrum
   (1 - normalised entropy) as an independent "how sharp was this memory" signal that
   widens or tightens the attention kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

from ..schemas import EpochFeatures
from ..util import clamp, sigmoid
from .synthetic import EEGRecording

#: Canonical posterior / visual-cortex montage used by THINGS-EEG2 and friends.
POSTERIOR_CHANNELS: tuple[str, ...] = ("O1", "O2", "Oz", "P3", "P4", "Pz")

BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass
class PSDResult:
    freqs: np.ndarray  # (n_freqs,)
    psd: np.ndarray  # (n_channels, n_freqs)
    channels: tuple[str, ...]


def compute_psd(
    signal: np.ndarray,
    sfreq: float,
    channels: tuple[str, ...],
    nperseg_s: float = 0.5,
    fmin: float = 1.0,
    fmax: float = 50.0,
) -> PSDResult:
    """Welch PSD per channel, band-limited, with a sane segment length."""
    signal = np.asarray(signal, dtype=np.float64)
    nperseg = int(min(signal.shape[-1], max(32, round(nperseg_s * sfreq))))
    freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg, axis=-1, detrend="constant")
    keep = (freqs >= fmin) & (freqs <= fmax)
    return PSDResult(freqs=freqs[keep], psd=psd[:, keep], channels=channels)


def band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Trapezoidal integral of PSD over ``band`` -> power per channel."""
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if mask.sum() < 2:
        return np.zeros(psd.shape[0], dtype=np.float64)
    return np.trapezoid(psd[:, mask], freqs[mask], axis=1)


def spectral_entropy(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Normalised Shannon entropy of the spectrum inside ``band`` (0 = pure tone)."""
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    sub = np.clip(psd[:, mask], 1e-24, None)
    p = sub / sub.sum(axis=1, keepdims=True)
    h = -(p * np.log(p)).sum(axis=1)
    return h / np.log(max(2, sub.shape[1]))


class SalienceCalibrator:
    """Robust calibration of log gamma/theta into [0.1, 1.0].

    Three modes, because "normalise the numbers" is where most EEG demos quietly
    destroy their own result:

    ``population``
        Fixed normative centre/scale. Keeps *absolute* vividness comparable across
        sessions and between the focus/distracted demo conditions. Default for demos.
    ``session``
        Median/MAD of this recording. Best within-subject contrast, but it forces every
        session to look equally vivid - do not use it to compare conditions.
    ``reference``
        Fit on an explicit rest/eyes-open baseline, then apply to task epochs.
    """

    MODES = ("population", "session", "reference")

    def __init__(self, center: float = -0.08, scale: float = 0.20, gain: float = 1.70):
        self.center = float(center)
        self.scale = max(float(scale), 1e-3)
        self.gain = float(gain)
        self.mode = "population"

    def fit(self, log_ratio: np.ndarray, mode: str = "session") -> SalienceCalibrator:
        x = np.asarray(log_ratio, dtype=np.float64)
        x = x[np.isfinite(x)]
        if x.size >= 4:
            med = float(np.median(x))
            mad = float(np.median(np.abs(x - med))) * 1.4826
            self.center = med
            self.scale = mad if mad > 1e-3 else (float(np.std(x)) or 0.20)
            self.mode = mode
        return self

    def transform_one(self, log_ratio: float) -> float:
        z = (float(log_ratio) - self.center) / self.scale
        return clamp(0.1 + 0.9 * sigmoid(self.gain * z), 0.1, 1.0)

    def transform(self, log_ratio: np.ndarray) -> np.ndarray:
        z = (np.asarray(log_ratio, dtype=np.float64) - self.center) / self.scale
        return np.clip(0.1 + 0.9 / (1.0 + np.exp(-self.gain * z)), 0.1, 1.0)


def _pick_channels(
    channels: tuple[str, ...], requested: tuple[str, ...] | None
) -> list[int]:
    if requested is None:
        requested = POSTERIOR_CHANNELS
    idx = [i for i, c in enumerate(channels) if c in requested]
    return idx or list(range(len(channels)))


def _side_ratio(
    freqs: np.ndarray, psd: np.ndarray, channels: tuple[str, ...], left: str, right: str
) -> float:
    try:
        li = channels.index(left)
        ri = channels.index(right)
    except ValueError:
        return 0.0
    p = band_power(freqs, psd, BANDS["gamma"])
    a, b = float(p[li]), float(p[ri])
    if a + b < 1e-18:
        return 0.0
    return clamp((b - a) / (b + a), -1.0, 1.0)


def analyze_epochs(
    recording: EEGRecording,
    epoch_s: float = 0.75,
    channels: tuple[str, ...] | None = None,
    calibrator: SalienceCalibrator | None = None,
    calibration: str = "population",
    baseline_epochs: int = 0,
) -> tuple[list[EpochFeatures], SalienceCalibrator, dict[str, float]]:
    """Windowed spectral analysis -> one :class:`EpochFeatures` per epoch.

    Returns ``(epochs, fitted_calibrator, raw_band_table)``.
    """
    signal = np.asarray(recording.signal, dtype=np.float64)
    sfreq = recording.sfreq
    ch_idx = _pick_channels(recording.channels, channels)
    ch_names = [recording.channels[i] for i in ch_idx]

    step = max(16, int(epoch_s * sfreq))
    if step < 16:
        raise ValueError("epoch_s too short for the sampling rate")
    n_epochs = signal.shape[1] // step

    raws: list[dict[str, float]] = []
    for ep in range(n_epochs):
        seg = signal[ch_idx, ep * step : (ep + 1) * step]
        psd_res = compute_psd(seg, sfreq, tuple(ch_names), nperseg_s=min(epoch_s, 0.5))
        f, p = psd_res.freqs, psd_res.psd
        bp = {name: band_power(f, p, band) for name, band in BANDS.items()}
        total = sum(bp.values()) + 1e-18
        mean = {name: float(np.mean(v)) for name, v in bp.items()}

        theta = max(mean["theta"], 1e-12)
        gamma = max(mean["gamma"], 1e-12)
        ent = float(np.mean(spectral_entropy(f, p, (4.0, 45.0))))

        raws.append(
            {
                "theta": theta,
                "alpha": mean["alpha"],
                "beta": mean["beta"],
                "gamma": gamma,
                "gamma_theta": gamma / theta,
                "log_ratio": float(np.log10(gamma / theta)),
                "theta_rel": mean["theta"] / total,
                "alpha_rel": mean["alpha"] / total,
                "gamma_rel": mean["gamma"] / total,
                # entropy of a flat spectrum is high; concentration -> certainty
                # 0.45 is the empirical ceiling of 1-H/log(N) for posterior EEG; keeps the
                # spread term in a useful range instead of pinned at "very uncertain"
                "certainty": clamp((1.0 - ent) / 0.45, 0.0, 1.0),
                "lateralization": _side_ratio(f, p, tuple(ch_names), "O1", "O2"),
            }
        )

    log_ratios = np.array([r["log_ratio"] for r in raws])
    # Adjacent windows are autocorrelated; a 3-tap kernel suppresses single-epoch
    # noise without inventing a trend. Raw per-band powers stay in `epochs` untouched.
    if log_ratios.size >= 3:
        kernel = np.array([0.25, 0.5, 0.25])
        padded = np.pad(log_ratios, 1, mode="edge")
        log_ratios = np.convolve(padded, kernel, mode="valid")
    if calibrator is not None:
        cal = calibrator
    elif calibration == "session":
        cal = SalienceCalibrator().fit(log_ratios, mode="session")
    elif calibration == "reference" and 0 < baseline_epochs < len(raws):
        cal = SalienceCalibrator().fit(log_ratios[:baseline_epochs], mode="reference")
    else:
        cal = SalienceCalibrator()
        cal.mode = "population"
    salience = cal.transform(log_ratios)

    epochs: list[EpochFeatures] = []
    for i, r in enumerate(raws):
        # certainty is itself percent-normalised across the session so the UI slider
        # has contrast without hard-coded microvolt assumptions
        epochs.append(
            EpochFeatures(
                index=i,
                t0=round(i * epoch_s, 4),
                t1=round((i + 1) * epoch_s, 4),
                theta=round(r["theta"], 6),
                alpha=round(r["alpha"], 6),
                beta=round(r["beta"], 6),
                gamma=round(r["gamma"], 6),
                gamma_theta=round(r["gamma_theta"], 6),
                lateralization=round(r["lateralization"], 4),
                certainty=round(float(r["certainty"]), 4),
                salience=round(float(salience[i]), 4),
                focus_index=round(
                    clamp(0.65 * float(salience[i]) + 0.35 * float(r["certainty"]), 0.0, 1.0), 4
                ),
                channels_used=list(ch_names),
            )
        )

    table = {
        "theta_rel_mean": float(np.mean([r["theta_rel"] for r in raws])),
        "alpha_rel_mean": float(np.mean([r["alpha_rel"] for r in raws])),
        "gamma_rel_mean": float(np.mean([r["gamma_rel"] for r in raws])),
        "gamma_theta_mean": float(np.mean([r["gamma_theta"] for r in raws])),
        "calibrator_center": cal.center,
        "calibrator_scale": cal.scale,
        "calibration_mode": cal.mode,
        "sfreq": sfreq,
    }
    return epochs, cal, table
