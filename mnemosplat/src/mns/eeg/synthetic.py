"""Synthetic posterior EEG with controllable salience dynamics.

Purpose
-------
The scientific claim of MnemoSplat is that *posterior gamma/theta dynamics* gate
how much geometric density a remembered entity receives. To develop and demo the
system without a subject, we need EEG that is (a) spectrally realistic, (b) has a
known ground-truth salience schedule so the fusion can be unit-tested, and
(c) looks like the real thing to the PSD extractor.

This module generates 1/f cortical background + posterior alpha + theta + gamma
bursts, with per-epoch focus and hemifield lateralisation drives. It is clearly
labelled synthetic everywhere it surfaces (manifest.upstream), so a judge can
never mistake it for recorded data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_CHANNELS: tuple[str, ...] = ("O1", "O2", "Oz", "P3", "P4", "Pz")

#: rough posterior topography gain per channel, used so O1/O2 are not identical
_CHANNEL_GAIN = {"O1": 1.00, "O2": 1.00, "Oz": 1.15, "P3": 0.72, "P4": 0.72, "Pz": 0.80}


@dataclass
class EEGRecording:
    signal: np.ndarray  # (n_channels, n_samples), microvolts
    sfreq: float
    channels: tuple[str, ...]
    synthetic: bool = True
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.signal.ndim != 2:
            raise ValueError("signal must be (channels, samples)")
        if self.signal.shape[0] != len(self.channels):
            raise ValueError("channel count mismatch")

    @property
    def duration_s(self) -> float:
        return self.signal.shape[1] / self.sfreq


def _pink_noise(
    rng: np.random.Generator, n: int, sfreq: float, exponent: float = 1.1
) -> np.ndarray:
    """Unit-variance 1/f^exponent noise via spectral shaping."""
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    spectrum = (rng.normal(size=freqs.size) + 1j * rng.normal(size=freqs.size)) / np.power(
        freqs + 0.5, exponent / 2.0
    )
    spectrum[0] = 0.0
    x = np.fft.irfft(spectrum, n=n)
    std = float(np.std(x))
    return x / std if std > 1e-12 else x


def _burst(
    rng: np.random.Generator, n: int, sfreq: float, f0: float, amp: float, am_hz: float = 0.7
) -> np.ndarray:
    t = np.arange(n) / sfreq
    phase = rng.uniform(0, 2 * np.pi)
    envelope = 1.0 + 0.35 * np.sin(2 * np.pi * am_hz * t + rng.uniform(0, 2 * np.pi))
    return amp * envelope * np.sin(2 * np.pi * f0 * t + phase)


def generate_eeg(
    duration_s: float = 6.0,
    sfreq: float = 250.0,
    channels: tuple[str, ...] = DEFAULT_CHANNELS,
    epoch_s: float = 0.75,
    focus_schedule: np.ndarray | None = None,
    lateralization_schedule: np.ndarray | None = None,
    seed: int = 7,
) -> EEGRecording:
    """Build a multichannel posterior recording.

    Parameters
    ----------
    focus_schedule
        One value in [0, 1] per epoch. 1 = vivid, high-gamma recall; 0 = effortful,
        theta-dominated retrieval (diffuse memory). Defaults to a smooth rise/fall.
    lateralization_schedule
        One value in [-1, 1] per epoch. Positive = right occipital dominance, which
        the fusion stage uses as a hemifield attention prior.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * sfreq))
    n_epochs = max(1, int(round(duration_s / epoch_s)))

    if focus_schedule is None:
        x = np.linspace(0, np.pi, n_epochs)
        focus_schedule = 0.35 + 0.55 * np.sin(x) ** 2
    focus_schedule = np.clip(np.asarray(focus_schedule, dtype=np.float64).ravel(), 0.0, 1.0)
    if focus_schedule.size != n_epochs:
        focus_schedule = np.interp(
            np.linspace(0, 1, n_epochs), np.linspace(0, 1, focus_schedule.size), focus_schedule
        )
    if lateralization_schedule is None:
        lateralization_schedule = 0.45 * np.sin(np.linspace(0, 2 * np.pi, n_epochs))
    lateralization_schedule = np.clip(
        np.asarray(lateralization_schedule, dtype=np.float64).ravel(), -1.0, 1.0
    )
    if lateralization_schedule.size != n_epochs:
        lateralization_schedule = np.interp(
            np.linspace(0, 1, n_epochs),
            np.linspace(0, 1, lateralization_schedule.size),
            lateralization_schedule,
        )

    signal = np.zeros((len(channels), n), dtype=np.float64)

    for ep in range(n_epochs):
        s = int(round(ep * epoch_s * sfreq))
        e = min(n, int(round((ep + 1) * epoch_s * sfreq)))
        m = e - s
        if m <= 8:
            continue

        focus = float(focus_schedule[ep])
        li = float(lateralization_schedule[ep])

        # theta rises when retrieval is effortful; gamma rises with vividness
        theta_amp = 9.0 * (1.15 - 0.75 * focus)
        gamma_amp = 3.2 + 4.8 * focus
        alpha_amp = 16.0 * (1.0 - 0.45 * focus)

        for ci, ch in enumerate(channels):
            gain = _CHANNEL_GAIN.get(ch, 0.7)
            background = 14.0 * gain * _pink_noise(rng, m, sfreq)

            # hemifield drive: right occipital channels brighten with +lateralisation
            side = {"O1": -1.0, "P3": -1.0, "O2": 1.0, "P4": 1.0}.get(ch, 0.0)
            hemi = 1.0 + 0.30 * li * side if side else 1.0

            alpha = _burst(rng, m, sfreq, rng.uniform(9.2, 10.8), alpha_amp * gain, am_hz=0.4)
            theta = _burst(rng, m, sfreq, rng.uniform(5.4, 7.2), theta_amp * gain, am_hz=0.5)
            gamma = _burst(rng, m, sfreq, rng.uniform(34.0, 42.0), gamma_amp * gain * hemi, 1.1)
            beta = _burst(rng, m, sfreq, rng.uniform(16.0, 24.0), 3.0 * gain, 0.9)

            signal[ci, s:e] = background + alpha + theta + gamma + beta

    # a few blink/EMG-ish transients so the PSD code is exercised on ugly data
    for _ in range(max(1, int(duration_s * 0.8))):
        idx = rng.integers(0, max(1, n - 64))
        width = int(rng.integers(12, 48))
        ch = int(rng.integers(0, len(channels)))
        signal[ch, idx : idx + width] += rng.normal(0, 30) * np.hanning(width)

    return EEGRecording(
        signal=signal,
        sfreq=sfreq,
        channels=channels,
        synthetic=True,
        meta={
            "seed": seed,
            "epoch_s": epoch_s,
            "focus_schedule": [round(float(v), 4) for v in focus_schedule],
            "lateralization_schedule": [round(float(v), 4) for v in lateralization_schedule],
            "generator": "mns.eeg.synthetic/v1",
        },
    )
