"""Spectral pipeline: synthetic generation -> Welch -> calibrated salience.

These tests pin the properties the demo actually relies on, not an exact numeric
output: vivid epochs must rank above effortful ones, every scalar must stay inside
its documented range, and the three calibration modes must remain distinct.
"""

from __future__ import annotations

import numpy as np

from mns.eeg.psd import (
    BANDS,
    POSTERIOR_CHANNELS,
    SalienceCalibrator,
    analyze_epochs,
    band_power,
    compute_psd,
    spectral_entropy,
)
from mns.eeg.synthetic import generate_eeg


def _recording(focus: float, seed: int = 3):
    n_epochs = 6
    return generate_eeg(
        duration_s=n_epochs * 0.75,
        epoch_s=0.75,
        focus_schedule=np.full(n_epochs, focus),
        lateralization_schedule=np.zeros(n_epochs),
        seed=seed,
    )


def test_psd_shapes_and_band_limits():
    rec = _recording(0.5)
    result = compute_psd(rec.signal, rec.sfreq, rec.channels)
    assert result.psd.shape == (len(rec.channels), len(result.freqs))
    assert result.freqs[0] >= 1.0 and result.freqs[-1] <= 50.0
    for band in BANDS.values():
        power = band_power(result.freqs, result.psd, band)
        assert power.shape == (len(rec.channels),)
        assert np.all(power > 0)


def test_spectral_entropy_tone_is_lower_than_noise():
    sfreq = 250.0
    n = 500
    rng = np.random.default_rng(0)
    t = np.arange(n) / sfreq
    tone = np.sin(2 * np.pi * 10.0 * t)[None, :]
    noise = rng.normal(size=(1, n))
    tone_psd = compute_psd(tone, sfreq, ("Oz",))
    noise_psd = compute_psd(noise, sfreq, ("Oz",))
    band = (4.0, 45.0)
    assert spectral_entropy(tone_psd.freqs, tone_psd.psd, band)[0] < spectral_entropy(
        noise_psd.freqs, noise_psd.psd, band
    )[0]


def test_vivid_epochs_rank_above_effortful_epochs():
    low, _, low_table = analyze_epochs(_recording(0.05), epoch_s=0.75)
    high, _, high_table = analyze_epochs(_recording(0.95), epoch_s=0.75)

    assert np.mean([e.salience for e in high]) > np.mean([e.salience for e in low]) + 0.2
    assert high_table["gamma_theta_mean"] > low_table["gamma_theta_mean"]
    assert all(0.1 <= e.salience <= 1.0 for e in low + high)


def test_epoch_features_are_documented_and_finite():
    epochs, calibrator, table = analyze_epochs(_recording(0.6), epoch_s=0.75)
    assert calibrator.mode == "population"
    assert table["calibration_mode"] == "population"
    for ep in epochs:
        assert ep.channels_used and set(ep.channels_used) <= set(POSTERIOR_CHANNELS)
        assert 0.0 <= ep.certainty <= 1.0
        assert 0.0 <= ep.focus_index <= 1.0
        assert -1.0 <= ep.lateralization <= 1.0
        assert ep.theta > 0 and ep.gamma > 0 and ep.gamma_theta > 0
        assert np.isfinite([ep.theta, ep.alpha, ep.beta, ep.gamma, ep.salience]).all()


def test_calibration_modes_are_distinct_and_bounded():
    rec = _recording(0.5)
    population, cal_p, _ = analyze_epochs(rec, calibration="population")
    session, cal_s, _ = analyze_epochs(rec, calibration="session")
    assert cal_p.mode == "population"
    assert cal_s.mode == "session"
    # session calibration centres the recording on itself; the absolute scale differs.
    assert not np.allclose(
        [e.salience for e in population], [e.salience for e in session], atol=1e-3
    )
    assert all(0.1 <= e.salience <= 1.0 for e in population + session)


def test_calibrator_is_monotonic():
    cal = SalienceCalibrator(center=0.0, scale=0.2)
    values = cal.transform(np.array([-1.0, -0.3, 0.0, 0.3, 1.0]))
    assert np.all(np.diff(values) > 0)
    assert values.min() >= 0.1 and values.max() <= 1.0
    assert 0.1 <= cal.transform_one(10.0) <= 1.0
    assert 0.1 <= cal.transform_one(-10.0) <= 1.0
