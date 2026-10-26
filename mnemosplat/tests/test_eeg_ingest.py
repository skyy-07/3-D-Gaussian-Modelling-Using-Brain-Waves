"""Recorded-EEG ingest: formats, sampling-rate inference and beat aggregation."""

from __future__ import annotations

import io

import numpy as np
import pytest

from mns.eeg.ingest import (
    MAX_CHANNELS,
    EEGFormatError,
    parse_eeg_upload,
)
from mns.eeg.psd import aggregate_epochs, analyze_epochs
from mns.schemas import EpochFeatures


def _signal(seconds: float = 4.0, sfreq: float = 250.0, channels: int = 4) -> np.ndarray:
    t = np.arange(int(round(seconds * sfreq))) / sfreq
    out = []
    for i in range(channels):
        out.append(
            8.0 * np.sin(2 * np.pi * (9.5 + i) * t)
            + 4.0 * np.sin(2 * np.pi * (35.0 + i) * t)
            + 0.5 * np.sin(2 * np.pi * 6.0 * t)
        )
    return np.stack(out)


def _csv(
    signal: np.ndarray,
    sfreq: float,
    channels: tuple[str, ...] | None = None,
    time_unit: str | None = "s",
) -> bytes:
    n_channels, n_samples = signal.shape
    names = list(channels or [f"ch{i + 1}" for i in range(n_channels)])
    t = np.arange(n_samples) / sfreq
    if time_unit == "ms":
        t = t * 1000.0
    elif time_unit == "us":
        t = t * 1_000_000.0
    header = (["time"] if time_unit else []) + names
    rows = [",".join(header)]
    for i in range(n_samples):
        values = [f"{t[i]:.8f}"] if time_unit else []
        values += [f"{signal[c, i]:.8f}" for c in range(n_channels)]
        rows.append(",".join(values))
    return "\n".join(rows).encode("utf-8")


def test_csv_with_time_column_infers_sampling_rate():
    signal = _signal()
    raw = _csv(signal, 250.0, ("O1", "O2", "Oz", "Pz"), time_unit="s")
    recording = parse_eeg_upload(raw, "memory.csv", sfreq=1000.0)

    assert recording.synthetic is False
    assert recording.channels == ("O1", "O2", "Oz", "Pz")
    assert recording.sfreq == pytest.approx(250.0, rel=1e-4)
    assert recording.signal.shape == signal.shape
    assert recording.meta["sfreq_source"] == "time_column"
    assert recording.meta["source"] == "upload"


@pytest.mark.parametrize(("unit", "expected"), [("ms", 250.0), ("us", 250.0)])
def test_time_column_units_are_detected(unit, expected):
    raw = _csv(_signal(), 250.0, ("O1", "O2", "Oz", "Pz"), time_unit=unit)
    recording = parse_eeg_upload(raw, "memory.csv", sfreq=250.0)
    assert recording.sfreq == pytest.approx(expected, rel=1e-4)


def test_csv_without_time_column_uses_form_sampling_rate():
    raw = _csv(_signal(), 250.0, ("O1", "O2", "Oz", "Pz"), time_unit=None)
    recording = parse_eeg_upload(raw, "memory.csv", sfreq=250.0)
    assert recording.sfreq == 250.0
    assert recording.meta["sfreq_source"] == "form"
    assert recording.meta["time_column"] is False


def test_channel_names_are_canonicalised():
    raw = _csv(
        _signal(channels=3),
        250.0,
        ("EEG O1", "O2-A2", "Pz"),
        time_unit="s",
    )
    recording = parse_eeg_upload(raw, "memory.csv", sfreq=250.0)
    assert recording.channels == ("O1", "O2", "Pz")


def test_npy_accepts_both_orientations():
    signal = _signal(channels=3)
    channels_first = io.BytesIO()
    np.save(channels_first, signal)
    rec = parse_eeg_upload(channels_first.getvalue(), "memory.npy", sfreq=250.0)
    assert rec.signal.shape == signal.shape
    assert rec.meta["orientation"] == "channels,samples"

    samples_first = io.BytesIO()
    np.save(samples_first, signal.T)
    rec = parse_eeg_upload(samples_first.getvalue(), "memory.npy", sfreq=250.0)
    assert rec.signal.shape == signal.shape
    assert rec.meta["orientation"] == "samples,channels"


def test_edf_is_rejected_with_an_export_hint():
    with pytest.raises(EEGFormatError, match="not supported"):
        parse_eeg_upload(b"\x00" * 32, "memory.edf", sfreq=250.0)


def test_empty_file_is_rejected():
    with pytest.raises(EEGFormatError, match="empty"):
        parse_eeg_upload(b"", "memory.csv", sfreq=250.0)


def test_non_finite_values_are_rejected():
    signal = _signal()
    signal[0, 10] = np.nan
    raw = _csv(signal, 250.0, time_unit="s")
    with pytest.raises(EEGFormatError, match="NaN|infinite"):
        parse_eeg_upload(raw, "memory.csv", sfreq=250.0)


def test_oversized_csv_field_is_a_format_error():
    raw = b"time,O1\n0," + b"1" * 200_000 + b"\n"
    with pytest.raises(EEGFormatError, match="could not parse"):
        parse_eeg_upload(raw, "memory.csv", sfreq=250.0)


def test_flat_recording_is_rejected():
    signal = np.zeros((2, 500))
    raw = _csv(signal, 250.0, ("O1", "O2"), time_unit="s")
    with pytest.raises(EEGFormatError, match="flat"):
        parse_eeg_upload(raw, "memory.csv", sfreq=250.0)


def test_low_sampling_rate_is_rejected():
    raw = _csv(_signal(), 50.0, time_unit="s")
    with pytest.raises(EEGFormatError, match="between"):
        parse_eeg_upload(raw, "memory.csv", sfreq=50.0)


def test_time_column_outside_supported_range_does_not_fall_back_to_the_form():
    raw = _csv(_signal(seconds=1.0), 50.0, time_unit="s")
    with pytest.raises(EEGFormatError, match="could not infer"):
        parse_eeg_upload(raw, "memory.csv", sfreq=250.0)


def test_too_many_channels_are_rejected():
    signal = _signal(channels=MAX_CHANNELS + 1, seconds=2.0)
    raw = _csv(signal, 250.0, time_unit="s")
    with pytest.raises(EEGFormatError, match="too many channels"):
        parse_eeg_upload(raw, "memory.csv", sfreq=250.0)


def test_uploaded_recording_aggregates_into_four_beats():
    raw = _csv(_signal(seconds=6.0, channels=6), 250.0, time_unit="s")
    recording = parse_eeg_upload(raw, "memory.csv", sfreq=250.0)
    epochs, calibrator, table = analyze_epochs(recording, epoch_s=0.75)
    assert len(epochs) == 8
    assert calibrator.mode == "population"
    assert table["sfreq"] == pytest.approx(250.0, rel=1e-6)

    beats = aggregate_epochs(epochs, 4, epoch_s=0.75)
    assert len(beats) == 4
    assert [beat.index for beat in beats] == [0, 1, 2, 3]
    assert [beat.t0 for beat in beats] == [0.0, 0.75, 1.5, 2.25]
    assert [beat.t1 for beat in beats] == [0.75, 1.5, 2.25, 3.0]
    assert all(0.1 <= beat.salience <= 1.0 for beat in beats)
    assert all(0.0 <= beat.certainty <= 1.0 for beat in beats)
    assert all(-1.0 <= beat.lateralization <= 1.0 for beat in beats)


def test_short_recording_interpolates_to_four_beats():
    epochs = [
        EpochFeatures(
            index=i,
            t0=i * 0.75,
            t1=(i + 1) * 0.75,
            theta=10.0 + i,
            alpha=20.0 + i,
            beta=15.0,
            gamma=8.0 + i,
            gamma_theta=0.8,
            lateralization=0.1 * i,
            certainty=0.5,
            salience=0.2 + 0.1 * i,
            focus_index=0.3,
            channels_used=["O1", "O2"],
        )
        for i in range(3)
    ]
    beats = aggregate_epochs(epochs, 4, epoch_s=0.75)
    assert len(beats) == 4
    assert beats[0].salience == pytest.approx(0.2)
    assert beats[-1].salience == pytest.approx(0.4)
    assert beats[0].t0 == 0.0 and beats[-1].t1 == 3.0
