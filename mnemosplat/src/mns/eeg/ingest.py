"""Recorded-EEG ingest: CSV/TSV tables and NumPy arrays -> :class:`EEGRecording`.

The public demo accepts a recording but never lets it touch the image. This module
only turns a user file into the same ``(channels, samples)`` contract the synthetic
generator produces, so the rest of the pipeline cannot tell the difference beyond
``recording.synthetic == False``.

Supported inputs
----------------
CSV / TSV / TXT
    A header row is recommended. Channel columns are named (``O1``, ``EEG O2``,
    ``Pz``); an optional time column (``time``, ``timestamp``, ``seconds``, ...) is
    detected and removed. If the time column uses seconds, milliseconds or
    microseconds, the sampling rate is inferred from its median step.
NPY
    A 2-D numeric array in either ``(channels, samples)`` or ``(samples, channels)``
    orientation. ``.npy`` carries no sampling rate or channel names, so the caller
    must supply ``sfreq``; generic channel names are used.

EDF/BDF/EEGLAB/FIF are deliberately rejected with an export hint rather than pulling
a heavy reader into the public deployment.
"""

from __future__ import annotations

import csv
import io
import math
import re
from pathlib import Path

import numpy as np

from .synthetic import EEGRecording

#: The gamma band tops out at 45 Hz, so a meaningful recording needs at least 100 Hz.
MIN_SFREQ = 100.0
MAX_SFREQ = 2_000.0
MAX_CHANNELS = 32
MAX_SAMPLES_PER_CHANNEL = 600_000
MAX_TOTAL_SAMPLES = 3_000_000

_TIME_NAMES = {
    "time",
    "t",
    "timestamp",
    "sec",
    "secs",
    "second",
    "seconds",
    "time_s",
    "time_sec",
    "time_ms",
    "time_seconds",
    "time_milliseconds",
}
_INDEX_NAMES = {
    "sample",
    "samples",
    "sample_index",
    "sampleindex",
    "index",
    "idx",
    "n",
}
_UNSUPPORTED = {
    ".edf": "EDF",
    ".bdf": "BDF",
    ".set": "EEGLAB .set",
    ".fif": "FIF",
    ".vhdr": "BrainVision",
    ".eeg": "BrainVision",
}

#: Canonical spellings for the posterior montage used by the salience extractor.
_POSTERIOR_CASE = {
    "O1": "O1",
    "O2": "O2",
    "OZ": "Oz",
    "P3": "P3",
    "P4": "P4",
    "PZ": "Pz",
}
_KNOWN_CHANNELS = {
    "FP1",
    "FP2",
    "F7",
    "F3",
    "FZ",
    "F4",
    "F8",
    "FC1",
    "FC2",
    "FC5",
    "FC6",
    "T3",
    "T4",
    "T5",
    "T6",
    "C3",
    "CZ",
    "C4",
    "CP3",
    "CP4",
    "CPZ",
    "P3",
    "P4",
    "PZ",
    "P7",
    "P8",
    "PO3",
    "PO4",
    "POZ",
    "O1",
    "O2",
    "OZ",
    "A1",
    "A2",
    "M1",
    "M2",
    "FT9",
    "FT10",
    "TP9",
    "TP10",
    "AF3",
    "AF4",
    "AF7",
    "AF8",
}


class EEGFormatError(ValueError):
    """A user-facing upload problem (400, not a server crash)."""


def safe_filename(filename: str | None, fallback: str = "recording") -> str:
    """Basename without control characters, safe for display and manifests."""
    name = Path(str(filename or "")).name
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    return name[:96] or fallback


def parse_eeg_upload(
    raw: bytes,
    filename: str | None = None,
    sfreq: float | None = None,
) -> EEGRecording:
    """Parse an uploaded file into a recorded :class:`EEGRecording`.

    Raises
    ------
    EEGFormatError
        For an empty, unsupported, malformed, non-finite or excessively large file.
    """
    if not raw:
        raise EEGFormatError("the uploaded file is empty")

    name = safe_filename(filename)
    suffix = Path(name).suffix.lower()

    if suffix in _UNSUPPORTED:
        raise EEGFormatError(
            f"{_UNSUPPORTED[suffix]} is not supported yet. Export the recording as a "
            "CSV with one channel per column (optionally a time column) or as a .npy array."
        )

    if suffix == ".npy" or raw[:6] == b"\x93NUMPY":
        rec = _parse_npy(raw, name, sfreq)
    elif suffix in {".csv", ".tsv", ".txt", ""}:
        rec = _parse_table(raw, name, suffix, sfreq)
    else:
        raise EEGFormatError(
            "unsupported file type. Upload a .csv, .tsv, .txt or .npy EEG recording."
        )

    _validate_recording(rec)
    return rec


# --------------------------------------------------------------------------- #
# table input
# --------------------------------------------------------------------------- #
def _parse_table(
    raw: bytes, filename: str, suffix: str, sfreq: float | None
) -> EEGRecording:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise EEGFormatError("the CSV must be UTF-8 text (not a binary or UTF-16 file)") from exc

    if not text.strip():
        raise EEGFormatError("the CSV has no rows")

    sample = "\n".join(
        line
        for line in text.splitlines()[:24]
        if line.strip() and not line.lstrip().startswith("#")
    )
    delimiter = "\t" if suffix == ".tsv" else ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        delimiter = dialect.delimiter
    except csv.Error:
        if suffix == ".tsv":
            delimiter = "\t"
        elif suffix == ".txt":
            delimiter = "," if "," in sample else "\t"

    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows: list[list[str]] = []
    try:
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                continue
            if row[0].lstrip().startswith("#"):
                continue
            rows.append([cell.strip() for cell in row])
    except csv.Error as exc:
        raise EEGFormatError(f"could not parse the CSV: {exc}") from exc

    if not rows:
        raise EEGFormatError("the CSV has no data rows")
    n_cols = len(rows[0])
    if n_cols < 1:
        raise EEGFormatError("the CSV has no columns")

    has_header = any(not _is_float(cell) for cell in rows[0])
    if has_header:
        names = [cell or f"ch{i + 1}" for i, cell in enumerate(rows[0])]
        data_rows = rows[1:]
    else:
        names = [f"ch{i + 1}" for i in range(n_cols)]
        data_rows = rows

    if not data_rows:
        raise EEGFormatError("the CSV has a header but no samples")

    values = np.empty((len(data_rows), n_cols), dtype=np.float64)
    for r, row in enumerate(data_rows):
        if len(row) != n_cols:
            raise EEGFormatError(
                f"row {r + 2} has {len(row)} columns; expected {n_cols} (same as the header)"
            )
        for c, cell in enumerate(row):
            try:
                value = float(cell)
            except ValueError as exc:
                raise EEGFormatError(
                    f"non-numeric value {cell!r} at row {r + 2}, column {c + 1}"
                ) from exc
            if not math.isfinite(value):
                raise EEGFormatError(
                    f"non-finite (NaN or infinite) value at row {r + 2}, column {c + 1}"
                )
            values[r, c] = value

    time_idx: int | None = None
    time_kind: str | None = None
    if has_header:
        for i, name in enumerate(names):
            kind = _time_kind(name)
            if kind is not None:
                time_idx = i
                time_kind = kind
                break
    elif n_cols >= 2 and _looks_like_uniform_ramp(values[:, 0]):
        time_idx = 0
        time_kind = "time"

    sfreq_source = "form"
    sf = _check_sfreq(sfreq, required=(time_idx is None or time_kind == "index"))
    if time_idx is not None:
        times = values[:, time_idx]
        _validate_time(times)
        if time_kind == "time":
            inferred = _infer_sfreq(times, sf)
            if inferred is None:
                raise EEGFormatError(
                    "could not infer a sampling rate between 100 and 2000 Hz from the "
                    "time column; use seconds, milliseconds or microseconds, or remove "
                    "the time column and enter the recording's rate"
                )
            sf = inferred
            sfreq_source = "time_column"
    if sf is None:
        raise EEGFormatError("sampling rate is required for a file without a time column")
    sf = _check_sfreq(sf, required=True)

    if time_idx is not None:
        keep = [i for i in range(n_cols) if i != time_idx]
        values = values[:, keep]
        names = [names[i] for i in keep]

    if not names:
        raise EEGFormatError("no channel columns remain after removing the time column")

    canonical = [_canonical_channel(name, i) for i, name in enumerate(names)]
    canonical = _dedupe_names(canonical)
    signal = values.T

    return _make_recording(
        signal=signal,
        sfreq=sf,
        channels=canonical,
        filename=filename,
        fmt=f"table:{delimiter}",
        sfreq_source=sfreq_source,
        extra={"time_column": bool(time_idx is not None)},
    )


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _time_kind(name: str) -> str | None:
    key = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    if key in _TIME_NAMES:
        return "time"
    if key in _INDEX_NAMES:
        return "index"
    # A few exporters write "Time (s)" or "seconds_s".
    if key.startswith("time") or key.endswith("_seconds"):
        return "time"
    if key.startswith("sample") or key == "index":
        return "index"
    return None


def _looks_like_uniform_ramp(column: np.ndarray) -> bool:
    """Headerless files: a perfectly uniform increasing first column is time."""
    if column.size < 4:
        return False
    diff = np.diff(column)
    if not np.all(np.isfinite(diff)) or not np.all(diff > 0):
        return False
    step = float(np.median(diff))
    return bool(step > 0 and np.allclose(diff, step, rtol=1e-4, atol=1e-12))


def _validate_time(times: np.ndarray) -> None:
    if times.size < 2:
        raise EEGFormatError("the time column must contain at least two samples")
    diff = np.diff(times)
    if not np.all(np.isfinite(diff)) or not np.all(diff > 0):
        raise EEGFormatError("the time column must be strictly increasing")


def _infer_sfreq(times: np.ndarray, provided: float | None) -> float | None:
    """Infer Hz from a time column in seconds, milliseconds or microseconds."""
    diff = np.diff(times)
    diff = diff[np.isfinite(diff) & (diff > 0)]
    if diff.size < 3:
        return None
    dt = float(np.median(diff))
    if not (1e-12 < dt < 1e12):
        return None

    candidates = [
        (1.0 / dt, "seconds"),
        (1_000.0 / dt, "milliseconds"),
        (1_000_000.0 / dt, "microseconds"),
    ]
    valid = [(hz, unit) for hz, unit in candidates if MIN_SFREQ <= hz <= MAX_SFREQ]
    if not valid:
        return None
    if provided is not None:
        # A real time column is more reliable than a form default. Use the supplied
        # value only to break ties between unit interpretations.
        valid.sort(key=lambda item: abs(item[0] - provided))
    return round(float(valid[0][0]), 3)


def _canonical_channel(name: str, index: int) -> str:
    raw = "".join(ch for ch in str(name) if ch.isprintable()).strip()
    if not raw:
        return f"ch{index + 1}"

    cleaned = re.sub(r"^(EEG|CH|CHANNEL)[\s._-]*", "", raw, flags=re.IGNORECASE)
    tokens = re.findall(r"[A-Za-z]+[0-9]*", cleaned.upper())
    for token in tokens:
        if token in _KNOWN_CHANNELS:
            return _POSTERIOR_CASE.get(token, token.upper())
    fallback = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_")
    return (fallback[:32] or f"ch{index + 1}")


def _dedupe_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in names:
        count = seen.get(name, 0) + 1
        seen[name] = count
        out.append(name if count == 1 else f"{name}_{count}")
    return out


# --------------------------------------------------------------------------- #
# NumPy input
# --------------------------------------------------------------------------- #
def _parse_npy(raw: bytes, filename: str, sfreq: float | None) -> EEGRecording:
    try:
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - surface a user-facing parser message
        raise EEGFormatError(f"could not read the .npy file: {exc}") from exc
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        raise EEGFormatError("the .npy array must be two-dimensional")
    if not np.issubdtype(arr.dtype, np.number):
        raise EEGFormatError("the .npy array must contain numeric samples")

    values = np.asarray(arr, dtype=np.float64)
    rows, cols = values.shape
    if rows < 1 or cols < 1:
        raise EEGFormatError("the .npy array is empty")

    if rows <= MAX_CHANNELS and cols > rows:
        signal = values
        orientation = "channels,samples"
    elif cols <= MAX_CHANNELS and rows > cols:
        signal = values.T
        orientation = "samples,channels"
    elif rows <= MAX_CHANNELS and cols <= MAX_CHANNELS:
        signal = values
        orientation = "channels,samples"
    else:
        raise EEGFormatError(
            "ambiguous .npy shape. Expected (channels, samples) or (samples, channels) "
            f"with at most {MAX_CHANNELS} channels; got {values.shape}."
        )

    sf = _check_sfreq(sfreq, required=True)
    assert sf is not None
    channels = [f"ch{i + 1}" for i in range(signal.shape[0])]
    return _make_recording(
        signal=signal,
        sfreq=sf,
        channels=channels,
        filename=filename,
        fmt="npy",
        sfreq_source="form",
        extra={"orientation": orientation},
    )


# --------------------------------------------------------------------------- #
# shared validation
# --------------------------------------------------------------------------- #
def _check_sfreq(value: float | None, required: bool) -> float | None:
    if value is None:
        if required:
            raise EEGFormatError(
                "sampling rate is required; enter the recording's sampling rate in Hz "
                "(for example 250)"
            )
        return None
    try:
        sf = float(value)
    except (TypeError, ValueError) as exc:
        raise EEGFormatError("sampling rate must be a number") from exc
    if not math.isfinite(sf) or not (MIN_SFREQ <= sf <= MAX_SFREQ):
        raise EEGFormatError(
            f"sampling rate must be between {MIN_SFREQ:.0f} and {MAX_SFREQ:.0f} Hz "
            "(the gamma band needs a Nyquist frequency above 45 Hz)"
        )
    return sf


def _make_recording(
    signal: np.ndarray,
    sfreq: float,
    channels: list[str],
    filename: str,
    fmt: str,
    sfreq_source: str,
    extra: dict | None = None,
) -> EEGRecording:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2:
        raise EEGFormatError("the recording must be two-dimensional")
    meta = {
        "source": "upload",
        "filename": filename,
        "format": fmt,
        "sfreq_source": sfreq_source,
        "channels": list(channels),
        "n_channels": int(signal.shape[0]),
        "n_samples": int(signal.shape[1]),
        "duration_s": round(float(signal.shape[1]) / sfreq, 3),
        "units": "input units (band ratios are scale-invariant)",
    }
    if extra:
        meta.update(extra)
    return EEGRecording(
        signal=signal,
        sfreq=float(sfreq),
        channels=tuple(channels),
        synthetic=False,
        meta=meta,
    )


def _validate_recording(recording: EEGRecording) -> None:
    signal = np.asarray(recording.signal, dtype=np.float64)
    n_channels, n_samples = signal.shape
    if n_channels < 1:
        raise EEGFormatError("no EEG channels found")
    if n_channels > MAX_CHANNELS:
        raise EEGFormatError(f"too many channels ({n_channels}); the limit is {MAX_CHANNELS}")
    if n_samples < 8:
        raise EEGFormatError("too few samples; the recording is effectively empty")
    if n_samples > MAX_SAMPLES_PER_CHANNEL:
        raise EEGFormatError(
            f"too many samples per channel ({n_samples:,}); the limit is "
            f"{MAX_SAMPLES_PER_CHANNEL:,}. Export a shorter segment."
        )
    if n_channels * n_samples > MAX_TOTAL_SAMPLES:
        raise EEGFormatError(
            "the recording is too large after parsing "
            f"({n_channels} x {n_samples:,} = {n_channels * n_samples:,} values); "
            f"the limit is {MAX_TOTAL_SAMPLES:,}. Export a shorter segment."
        )
    if not np.isfinite(signal).all():
        raise EEGFormatError("the recording contains NaN or infinite values")
    if float(np.max(np.abs(signal))) > 1e12:
        raise EEGFormatError(
            "the recording contains implausibly large values; check that the time "
            "column was not parsed as a channel"
        )

    std = np.std(signal, axis=1)
    if not np.any(std > 1e-12):
        raise EEGFormatError("all channels are flat; there is no EEG signal to analyse")

    # Ratios are scale-invariant, but removing each channel's DC offset keeps the
    # Welch estimate numerically well behaved and matches common EEG exports.
    centered = signal - np.median(signal, axis=1, keepdims=True)
    recording.signal = centered.astype(np.float64, copy=False)
    recording.meta["n_samples"] = int(n_samples)
    recording.meta["n_channels"] = int(n_channels)
    recording.meta["duration_s"] = round(float(n_samples) / recording.sfreq, 3)


__all__ = [
    "EEGFormatError",
    "MAX_CHANNELS",
    "MAX_SAMPLES_PER_CHANNEL",
    "MAX_SFREQ",
    "MAX_TOTAL_SAMPLES",
    "MIN_SFREQ",
    "parse_eeg_upload",
    "safe_filename",
]
