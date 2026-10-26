"""EEG ingest adapters."""

from .ingest import (
    MAX_CHANNELS,
    MAX_SAMPLES_PER_CHANNEL,
    MAX_SFREQ,
    MIN_SFREQ,
    EEGFormatError,
    parse_eeg_upload,
    safe_filename,
)
from .psd import (
    BANDS,
    POSTERIOR_CHANNELS,
    PSDResult,
    SalienceCalibrator,
    aggregate_epochs,
    analyze_epochs,
    band_power,
    compute_psd,
    spectral_entropy,
)
from .synthetic import DEFAULT_CHANNELS, EEGRecording, generate_eeg

__all__ = [
    "BANDS",
    "DEFAULT_CHANNELS",
    "EEGFormatError",
    "EEGRecording",
    "MAX_CHANNELS",
    "MAX_SAMPLES_PER_CHANNEL",
    "MAX_SFREQ",
    "MIN_SFREQ",
    "POSTERIOR_CHANNELS",
    "PSDResult",
    "SalienceCalibrator",
    "aggregate_epochs",
    "analyze_epochs",
    "band_power",
    "compute_psd",
    "generate_eeg",
    "parse_eeg_upload",
    "safe_filename",
    "spectral_entropy",
]
