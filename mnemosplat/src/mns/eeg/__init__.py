"""EEG ingest adapters."""

from .psd import (
    BANDS,
    POSTERIOR_CHANNELS,
    PSDResult,
    SalienceCalibrator,
    analyze_epochs,
    band_power,
    compute_psd,
    spectral_entropy,
)
from .synthetic import DEFAULT_CHANNELS, EEGRecording, generate_eeg

__all__ = [
    "BANDS",
    "DEFAULT_CHANNELS",
    "EEGRecording",
    "POSTERIOR_CHANNELS",
    "PSDResult",
    "SalienceCalibrator",
    "analyze_epochs",
    "band_power",
    "compute_psd",
    "generate_eeg",
    "spectral_entropy",
]
