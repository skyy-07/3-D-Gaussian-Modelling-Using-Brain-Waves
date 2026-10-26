"""Bridge to `RomGai/BrainVis` (ICASSP 2025) and THINGS-EEG2 style recordings.

BrainVis is the backbone for the *learned* branch of MnemoSplat. Its contribution to
our pipeline is narrow and deliberate:

    raw posterior EEG -> [BrainVis freq encoder + time encoder] -> CLIP-aligned EEG
    embedding -> small MLP -> per-entity salience weights

We do **not** let that embedding synthesise pixels (see docs/SCIENCE.md, Failure Mode
1). It only ranks entities, which is a low-dimensional, high-signal target that
survives EEG noise far better than RGB regression.

This module ships the data plumbing so the swap is a config change, not a rewrite:

* :func:`load_recording` reads raw arrays / .npy / .pth (incl. BrainVis block dumps).
* :class:`SalienceEncoder` is the interface any learned encoder implements.
* :func:`resolve_brainvis` locates a local clone and reports its training artefacts.

The default :class:`SpectralSalienceEncoder` needs no weights, no GPU and no
checkpoint, so the hackathon demo is never blocked on the BrainVis authors' email
checkpoint release.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .psd import SalienceCalibrator, analyze_epochs
from .synthetic import DEFAULT_CHANNELS, EEGRecording

#: Files BrainVis needs before its encoders can be trained (README "Preparation").
BRAINVIS_ARTEFACTS = (
    "data/EEG/block_splits_by_image_all.pth",
    "data/EEG/eeg_5_95_std.pth",
    "pretrained_model/v1-5-pruned-emaonly.ckpt",
)


def resolve_brainvis(root: str | os.PathLike | None = None) -> dict:
    """Report whether a BrainVis checkout and its artefacts are available."""
    if root is None:
        root = os.environ.get("BRAINVIS_ROOT", "")
    base = Path(root) if root else None
    present = bool(base and base.exists())
    artefacts = {}
    if present:
        for rel in BRAINVIS_ARTEFACTS:
            artefacts[rel] = (base / rel).exists()
    return {
        "root": str(base) if base else "",
        "present": present,
        "artefacts": artefacts,
        "ready": present and all(artefacts.values()),
        "note": (
            "BrainVis checkpoints are released on request by the authors; the spectral "
            "salience encoder is the zero-dependency fallback."
        ),
    }


def _as_channels_first(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    # heuristic: EEG has many more samples than channels
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return arr


def load_recording(
    path: str | os.PathLike,
    sfreq: float = 250.0,
    channels: tuple[str, ...] | None = None,
    key: str | None = None,
    trial: int = 0,
) -> EEGRecording:
    """Load a recording from .npy/.npz/.pth, or a BrainVis block dump.

    BrainVis stores ``dataset[split][i]['eeg']`` arrays per image block. Pass
    ``key='dataset'`` / ``trial=i`` to pull a single block out; the array is then
    treated as (channels, samples).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix == ".npy":
        arr = np.load(p)
    elif p.suffix == ".npz":
        z = np.load(p)
        arr = z[key or next(iter(z.files))]
    elif p.suffix in {".pth", ".pt"}:
        import torch  # local import: torch is optional for the CPU demo path

        obj = torch.load(p, map_location="cpu", weights_only=False)
        if key is None:
            for candidate in ("eeg", "signal", "data", "x", "dataset"):
                if isinstance(obj, dict) and candidate in obj:
                    key = candidate
                    break
        if key is not None and isinstance(obj, dict):
            obj = obj[key]
        if key == "dataset" or (isinstance(obj, list) and obj and isinstance(obj[0], dict)):
            split = obj if isinstance(obj, list) else obj[next(iter(obj))]
            block = split[trial]
            obj = block["eeg"] if isinstance(block, dict) else block
        arr = obj.cpu().numpy() if hasattr(obj, "cpu") else np.asarray(obj)
    else:
        arr = np.loadtxt(p, delimiter=",")

    arr = _as_channels_first(arr)
    ch = channels or DEFAULT_CHANNELS
    if arr.shape[0] != len(ch):
        # keep analysis running on arbitrary montages; names are index-based
        ch = tuple(f"ch{i}" for i in range(arr.shape[0]))
    return EEGRecording(
        signal=arr, sfreq=sfreq, channels=ch, synthetic=False, meta={"path": str(p)}
    )


class SalienceEncoder(Protocol):
    """Anything that turns a recording into per-epoch salience features."""

    def encode(self, recording: EEGRecording, epoch_s: float = 0.75): ...


@dataclass
class SpectralSalienceEncoder:
    """Zero-shot encoder: Welch + robust calibration. Runs on CPU in milliseconds."""

    epoch_s: float = 0.75
    channels: tuple[str, ...] | None = None
    calibrator: SalienceCalibrator | None = None

    def encode(self, recording: EEGRecording, epoch_s: float | None = None):
        return analyze_epochs(
            recording,
            epoch_s=epoch_s or self.epoch_s,
            channels=self.channels,
            calibrator=self.calibrator,
        )


@dataclass
class BrainVisSalienceEncoder:
    """Learned encoder placeholder with a hard, explicit contract.

    To activate on a GPU box:

    1. Clone BrainVis, follow its README, train/obtain the time+freq encoders.
    2. Export an ``nn.Module`` mapping ``(B, C, T) -> (B, D)`` CLIP-aligned embeddings.
    3. Point ``encoder_path`` at a TorchScript export and set ``BRAINVIS_ROOT``.
    4. Fit a 2-layer MLP from ``D`` to the per-object salience targets in
       ``data/salience_targets.jsonl``. This checkout does not ship a ``labels``
       command; label export belongs to the study code.

    Until then this raises instead of silently pretending to be trained: an untrained
    pseudo-encoder would poison the attention map and is exactly the failure mode the
    blueprint warns about.
    """

    encoder_path: str | None = None
    epoch_s: float = 0.75

    @property
    def available(self) -> bool:
        return bool(self.encoder_path and Path(self.encoder_path).exists())

    def encode(self, recording: EEGRecording, epoch_s: float | None = None):
        if not self.available:
            raise RuntimeError(
                "BrainVisSalienceEncoder has no trained encoder at `encoder_path`. "
                "Set BRAINVIS_ROOT and export a TorchScript encoder, or use "
                "SpectralSalienceEncoder (default). See docs/BRAINVIS.md"
            )
        raise NotImplementedError(
            "TorchScript inference is wired in the GPU image; see docs/BRAINVIS.md "
            "for the export contract (B,C,T)->(B,D)."
        )
