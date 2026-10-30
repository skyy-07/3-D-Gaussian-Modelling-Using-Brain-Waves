"""CUDA worker for the experimental BrainVis reconstruction branch.

Run this on a GPU machine, not on the Render CPU web service. It owns the BrainVis
checkpoint, Stable Diffusion 1.5 and the CUDA context. The web app only forwards
EEG windows and stores PNGs.
"""

from __future__ import annotations

import base64
import io
import json
import os
import secrets
import threading

import numpy as np
from engine import CHANNELS, WINDOW_SAMPLES, BrainVisEngine
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from PIL import Image
from scipy import signal

MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_WINDOWS = 8
TOKEN = os.environ.get("MNS_BRAINVIS_WORKER_TOKEN", "")
NORMALIZE = os.environ.get("MNS_BRAINVIS_NORMALIZE", "zscore").strip().lower()
ENGINE = BrainVisEngine(
    steps=int(os.environ.get("MNS_BRAINVIS_STEPS", "50")),
    stage2_steps=int(os.environ.get("MNS_BRAINVIS_STAGE2_STEPS", "100")),
    num_samples=int(os.environ.get("MNS_BRAINVIS_NUM_SAMPLES", "1")),
    seed=int(os.environ.get("MNS_BRAINVIS_SEED", "3407")),
)
_INFER_LOCK = threading.Lock()

app = FastAPI(title="MnemoSplat BrainVis worker", version="0.1")


def _auth(authorization: str | None) -> None:
    if not TOKEN:
        return
    supplied = ""
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not secrets.compare_digest(supplied, TOKEN):
        raise HTTPException(401, "invalid or missing worker token")


def _normalize(window: np.ndarray) -> np.ndarray:
    """Approximate the 5–95 standardisation used by the EEG-Image release.

    The exact dataset transform is not shipped with the checkpoint. Z-scoring each
    channel over the 440-sample trial is the closest defensible default; operators can
    set ``MNS_BRAINVIS_NORMALIZE=none`` when they already have the benchmark tensor.
    """
    x = np.asarray(window, dtype=np.float32)
    if NORMALIZE == "none":
        return x
    if NORMALIZE == "minmax":
        lo, hi = np.percentile(x, [5, 95], axis=-1, keepdims=True)
        return (x - lo) / np.maximum(hi - lo, 1e-6)
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-6)


def _resample(window: np.ndarray) -> np.ndarray:
    x = np.asarray(window, dtype=np.float32)
    if x.shape[-1] == WINDOW_SAMPLES:
        return x
    if x.shape[-1] < 8:
        raise HTTPException(422, "each EEG window needs at least 8 samples")
    # The caller supplies 0.44 s windows. Resample the time axis to the 440-sample
    # BrainVis contract; this is not a substitute for correct acquisition filtering.
    return signal.resample(x, WINDOW_SAMPLES, axis=-1).astype(np.float32)


@app.get("/health")
def health() -> dict:
    status = ENGINE.status()
    status["normalize"] = NORMALIZE
    status["window_samples"] = WINDOW_SAMPLES
    status["channels"] = CHANNELS
    return status


@app.post("/reconstruct")
async def reconstruct(
    file: UploadFile = File(...),
    sfreq: float = Form(1000.0),
    channels: str = Form("[]"),
    prompt: str = Form(""),
    seed: int | None = Form(None),
    normalize: bool = Form(True),
    authorization: str | None = Header(default=None),
) -> dict:
    _auth(authorization)
    status = ENGINE.status()
    if not status["ready"]:
        raise HTTPException(503, status["reason"])

    raw = await file.read()
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"EEG upload must be between 1 byte and {MAX_UPLOAD_BYTES} bytes")
    try:
        payload = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"could not read the .npy payload: {exc}") from exc

    arr = np.asarray(payload)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise HTTPException(422, "expected an array shaped (windows, channels, samples)")
    if arr.shape[0] < 1 or arr.shape[0] > MAX_WINDOWS:
        raise HTTPException(422, f"expected 1–{MAX_WINDOWS} windows, got {arr.shape[0]}")
    if arr.shape[1] != CHANNELS:
        raise HTTPException(
            422,
            f"BrainVis requires {CHANNELS} channels; got {arr.shape[1]}. "
            "The CPU app should reject this before calling the worker.",
        )
    if not np.isfinite(arr).all():
        raise HTTPException(422, "EEG array contains NaN or infinite values")
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise HTTPException(422, "sfreq must be a positive number")

    try:
        names = json.loads(channels)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, "channels must be a JSON array of names") from exc
    if not isinstance(names, list) or len(names) != CHANNELS:
        raise HTTPException(422, f"channels must contain {CHANNELS} names in training order")

    frames = []
    with _INFER_LOCK:
        for index, window in enumerate(arr):
            prepared = _normalize(_resample(window)) if normalize else _resample(window)
            try:
                generated = ENGINE.reconstruct(
                    prepared,
                    prompt=prompt.strip() or None,
                    seed=seed,
                )
            except Exception as exc:  # noqa: BLE001 - return a useful worker error
                raise HTTPException(500, f"BrainVis inference failed: {exc}") from exc
            # The app displays one frame per window. If num_samples > 1, return the
            # first variant and retain the count in metadata.
            item = generated[0]
            with Image.open(io.BytesIO(item.png)) as image:
                width, height = image.size
            frames.append(
                {
                    "index": index,
                    "png_b64": base64.b64encode(item.png).decode("ascii"),
                    "predicted_class": item.predicted_class,
                    "label": item.label,
                    "prompt": item.prompt,
                    "width": width,
                    "height": height,
                }
            )

    return {
        "frames": frames,
        "meta": {
            "engine": "brainvis",
            "kind": "experimental_static_reconstruction",
            "window_s": WINDOW_SAMPLES / float(sfreq),
            "window_samples": WINDOW_SAMPLES,
            "channels": CHANNELS,
            "sfreq": float(sfreq),
            "normalize": NORMALIZE if normalize else "none",
            "num_samples": ENGINE.num_samples,
            "steps": ENGINE.steps,
            "stage2_steps": ENGINE.stage2_steps,
            "checkpoint": str(ENGINE.checkpoint),
        },
    }


__all__ = ["app"]
