"""Depth providers.

Priority order used by the pipeline:

1. **Oracle scene depth** when the keyframe came from the procedural renderer. It is
   the true depth of *our own* synthetic frame, so there is no leakage from any
   stimulus image.
2. **Depth-Anything-V2** (``depth-anything/Depth-Anything-V2-Small-hf``) on CPU or GPU
   when ``transformers`` + ``torch`` are installed. Used for Muse Image keyframes.
3. **Heuristic depth** as a last resort so the pipeline never hard-fails.

Scale ambiguity: DA-V2 predicts *relative inverse* depth. We map it into a plausible
metric range (0.85-6.2 m). Absolute scale only changes the apparent size of the whole
volume in the headset; per-pixel ratios - which drive splat shape - are unaffected. We
expose ``depth_source`` in the manifest so nobody mistakes it for metric truth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from PIL import Image

log = logging.getLogger("mns.depth")

Z_NEAR, Z_FAR = 0.85, 6.2


@dataclass
class DepthResult:
    depth: np.ndarray  # (h, w) metres
    source: str
    meta: dict


def heuristic_depth(image: Image.Image, width: int, height: int) -> DepthResult:
    """Cheap monocular prior: bright + high in frame = far, dark + low = near."""
    g = np.asarray(
        image.convert("L").resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32
    )
    g = g / 255.0
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    # vertical gradient dominates; brightness adds a soft sky/wall prior
    rel = 0.68 * yy + 0.32 * (1.0 - g)
    rel = np.clip(rel, 0, 1)
    return DepthResult(
        depth=(Z_NEAR + (Z_FAR - Z_NEAR) * rel**1.5).astype(np.float32),
        source="heuristic",
        meta={"note": "no learned depth model available"},
    )


def depth_anything_v2(
    image: Image.Image,
    width: int,
    height: int,
    model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
    device: str | None = None,
) -> DepthResult:
    """Run DA-V2 via transformers. Raises ImportError if the extra is missing."""
    import torch
    from transformers import pipeline

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipeline("depth-estimation", model=model_id, device=0 if device == "cuda" else -1)
    out = pipe(image.convert("RGB"))
    pred = out["predicted_depth"]
    if hasattr(pred, "detach"):
        arr = pred.squeeze().detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(pred, dtype=np.float32)

    # resize to target
    if arr.shape != (height, width):
        arr = np.asarray(
            Image.fromarray(arr).resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32
        )
    # relative inverse depth -> [0,1] "farness"
    lo, hi = np.percentile(arr, [1, 99])
    rel = 1.0 - clamp01((arr - lo) / max(1e-6, hi - lo))
    z = Z_NEAR + (Z_FAR - Z_NEAR) * rel**1.6
    return DepthResult(
        depth=z.astype(np.float32), source=f"da-v2:{model_id}", meta={"device": device}
    )


def clamp01(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0)


class DepthEstimator:
    """Facade with graceful degradation and one cached pipeline."""

    def __init__(
        self, prefer: str = "auto", model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    ):
        self.prefer = prefer
        self.model_id = model_id
        self._pipe = None
        self._failed: str | None = None

    @property
    def learned_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def estimate(
        self,
        image: Image.Image,
        width: int,
        height: int,
        oracle: np.ndarray | None = None,
    ) -> DepthResult:
        if oracle is not None and self.prefer in ("auto", "oracle"):
            return DepthResult(depth=oracle.astype(np.float32), source="oracle-keyframe", meta={})
        if self.prefer in ("auto", "da-v2") and self.learned_available and self._failed is None:
            try:
                return depth_anything_v2(image, width, height, self.model_id)
            except Exception as exc:  # noqa: BLE001 - model download / OOM / version drift
                self._failed = str(exc)
                log.warning("DA-V2 unavailable (%s); using heuristic depth", exc)
        return heuristic_depth(image, width, height)
