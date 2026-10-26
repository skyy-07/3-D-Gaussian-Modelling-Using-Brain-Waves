"""Stage 2: EEG-conditioned 2D attention field M(u, v).

Design note that decides whether this project works
---------------------------------------------------
A naive implementation normalises M to peak 1.0 every frame. That destroys the
neural signal: a distracted epoch and a vivid epoch both produce a bright blob, and
the only difference left is shape. We therefore split the two roles explicitly:

* **Shape** comes from the normalised field (where attention sits).
* **Contrast / density** comes from the absolute salience scalar (how strongly the
  subject encoded it). Low-salience epochs get a *higher ambient floor* and a
  *flatter exponent*, so the memory becomes volumetric fog - not a black frame.

This is why ``object_salience`` is carried through to the exporter alongside the map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from ..schemas import BBox, EpochFeatures, SceneGraph
from ..util import clamp

# --- colour ramp for the salience overlay (inferno-ish, no matplotlib dependency) ---
_RAMP = np.array(
    [
        [0.02, 0.02, 0.10],
        [0.12, 0.05, 0.32],
        [0.36, 0.06, 0.42],
        [0.62, 0.14, 0.36],
        [0.83, 0.30, 0.22],
        [0.96, 0.55, 0.08],
        [0.99, 0.83, 0.28],
        [1.00, 0.98, 0.75],
    ],
    dtype=np.float32,
)


@dataclass
class AttentionBlob:
    """One anisotropic Gaussian in normalised screen space."""

    u: float
    v: float
    sigma_u: float
    sigma_v: float
    weight: float
    object_id: str = ""

    def field(self, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        vv = (np.arange(h, dtype=np.float32) / max(1, h - 1))[:, None]
        uu = (np.arange(w, dtype=np.float32) / max(1, w - 1))[None, :]
        du = (uu - self.u) / max(self.sigma_u, 1e-4)
        dv = (vv - self.v) / max(self.sigma_v, 1e-4)
        return np.exp(-0.5 * (du * du + dv * dv)).astype(np.float32)


@dataclass
class AttentionField:
    """M(u,v) plus everything the exporter and the UI need to explain it."""

    map: np.ndarray  # (h, w) float32 in [0, 1]
    blobs: list[AttentionBlob]
    salience: dict[str, float]
    floor: float
    contrast: float
    hemifield: float = 0.0
    masks: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.map.shape[0]), int(self.map.shape[1]))

    def peak(self) -> tuple[float, float, float]:
        idx = int(np.argmax(self.map))
        h, w = self.shape
        return (idx % w / max(1, w - 1), idx // w / max(1, h - 1), float(self.map.flat[idx]))

    def object_density(self, obj_id: str) -> float:
        """Mean attention inside the object's mask (or its blob, if no mask)."""
        if obj_id in self.masks:
            m = self.masks[obj_id]
            denom = float(m.sum())
            if denom > 1e-6:
                return float((self.map * m).sum() / denom)
        b = next((b for b in self.blobs if b.object_id == obj_id), None)
        if b is None:
            return 0.0
        return float(b.weight)


# --------------------------------------------------------------------------- #
# kernel geometry
# --------------------------------------------------------------------------- #
def _spread(certainty: float) -> float:
    """Uncertain memories spread. 0.75x at full certainty, 1.65x at zero."""
    return 0.75 + 0.90 * (1.0 - clamp(certainty, 0.0, 1.0))


def blobs_from_objects(
    graph: SceneGraph,
    salience: dict[str, float],
    certainty: float = 0.5,
    max_sigma: float = 0.32,
    min_sigma: float = 0.028,
) -> list[AttentionBlob]:
    blobs: list[AttentionBlob] = []
    for obj in graph.objects:
        s = clamp(salience.get(obj.id, obj.base_salience), 0.1, 1.0)
        bw, bh = obj.bbox.size
        spread = _spread(certainty)
        sigma_u = clamp(0.62 * bw * spread, min_sigma, max_sigma)
        sigma_v = clamp(0.62 * bh * spread, min_sigma, max_sigma)
        blobs.append(
            AttentionBlob(
                u=obj.bbox.center[0],
                v=obj.bbox.center[1],
                sigma_u=sigma_u,
                sigma_v=sigma_v,
                weight=s,
                object_id=obj.id,
            )
        )

    return blobs


def hemifield_blob(lateralization: float, salience: float) -> AttentionBlob | None:
    if abs(lateralization) < 0.08:
        return None
    u = 0.5 + 0.34 * math.copysign(1.0, lateralization)
    return AttentionBlob(
        u=u,
        v=0.52,
        sigma_u=0.26,
        sigma_v=0.34,
        weight=0.16 * abs(lateralization) * salience,
        object_id="__hemifield__",
    )


# --------------------------------------------------------------------------- #
# rasterisation
# --------------------------------------------------------------------------- #
def resize_bilinear(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Dependency-free bilinear resize for float masks."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3:
        return np.stack([resize_bilinear(a[..., c], shape) for c in range(a.shape[2])], axis=-1)
    h, w = shape
    if a.shape == (h, w):
        return a
    ys = np.linspace(0, a.shape[0] - 1, h)
    xs = np.linspace(0, a.shape[1] - 1, w)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, a.shape[0] - 1)
    x1 = np.minimum(x0 + 1, a.shape[1] - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    top = a[np.ix_(y0, x0)] * (1 - wx) + a[np.ix_(y0, x1)] * wx
    bot = a[np.ix_(y1, x0)] * (1 - wx) + a[np.ix_(y1, x1)] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def build_attention_field(
    graph: SceneGraph,
    salience: dict[str, float],
    epoch: EpochFeatures,
    shape: tuple[int, int] = (512, 512),
    masks: dict[str, np.ndarray] | None = None,
    floor: float = 0.10,
    max_sigma: float = 0.32,
) -> AttentionField:
    """Compose M(u,v) from per-entity salience, kernels and optional SAM masks."""
    blobs = blobs_from_objects(
        graph,
        salience,
        certainty=epoch.certainty,
        max_sigma=max_sigma,
    )
    hb = hemifield_blob(epoch.lateralization, epoch.salience)
    if hb is not None:
        blobs.append(hb)

    acc = np.zeros(shape, dtype=np.float32)
    used_masks: dict[str, np.ndarray] = {}
    for b in blobs:
        f = b.field(shape) * b.weight
        if masks and b.object_id in masks:
            m = resize_bilinear(masks[b.object_id], shape)
            m = np.clip(m, 0.0, 1.0)
            used_masks[b.object_id] = m
            # soften mask edges so the Gaussian still bleeds a little (memory is not
            # pixel-exact); 0.82/0.18 keeps a halo around the segmented object
            f = f * (0.18 + 0.82 * m)
        acc += f

    peak = float(acc.max())
    if peak > 1e-8:
        acc /= peak
    # global vividness -> contrast. Vivid memory: tight peak, dark surround (gamma up).
    # Diffuse memory: the map is nearly flat, so every object gets similar weight.
    contrast = 1.05 + 1.30 * clamp(epoch.salience, 0.1, 1.0)
    shaped = np.power(np.clip(acc, 0.0, 1.0), contrast)
    field = floor + (1.0 - floor) * shaped
    return AttentionField(
        map=field.astype(np.float32),
        blobs=blobs,
        salience={k: float(v) for k, v in salience.items()},
        floor=floor,
        contrast=float(contrast),
        hemifield=float(epoch.lateralization),
        masks=used_masks,
    )


# --------------------------------------------------------------------------- #
# visualisation / interop
# --------------------------------------------------------------------------- #
def colorize(field: AttentionField | np.ndarray) -> Image.Image:
    m = field.map if isinstance(field, AttentionField) else np.asarray(field, dtype=np.float32)
    m = np.clip(m, 0.0, 1.0)
    idx = m * (len(_RAMP) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(_RAMP) - 1)
    frac = (idx - lo)[..., None]
    rgb = (_RAMP[lo] * (1 - frac) + _RAMP[hi] * frac) * 255.0
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def overlay(
    rgb: Image.Image, field: AttentionField | np.ndarray, alpha: float = 0.45
) -> Image.Image:
    base = rgb.convert("RGB")
    heat = colorize(field).resize(base.size, Image.Resampling.BILINEAR)
    return Image.blend(base, heat, clamp(alpha, 0.0, 1.0))


def latent_bias(field: AttentionField | np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Signed bias for cross-attention: aggressive where salient, negative elsewhere.

    Returns float32 ``(h, w)`` in [-1, 1], ready to broadcast over attention logits.
    """
    m = field.map if isinstance(field, AttentionField) else np.asarray(field, dtype=np.float32)
    r = resize_bilinear(m, shape)
    return np.clip((r - float(np.mean(r))) * 2.0, -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# epoch <-> entity alignment
# --------------------------------------------------------------------------- #
def assign_object_salience(
    graph: SceneGraph,
    epochs: list[EpochFeatures],
    focus_boost: float = 0.14,
    hemi_gain: float = 0.22,
) -> list[dict[str, float]]:
    """Map per-epoch neural scalars onto per-entity salience.

    Documented hypothesis (see docs/SCIENCE.md):
      * global gamma/theta sets how strongly everything was encoded;
      * O1/O2 asymmetry biases *which hemifield* was attended;
      * the beat's ``focus`` list (from the language model) is a weak prior only.

    The language prior is capped at +-14% so a hallucinated scene graph cannot
    fabricate salience that the EEG did not show.
    """
    by_id = graph.object_map()
    out: list[dict[str, float]] = []
    # beats live on a 0..1 timeline; epochs are in seconds - normalise before matching
    duration = max(1e-6, epochs[-1].t1 if epochs else 1.0)

    for ep in epochs:
        t = 0.5 * (ep.t0 + ep.t1) / duration
        beat = next((b for b in graph.beats if b.t0 <= t < b.t1), None)
        if beat is None and graph.beats:
            beat = min(graph.beats, key=lambda b: abs(0.5 * (b.t0 + b.t1) - t))
        focus = set(beat.focus) if beat else set()

        # normalised position of this epoch inside the beat, for temporal alignment
        sal: dict[str, float] = {}
        for oid, obj in by_id.items():
            prior = clamp(obj.base_salience, 0.0, 1.0) ** 0.6
            vivid = 0.30 + 0.70 * clamp(ep.salience, 0.1, 1.0)
            cu = obj.bbox.center[0]
            hemi = 1.0 + hemi_gain * clamp(ep.lateralization, -1.0, 1.0) * (2.0 * cu - 1.0)
            boost = 1.0 + focus_boost if oid in focus else 1.0
            sal[oid] = clamp(prior * vivid * hemi * boost, 0.1, 1.0)

        top = max(sal.values()) if sal else 1.0
        if top > 1.0:
            sal = {k: clamp(v / top, 0.1, 1.0) for k, v in sal.items()}
        out.append({k: round(v, 4) for k, v in sal.items()})

    return out


def bbox_mask(bbox: BBox, shape: tuple[int, int], feather: int = 6) -> np.ndarray:
    """Soft rectangular mask, used when SAM is unavailable."""
    h, w = shape
    x1, y1, x2, y2 = bbox.to_pixels(w, h)
    m = np.zeros((h, w), dtype=np.float32)
    m[y1:y2, x1:x2] = 1.0
    if feather > 0:
        from PIL import ImageFilter

        img = Image.fromarray((m * 255).astype(np.uint8), mode="L")
        img = img.filter(ImageFilter.GaussianBlur(feather))
        m = np.asarray(img, dtype=np.float32) / 255.0
    return m


def default_masks(graph: SceneGraph, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {o.id: bbox_mask(o.bbox.ordered(), shape) for o in graph.objects}


def object_summary(graph: SceneGraph, salience: dict[str, float]) -> list[dict]:
    rows = []
    for o in graph.objects:
        rows.append(
            {
                "id": o.id,
                "label": o.label,
                "bbox": o.bbox.model_dump(),
                "salience": round(float(salience.get(o.id, o.base_salience)), 4),
                "depth_hint": o.depth_hint,
            }
        )
    return sorted(rows, key=lambda r: -r["salience"])
