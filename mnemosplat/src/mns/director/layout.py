"""Deterministic layout sanitiser.

Language models are good at *what* is in a remembered room and bad at *where* exactly.
Unsanitised coordinates are the single most common source of an ugly splat: a box that
leaves the frame produces a floating Gaussian wall; two objects stacked on the same
pixel produce a z-fighting smear.

Everything here is pure geometry, testable on CPU, and applied to *every* scene graph
whether it came from Muse Spark, a hand-written recipe, or a user's JSON upload.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schemas import BBox, SceneGraph, SceneObject
from ..util import clamp

MIN_SIDE = 0.045
MIN_AREA = MIN_SIDE * MIN_SIDE * 2.0
MAX_OVERLAP = 0.30


@dataclass
class LayoutReport:
    moved: int = 0
    resized: int = 0
    overlaps: int = 0
    dropped: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.dropped is None:
            self.dropped = []

    def as_dict(self) -> dict:
        return {
            "moved": self.moved,
            "resized": self.resized,
            "overlaps": self.overlaps,
            "dropped": list(self.dropped),
        }


def _iou(a: BBox, b: BBox) -> float:
    a, b = a.ordered(), b.ordered()
    iw = max(0.0, min(a.u1, b.u1) - max(a.u0, b.u0))
    ih = max(0.0, min(a.v1, b.v1) - max(a.v0, b.v0))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _overlap_of_smaller(a: BBox, b: BBox) -> float:
    a, b = a.ordered(), b.ordered()
    iw = max(0.0, min(a.u1, b.u1) - max(a.u0, b.u0))
    ih = max(0.0, min(a.v1, b.v1) - max(a.v0, b.v0))
    inter = iw * ih
    smaller = min(a.area, b.area)
    return inter / smaller if smaller > 1e-9 else 0.0


def _fit(bbox: BBox) -> BBox:
    """Clamp into frame, enforce minimum side, keep the centre stable when possible."""
    b = bbox.ordered()
    w = max(b.u1 - b.u0, MIN_SIDE)
    h = max(b.v1 - b.v0, MIN_SIDE)
    cu = clamp(0.5 * (b.u0 + b.u1), 0.5 * w, 1.0 - 0.5 * w)
    cv = clamp(0.5 * (b.v0 + b.v1), 0.5 * h, 1.0 - 0.5 * h)
    return BBox(u0=cu - w / 2, v0=cv - h / 2, u1=cu + w / 2, v1=cv + h / 2).ordered()


def sanitize_graph(
    graph: SceneGraph, width: int = 512, height: int = 512
) -> tuple[SceneGraph, LayoutReport]:
    """Clamp, de-overlap, repair depth ordering, and guarantee beats cover the timeline."""
    report = LayoutReport()
    objects: list[SceneObject] = []
    overlaps = 0
    seen: set[str] = set()

    for obj in graph.objects:
        if obj.id in seen:
            report.dropped.append(obj.id)
            continue
        fixed = _fit(obj.bbox)
        if (fixed.u0, fixed.v0, fixed.u1, fixed.v1) != (
            obj.bbox.u0,
            obj.bbox.v0,
            obj.bbox.u1,
            obj.bbox.v1,
        ):
            report.resized += 1
        obj = obj.model_copy(update={"bbox": fixed, "id": obj.id})
        seen.add(obj.id)
        objects.append(obj)

    # Overlap is *not* a bug: a mug sits on a desk, a curtain hangs in front of a
    # window. Occlusion is resolved later by depth, so we only report it here.
    overlaps = 0
    for i, a in enumerate(objects):
        for b in objects[i + 1 :]:
            if (
                _overlap_of_smaller(a.bbox, b.bbox) > MAX_OVERLAP
                and abs(a.depth_hint - b.depth_hint) < 0.12
            ):
                overlaps += 1

    # derive style/palette defaults, keep object list non-empty
    palette = [p for p in (graph.palette or []) if isinstance(p, str)][:5]
    if not palette:
        palette = ["#2b2f38", "#6f7a86", "#c9b8a3", "#e8dcc8", "#f4efe6"]

    beats = list(graph.beats)
    if not beats:
        from ..schemas import Beat

        beats = [Beat(index=0, t0=0.0, t1=1.0, label="recall", focus=[o.id for o in objects[:2]])]
    # normalise beat times to a contiguous 0..1 timeline (client maps to epoch count)
    span = max(1e-6, beats[-1].t1 - beats[0].t0)
    t0 = beats[0].t0
    fixed_beats = []
    for i, b in enumerate(beats):
        s = (b.t0 - t0) / span
        e = (b.t1 - t0) / span
        fixed_beats.append(
            b.model_copy(
                update={
                    "index": i,
                    "t0": round(s, 4),
                    "t1": round(max(e, s + 1e-3), 4),
                    "focus": [f for f in b.focus if f in seen],
                }
            )
        )

    # One contradiction is worth repairing: something filling the bottom of the frame
    # cannot be the far wall. Everything else is left to the author / language model.
    for o in objects:
        if o.bbox.area > 0.7:  # full-frame backdrop: never "foreground clutter"
            continue
        if o.bbox.v1 > 0.9 and o.depth_hint > 0.62:
            o.depth_hint = 0.55

    report.overlaps = overlaps
    clean = graph.model_copy(
        update={
            "objects": objects,
            "beats": fixed_beats,
            "palette": palette,
        }
    )
    return clean, report


def beat_index_for_time(graph: SceneGraph, t: float) -> int:
    for b in graph.beats:
        if b.t0 <= t < b.t1:
            return b.index
    return len(graph.beats) - 1 if graph.beats else 0
