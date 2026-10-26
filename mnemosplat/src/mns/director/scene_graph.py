"""Cognitive director: transcript -> validated, spatially sane scene graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..schemas import BBox, Beat, SceneGraph, SceneObject
from ..util import clamp, slugify
from .layout import LayoutReport, sanitize_graph
from .meta_client import MetaAPIError, MetaModelAPI, SegmentResult
from .prompts import SYSTEM, USER_TEMPLATE, keyframe_prompt, scene_graph_schema
from .recipes import GENERIC, heuristic_graph, pick_recipe

log = logging.getLogger("mns.director")

__all__ = [
    "DirectorResult",
    "build_scene_graph",
    "heuristic_graph",
    "keyframe_prompt",
    "refine_with_sam",
    "scene_graph_schema",
]


@dataclass
class DirectorResult:
    graph: SceneGraph
    layout: LayoutReport
    used_model: bool
    model: str | None
    warning: str | None = None


def _coerce_graph(raw: dict, narrative: str, source: str, model: str | None) -> SceneGraph:
    palette = [str(c) for c in (raw.get("palette") or []) if isinstance(c, str)][:5]
    objects: list[SceneObject] = []
    seen: set[str] = set()
    for i, o in enumerate(raw.get("objects") or []):
        if not isinstance(o, dict):
            continue
        try:
            bbox = BBox(**{k: float(o["bbox"][k]) for k in ("u0", "v0", "u1", "v1")})
        except (KeyError, TypeError, ValueError):
            continue
        oid = slugify(o.get("id") or o.get("label") or f"obj{i}")
        if oid in seen:
            oid = f"{oid}-{i}"
        seen.add(oid)
        objects.append(
            SceneObject(
                id=oid,
                label=str(o.get("label") or oid),
                bbox=bbox,
                depth_hint=clamp(float(o.get("depth_hint", 0.5)), 0.0, 1.0),
                base_salience=clamp(float(o.get("base_salience", 0.5)), 0.0, 1.0),
                prompt=str(o.get("prompt") or o.get("label") or ""),
                color=str(o.get("color") or (palette[i % len(palette)] if palette else "#8a8f98")),
            )
        )

    raw_beats = [b for b in (raw.get("beats") or []) if isinstance(b, dict)]
    if not raw_beats:
        raw_beats = [
            {"label": "recall", "focus": [o.id for o in objects[:2]], "narrative": narrative}
        ]
    n = len(raw_beats)
    ids = {o.id for o in objects}
    beats = [
        Beat(
            index=i,
            t0=i / n,
            t1=(i + 1) / n,
            label=str(b.get("label") or f"beat {i + 1}"),
            focus=[f for f in (b.get("focus") or []) if f in ids],
            narrative=str(b.get("narrative") or ""),
        )
        for i, b in enumerate(raw_beats)
    ]
    return SceneGraph(
        title=str(raw.get("title") or "A remembered place"),
        narrative=narrative,
        style=str(raw.get("style") or "photorealistic memory, 35mm"),
        lighting=str(raw.get("lighting") or "soft ambient daylight"),
        palette=palette,
        objects=objects,
        beats=beats,
        source=source,
        model=model,
    )


def build_scene_graph(
    narrative: str,
    api: MetaModelAPI | None = None,
    viewpoint: str = "standing eye height, looking slightly down into the room",
    mood: str = "quiet, reflective",
    width: int = 512,
    height: int = 512,
    allow_network: bool = True,
) -> DirectorResult:
    """Muse Spark if possible, hand-authored recipe otherwise. Never raises."""
    narrative = (narrative or "").strip() or GENERIC.title
    warning: str | None = None

    if api is not None and api.available and allow_network:
        try:
            raw, model = api.chat_json(
                SYSTEM,
                USER_TEMPLATE.format(
                    narrative=narrative,
                    viewpoint=viewpoint,
                    mood=mood,
                    width=width,
                    height=height,
                    aspect=width / height,
                ),
                scene_graph_schema(),
            )
            graph = _coerce_graph(raw, narrative, source="meta", model=model)
            if len(graph.objects) >= 3:
                graph, report = sanitize_graph(graph, width, height)
                return DirectorResult(graph, report, True, model)
            warning = "model returned fewer than 3 usable objects; using offline recipe"
        except (MetaAPIError, ValueError, KeyError) as exc:
            warning = f"director fallback: {exc}"
            log.warning(warning)

    # offline: match a recipe by keyword, but keep the user's own transcript
    graph = pick_recipe(narrative).build(narrative)
    if len(graph.objects) < 3:
        graph = GENERIC.build(narrative)
    graph, report = sanitize_graph(graph, width, height)
    return DirectorResult(graph, report, False, None, warning)


def refine_with_sam(
    graph: SceneGraph,
    api: MetaModelAPI,
    image_b64: str,
    max_objects: int = 5,
    blend: float = 0.45,
    min_iou: float = 0.08,
) -> tuple[SceneGraph, dict[str, object], list[dict]]:
    """Snap semantic boxes onto real pixels using SAM 3.1.

    The language model decides *what* is in the scene; SAM decides *exactly where*.
    We blend rather than replace (``blend``) so a bad segmentation cannot throw an
    object out of frame, and we require a loose IoU match so masks are never
    assigned to the wrong entity.

    Returns ``(graph, masks, diagnostics)``.
    """
    import numpy as np

    from ..fusion.attention import bbox_mask

    masks: dict[str, np.ndarray] = {}
    diag: list[dict] = []
    objects = list(graph.objects)

    def iou(a: BBox, b: BBox) -> float:
        a, b = a.ordered(), b.ordered()
        iw = max(0.0, min(a.u1, b.u1) - max(a.u0, b.u0))
        ih = max(0.0, min(a.v1, b.v1) - max(a.v0, b.v0))
        inter = iw * ih
        union = a.area + b.area - inter
        return inter / union if union > 1e-9 else 0.0

    for obj in sorted(objects, key=lambda o: -o.bbox.area)[:max_objects]:
        try:
            segments: list[SegmentResult] = api.segment(image_b64, obj.label)
        except MetaAPIError as exc:
            diag.append({"object": obj.id, "error": str(exc)})
            continue
        if not segments:
            diag.append({"object": obj.id, "matched": False, "reason": "no segments"})
            continue
        best = max(segments, key=lambda s: iou(obj.bbox, BBox.from_xywh(*_xywh(s.bbox))))
        score = iou(obj.bbox, BBox.from_xywh(*_xywh(best.bbox)))
        if score < min_iou:
            diag.append({"object": obj.id, "matched": False, "iou": round(score, 3)})
            continue

        u0, v0, u1, v1 = best.bbox
        nb = BBox(
            u0=(1 - blend) * obj.bbox.u0 + blend * u0,
            v0=(1 - blend) * obj.bbox.v0 + blend * v0,
            u1=(1 - blend) * obj.bbox.u1 + blend * u1,
            v1=(1 - blend) * obj.bbox.v1 + blend * v1,
        ).ordered()
        obj.bbox = nb

        if best.mask:
            from .meta_client import decode_mask

            decoded = decode_mask(best.mask)
            if decoded is not None:
                masks[obj.id] = decoded.astype("float32")
                diag.append(
                    {"object": obj.id, "matched": True, "iou": round(score, 3), "mask": "sam"}
                )
                continue
        masks[obj.id] = bbox_mask(nb, (512, 512))
        diag.append(
            {
                "object": obj.id,
                "matched": True,
                "iou": round(score, 3),
                "mask": "box",
                "confidence": round(best.confidence, 3),
            }
        )

    return graph.model_copy(update={"objects": objects}), masks, diag


def _xywh(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    u0, v0, u1, v1 = bbox
    return (u0, v0, max(1e-3, u1 - u0), max(1e-3, v1 - v0))
