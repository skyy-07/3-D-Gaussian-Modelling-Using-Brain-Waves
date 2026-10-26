"""Keyframe providers: Muse Image (production) and procedural (offline)."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

from PIL import Image

from ..schemas import SceneGraph
from .procedural import RenderResult, depth_preview, render_scene

log = logging.getLogger("mns.render")

__all__ = [
    "KeyframeError",
    "KeyframeResult",
    "RenderResult",
    "depth_preview",
    "muse_keyframe",
    "procedural_keyframe",
]


class KeyframeError(RuntimeError):
    """Raised when the primary provider cannot produce a frame."""


@dataclass
class KeyframeResult:
    image: Image.Image
    provider: str
    prompt: str
    model: str | None = None
    meta: dict | None = None


def procedural_keyframe(
    graph: SceneGraph,
    width: int,
    height: int,
    salience: dict[str, float] | None = None,
    seed: int = 3,
) -> tuple[KeyframeResult, RenderResult]:
    r = render_scene(graph, width=width, height=height, salience=salience, seed=seed)
    return (
        KeyframeResult(
            image=r.keyframe,
            provider="procedural",
            prompt="(offline scene-graph render)",
            meta=r.meta,
        ),
        r,
    )


def muse_keyframe(
    graph: SceneGraph,
    api,
    width: int,
    height: int,
    beat_label: str | None = None,
    beat_narrative: str = "",
) -> KeyframeResult:
    """Generate the remembered keyframe with ``muse-image-1.0``."""
    from ..director.prompts import keyframe_prompt

    prompt = keyframe_prompt(graph, width, height, beat_label)
    if beat_narrative:
        prompt += f" Emotional register: {beat_narrative}"
    try:
        blobs = api.generate_image(
            prompt,
            size=f"{max(512, width)}x{max(512, height)}",
            output_format="png",
            reasoning_strength="high",
        )
    except Exception as exc:  # noqa: BLE001 - provider failure must degrade
        raise KeyframeError(str(exc)) from exc

    img = Image.open(io.BytesIO(blobs[0])).convert("RGB")
    img = _cover_resize(img, width, height)
    return KeyframeResult(
        image=img,
        provider="muse-image-1.0",
        prompt=prompt,
        model="muse-image-1.0",
        meta={"cropped_to": [width, height]},
    )


def _cover_resize(img: Image.Image, width: int, height: int) -> Image.Image:
    """Centre-crop to the target aspect, then resize (never letterbox a splat frame)."""
    src_ratio = img.width / img.height
    dst_ratio = width / height
    if src_ratio > dst_ratio:
        new_w = int(round(img.height * dst_ratio))
        left = (img.width - new_w) // 2
        img = img.crop((left, 0, left + new_w, img.height))
    elif src_ratio < dst_ratio:
        new_h = int(round(img.width / dst_ratio))
        top = (img.height - new_h) // 2
        img = img.crop((0, top, img.width, top + new_h))
    return img.resize((width, height), Image.Resampling.LANCZOS)
