"""Offline keyframe synthesis.

The Muse Image path is the production path (see ``muse_image.py``), but a hackathon
demo cannot depend on a network round-trip per timeline frame. This renderer builds a
plausible photograph of the *scene graph itself* - same boxes, same depth ordering,
same salience - so the geometry, the exporter and the headset build can be developed
and rehearsed offline.

It also returns a **metric depth map** and **per-object masks** for free. That is not
cheating: the depth belongs to the synthesised keyframe, not to any ground-truth
stimulus, so the "no ground-truth leakage" property is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..schemas import BBox, SceneGraph, SceneObject
from ..util import clamp

# metric depth range for a remembered room, in metres
Z_NEAR, Z_FAR = 0.85, 6.2


@dataclass
class RenderResult:
    keyframe: Image.Image
    depth: np.ndarray  # (h, w) float32 metres
    masks: dict[str, np.ndarray]  # (h, w) float32 in [0,1]
    meta: dict


def depth_from_hint(hint: float) -> float:
    """Non-linear but monotonic: arm's length -> far wall."""
    return float(Z_NEAR + (Z_FAR - Z_NEAR) * clamp(hint, 0.0, 1.0) ** 1.55)


def _hex(c: str) -> tuple[int, int, int]:
    c = (c or "#888888").lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    try:
        return tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    except ValueError:
        return (136, 143, 152)


def _mix(a, b, t: float):
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def _shade(c, f: float):
    """Scale a 0-255 colour. Must keep fractional precision: truncating here turns
    every mid-tone into black (a bug that cost us an hour and one very dark room)."""
    return tuple(clamp(float(v) * f, 0.0, 255.0) for v in c)


def _gradient(h: int, w: int, top, bottom, radial: tuple | None = None) -> np.ndarray:
    """Vertical gradient + optional radial light. Returns float32 (h,w,3)."""
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    top = np.array(top, dtype=np.float32) / 255.0
    bottom = np.array(bottom, dtype=np.float32) / 255.0
    img = top * (1 - t) + bottom * t
    img = np.repeat(img, w, axis=1)
    if radial is not None:
        cx, cy, radius, strength = radial
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.sqrt((xx / max(1, w) - cx) ** 2 + (yy / max(1, h) - cy) ** 2) / max(1e-3, radius)
        glow = np.exp(-d * d * 1.6)[..., None] * strength
        img = np.clip(img + glow, 0, 1)
    return img


def _noise(h: int, w: int, rng: np.random.Generator, sigma: float = 0.035) -> np.ndarray:
    small = rng.normal(0, sigma, size=(max(2, h // 4), max(2, w // 4), 1)).astype(np.float32)
    big = np.asarray(
        Image.fromarray(((small + 0.5) * 255).clip(0, 255).astype(np.uint8)[..., 0]).resize(
            (w, h), Image.Resampling.BICUBIC
        ),
        dtype=np.float32,
    )[..., None] / 255.0 - 0.5
    return big


def _rounded_mask(w: int, h: int, radius: int) -> np.ndarray:
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w - 1, h - 1), radius=max(1, radius), fill=255)
    return np.asarray(m, dtype=np.float32) / 255.0


def _cat(label: str, prompt: str) -> str:
    s = f"{label} {prompt}".lower()
    for key, cat in (
        ("window", "window"),
        ("sky", "sky"),
        ("wall", "wall"),
        ("road", "ground"),
        ("puddle", "ground"),
        ("desk", "surface"),
        ("table", "surface"),
        ("counter", "surface"),
        ("bed", "surface"),
        ("shelf", "surface"),
        ("bench", "surface"),
        ("chair", "rect"),
        ("building", "rect"),
        ("door", "rect"),
        ("mug", "cyl"),
        ("kettle", "cyl"),
        ("bowl", "cyl"),
        ("lamp", "lamp"),
        ("plant", "plant"),
        ("fern", "plant"),
        ("tree", "plant"),
        ("book", "flat"),
        ("notebook", "flat"),
        ("pillow", "flat"),
        ("rug", "flat"),
    ):
        if key in s:
            return cat
    return "rect"


def _paint_object(obj: SceneObject, size: tuple[int, int], rng, salience: float):
    """Return (rgba float32 image, alpha mask) for one object at one size."""
    w, h = size
    if w < 2 or h < 2:
        return None, None
    cat = _cat(obj.label, obj.prompt)
    base = np.array(_hex(obj.color), dtype=np.float32)  # 0..255
    cat_bright = clamp(0.68 + 0.34 * salience, 0.62, 1.05)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    v = yy / max(1, h - 1)
    u = xx / max(1, w - 1)

    alpha = _rounded_mask(
        w, h, radius=int(min(w, h) * (0.06 if cat not in ("cyl", "plant") else 0.3))
    )

    if cat == "sky":
        img = _gradient(h, w, _shade(base, 1.25), _shade(base, 0.55))
    elif cat == "wall":
        img = _gradient(h, w, _shade(base, 1.08), _shade(base, 0.72))
    elif cat == "window":
        sky = _gradient(h, w, (232, 241, 250), (176, 199, 219))
        img = sky * 0.96
        frame = np.array(_shade((58, 63, 70), 1.0), dtype=np.float32) / 255.0
        bar = (np.abs(u - 0.5) < 0.022) | (np.abs(v - 0.46) < 0.018)
        img[bar] = frame
        img *= (1.0 - 0.25 * (1 - alpha))[..., None]
        img = np.clip(img * 1.06, 0, 1)
    elif cat == "surface":
        top = _shade(base, 1.18)
        bottom = _shade(base, 0.62)
        img = _gradient(h, w, top, bottom)
        # front edge shadow band
        img *= np.where(v > 0.82, 0.72, 1.0)[..., None]
    elif cat == "ground":
        img = _gradient(h, w, _shade(base, 0.9), _shade(base, 0.55))
    elif cat == "cyl":
        # vertical highlight offset from centre
        d = np.sqrt((u - 0.36) ** 2 + ((v - 0.42) * 0.8) ** 2)
        spec = np.exp(-(d**2) * 9.0)[..., None]
        img = _gradient(h, w, _shade(base, 1.22), _shade(base, 0.55)) + spec * 0.28
        img = np.clip(img, 0, 1)
    elif cat == "lamp":
        img = _gradient(h, w, _shade(base, 1.35), _shade(base, 0.6))
        glow = np.exp(-(((u - 0.5) ** 2 + (v - 0.25) ** 2) * 6.0))[..., None]
        img = np.clip(img + glow * 0.45, 0, 1)
    elif cat == "plant":
        img = _gradient(h, w, _shade(base, 1.1), _shade(base, 0.5))
        # leaf-ish speckle
        leaves = (np.sin(u * 23 + v * 7) * np.cos(v * 19 - u * 5)) > 0.15
        img = np.where(leaves[..., None], img * 1.25, img * 0.7)
        alpha *= (0.55 + 0.45 * leaves).astype(np.float32)
    elif cat == "flat":
        img = _gradient(h, w, _shade(base, 1.1), _shade(base, 0.82))
    else:  # rect
        img = _gradient(h, w, _shade(base, 1.12), _shade(base, 0.66))

    img = np.clip(img * cat_bright + _noise(h, w, rng, 0.025), 0, 1)

    # contact shadow inside the bottom edge
    contact = np.clip((v - 0.9) / 0.1, 0, 1)[..., None] * 0.35
    img = np.clip(img * (1 - contact), 0, 1)

    rgba = np.dstack([img, alpha]).astype(np.float32)
    return rgba, alpha


def render_scene(
    graph: SceneGraph,
    width: int = 512,
    height: int = 512,
    salience: dict[str, float] | None = None,
    seed: int = 3,
    grain: float = 0.014,
) -> RenderResult:
    """Painter's-algorithm render of the scene graph with DOF tied to salience."""
    rng = np.random.default_rng(seed)
    sal = salience or {o.id: o.base_salience for o in graph.objects}

    palette = [_hex(c) for c in (graph.palette or ["#262a31", "#69727d"])]
    bg_top = _mix(palette[0], (255, 255, 255), 0.10)
    bg_bottom = _mix(palette[min(2, len(palette) - 1)], (24, 24, 24), 0.18)

    # light source near the brightest large object (usually the window)
    light = next((o for o in graph.objects if _cat(o.label, o.prompt) == "window"), None)
    radial = None
    if light is not None:
        cu, cv = light.bbox.center
        radial = (cu, cv, 0.55, 0.10)

    canvas = _gradient(height, width, bg_top, bg_bottom, radial)

    # focus plane = the most salient entity; DOF blurs everything off that plane
    focus_obj = max(graph.objects, key=lambda o: sal.get(o.id, o.base_salience))
    focus_z = depth_from_hint(focus_obj.depth_hint)

    canvas_img = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8), "RGB")
    _paint_floor(canvas_img, graph, palette)

    masks: dict[str, np.ndarray] = {}
    depth = np.full((height, width), depth_from_hint(1.0), dtype=np.float32)

    ordered = sorted(graph.objects, key=lambda o: -o.depth_hint)
    for obj in ordered:
        if _cat(obj.label, obj.prompt) in ("wall",):
            pass
        x1, y1, x2, y2 = obj.bbox.to_pixels(width, height)
        rw, rh = max(2, x2 - x1), max(2, y2 - y1)
        rgba, alpha = _paint_object(obj, (rw, rh), rng, sal.get(obj.id, obj.base_salience))
        if rgba is None or alpha is None:
            continue

        z = depth_from_hint(obj.depth_hint)
        blur = 0.5 + 5.5 * min(1.0, abs(z - focus_z) / 3.2)
        if blur > 0.8:
            layer = Image.fromarray((rgba[..., :3] * 255).astype(np.uint8), "RGB").filter(
                ImageFilter.GaussianBlur(blur)
            )
            rgba[..., :3] = np.asarray(layer, dtype=np.float32) / 255.0
            am = Image.fromarray((alpha * 255).astype(np.uint8), "L").filter(
                ImageFilter.GaussianBlur(blur * 0.6)
            )
            alpha = np.asarray(am, dtype=np.float32) / 255.0

        _paint_contact_shadow(canvas_img, obj, width, height, z, focus_z)

        layer_img = Image.fromarray((np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8), "RGB")
        mask_img = Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8), "L")
        canvas_img.paste(layer_img, (x1, y1), mask_img)

        sub = depth[y1:y2, x1:x2]
        if sub.shape == (rh, rw):
            # slight in-object gradient: top of a surface recedes
            ramp = np.linspace(z * 0.94, z * 1.04, rh, dtype=np.float32)[:, None]
            sub[:] = np.where(alpha > 0.35, ramp, sub)
        m = np.zeros((height, width), dtype=np.float32)
        m[y1:y2, x1:x2] = alpha
        masks[obj.id] = m

    # global grade: vignette, halation, grain
    arr = np.asarray(canvas_img, dtype=np.float32) / 255.0
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    r = np.sqrt(((xx / width) - 0.5) ** 2 + ((yy / height) - 0.5) ** 2) / 0.707
    arr *= (1.0 - 0.28 * np.clip(r, 0, 1) ** 2)[..., None]

    bright = np.clip(arr - 0.72, 0, 1)
    if bright.max() > 0.01:
        halo = np.asarray(
            Image.fromarray(
                (bright.max(axis=2) * 255 / max(1e-6, bright.max())).astype(np.uint8), "L"
            ).filter(ImageFilter.GaussianBlur(9)),
            dtype=np.float32,
        ) / 255.0
        arr = np.clip(arr + halo[..., None] * 0.10 * np.array([1.0, 0.92, 0.78]), 0, 1)

    # gentle S-curve + photographic softness: removes the "clip art" edge
    arr = np.clip(arr, 0, 1)
    arr = np.clip(0.5 + (arr - 0.5) * 1.06, 0, 1)
    arr = np.clip(arr + rng.normal(0, grain, arr.shape).astype(np.float32), 0, 1)
    arr = np.asarray(
        Image.fromarray((arr * 255).astype(np.uint8), "RGB").filter(ImageFilter.GaussianBlur(0.45)),
        dtype=np.float32,
    ) / 255.0

    return RenderResult(
        keyframe=Image.fromarray((arr * 255).astype(np.uint8), "RGB"),
        depth=depth,
        masks=masks,
        meta={
            "renderer": "procedural/v1",
            "focus_object": focus_obj.id,
            "focus_depth_m": round(focus_z, 3),
            "z_range_m": [Z_NEAR, Z_FAR],
            "seed": seed,
        },
    )


def depth_preview(depth: np.ndarray) -> Image.Image:
    """Near = bright. Visual sanity check for the lift stage."""
    z = np.asarray(depth, dtype=np.float32)
    lo, hi = float(np.percentile(z, 2)), float(np.percentile(z, 98))
    n = np.clip((z - lo) / max(1e-6, hi - lo), 0, 1)
    inv = (1.0 - n) ** 0.85
    rgb = np.dstack([inv**0.9, inv**1.2, inv**1.6])
    return Image.fromarray((rgb * 255).astype(np.uint8), "RGB")


def attention_debug_image(base: Image.Image, field, out_path) -> None:
    """Composite keyframe + salience heat overlay for slides and QA."""
    from ..fusion.attention import overlay

    overlay(base, field, alpha=0.5).save(out_path)


def _paint_floor(canvas: Image.Image, graph: SceneGraph, palette) -> None:
    """A soft floor plane behind the furniture. Without it the lower frame reads as wall."""
    w, h = canvas.size
    base = _mix(palette[1] if len(palette) > 1 else palette[0], (30, 26, 22), 0.35)
    top = _shade(base, 0.72)
    bottom = _shade(base, 0.42)
    floor = _gradient(h, w, top, bottom)
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    band = np.clip((yy - 0.68) / 0.10, 0, 1)  # horizon at ~0.68
    band = np.repeat(band, w, axis=1)
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = arr * (1 - band[..., None]) + floor * band[..., None]
    canvas.paste(Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGB"))


def _paint_contact_shadow(
    canvas: Image.Image, obj: SceneObject, width: int, height: int, z: float, focus_z: float
) -> None:
    """Soft dark ellipse under an object; cheap ambient occlusion, huge realism win."""
    if obj.bbox.area > 0.7:
        return
    x1, y1, x2, y2 = obj.bbox.to_pixels(width, height)
    cx = (x1 + x2) / 2
    cy = y2 - (y2 - y1) * 0.06
    rx = max(4.0, (x2 - x1) * 0.55)
    ry = max(2.0, rx * 0.22)
    blur = 1.0 + 5.0 * min(1.0, abs(z - focus_z) / 3.2)
    layer = Image.new("L", (width, height), 0)
    d = ImageDraw.Draw(layer)
    d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=110)
    layer = layer.filter(ImageFilter.GaussianBlur(blur))
    shadow = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(shadow, (0, 0), layer)


def _bbox_px(b: BBox, w: int, h: int):
    return b.to_pixels(w, h)
