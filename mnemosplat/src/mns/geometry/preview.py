"""Software splat rasteriser: proof of geometry without a GPU or a headset.

Used for QA (does the lift look like the keyframe?), for the slide deck (turntable),
and for CI (a PNG that a test can assert is not empty). A real renderer is
GaussianSplats3D in the browser; this exists so the Python side is verifiable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .gaussians import GaussianCloud


def _camera_basis(cam, look, up=(0.0, 1.0, 0.0)):
    eye = np.asarray(cam, dtype=np.float64)
    target = np.asarray(look, dtype=np.float64)
    f = target - eye
    f /= max(np.linalg.norm(f), 1e-9)
    up = np.asarray(up, dtype=np.float64)
    r = np.cross(f, up)
    r /= max(np.linalg.norm(r), 1e-9)
    u = np.cross(r, f)
    return eye, np.stack([r, u, f], axis=-1)


def rasterize(
    cloud: GaussianCloud,
    size: tuple[int, int] = (720, 720),
    camera_position=None,
    look_at=None,
    fov_deg: float = 60.0,
    max_splats: int = 40_000,
    background: tuple[int, int, int] = (10, 12, 18),
    seed: int = 0,
) -> Image.Image:
    w, h = size
    cam = np.asarray(
        camera_position if camera_position is not None else cloud.camera_position, float
    )
    look = np.asarray(look_at if look_at is not None else cloud.camera_look_at, float)
    eye, basis = _camera_basis(cam, look)

    idx = np.arange(len(cloud))
    if len(idx) > max_splats:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_splats, replace=False))

    p = cloud.xyz[idx].astype(np.float64) - eye
    xc, yc, zc = p @ basis[:, 0], p @ basis[:, 1], p @ basis[:, 2]
    keep = zc > 0.08
    if not keep.any():
        return Image.new("RGB", size, background)
    xc, yc, zc = xc[keep], yc[keep], zc[keep]
    idx = idx[keep]

    fx = 0.5 * w / np.tan(np.radians(fov_deg) * 0.5)
    fy = fx
    u = fx * xc / zc + w * 0.5
    v = -fy * yc / zc + h * 0.5
    radii = np.clip(2.6 * (cloud.scales[idx, 0] * fx / zc), 0.8, 90.0)

    colors = cloud.colors[idx].astype(np.float32)
    alphas = cloud.opacity[idx].astype(np.float32)

    acc_c = np.zeros((h, w, 3), dtype=np.float32)
    acc_a = np.zeros((h, w), dtype=np.float32)

    order = np.argsort(-zc)  # painter's algorithm: far first
    for i in order:
        r = radii[i]
        x0, x1 = int(max(0, u[i] - r)), int(min(w, u[i] + r + 1))
        y0, y1 = int(max(0, v[i] - r)), int(min(h, v[i] + r + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d2 = ((xx - u[i]) ** 2 + (yy - v[i]) ** 2) / max(r * r, 1e-6)
        wgt = np.exp(-1.8 * d2) * alphas[i]
        acc_c[y0:y1, x0:x1] += wgt[..., None] * colors[i]
        acc_a[y0:y1, x0:x1] += wgt

    alpha = 1.0 - np.exp(-acc_a)
    safe = np.maximum(acc_a, 1e-6)[..., None]
    rgb = acc_c / safe
    bg = np.array(background, dtype=np.float32) / 255.0
    out = rgb * alpha[..., None] + bg * (1 - alpha[..., None])
    out = np.clip(out, 0, 1) ** (1 / 1.06)
    return Image.fromarray((out * 255).astype(np.uint8), "RGB")


def save_preview(cloud: GaussianCloud, path: str | Path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rasterize(cloud, **kwargs).save(path)
    return path


def turntable(
    cloud: GaussianCloud,
    out_dir: str | Path,
    views: int = 5,
    distance: float | None = None,
    size: tuple[int, int] = (512, 512),
    span_deg: float = 32.0,
    max_splats: int = 55_000,
) -> list[Path]:
    """Small parallax arc around the lift camera.

    A full 360 orbit is dishonest for a single-view lift: the side views are a thin
    sheet, which reads as a broken asset rather than as a memory. A +/-27 degree arc
    proves the volume has depth while staying inside the geometry the depth model
    actually observed.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = np.asarray(cloud.camera_position, dtype=np.float64)
    if distance is not None:
        radius = float(distance)
        base = base / max(np.linalg.norm(base), 1e-9) * radius
    angles = np.linspace(-0.5 * np.radians(span_deg), 0.5 * np.radians(span_deg), views)
    paths: list[Path] = []
    for i, angle in enumerate(angles):
        ca, sa = np.cos(angle), np.sin(angle)
        cam = np.array([base[0] * ca + base[2] * sa, base[1], -base[0] * sa + base[2] * ca])
        img = rasterize(
            cloud,
            size=size,
            camera_position=cam.tolist(),
            look_at=[0.0, 0.0, 0.0],
            max_splats=max_splats,
        )
        p = out / f"turntable-{i:02d}.png"
        img.save(p)
        paths.append(p)
    return paths
