"""Stage 3: single-view depth lifting -> cognitive Gaussian primitives.

The physics that matters
-----------------------
The size of one Gaussian must be derived from its *pixel footprint at its own depth*:
``s = z / f * stride``. If you use a constant scale, near objects turn into mush and
far objects turn into holes. This one line is the difference between a volumetric
memory and a confetti explosion.

Coordinate frame
----------------
Lifting happens in camera space (X right, **Y down**, Z forward) because that is what
``u = fx*X/Z + cx`` gives you for free. Before export we rotate 180 degrees about X so
the cloud lands in the three.js convention (Y up, -Z forward, camera at +Z looking at
the origin). The WebXR client then needs zero coordinate hacks.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

import numpy as np

from ..fusion.attention import AttentionField, resize_bilinear
from ..schemas import CameraModel
from ..util import clamp
from .camera import basis_facing, quat_from_matrix

#: camera-space (Y down, +Z fwd)  ->  three.js / WebXR (Y up, -Z fwd)
_TO_THREE = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


@dataclass
class GaussianConfig:
    target_splats: int = 120_000
    hard_cap: int = 262_144  # 512x512; the documented WebXR budget
    surfel_scale: float = 1.02  # multiple of one pixel footprint
    diffuse_gain: float = 0.95  # how much low-salience splats expand
    thickness: float = 0.32  # flat-surfel aspect (local z / local x)
    opacity_floor: float = 0.06
    prune_alpha: float = 0.05
    max_scale_m: float = 0.09
    min_depth_m: float = 0.20
    max_depth_m: float = 14.0
    environment_shell: bool = True
    shell_radius_m: float = 7.5
    shell_alpha: float = 0.14
    seed: int = 11


@dataclass
class GaussianCloud:
    xyz: np.ndarray  # (N,3) float32, three.js frame, metres
    scales: np.ndarray  # (N,3) float32, linear std-dev in metres
    quats: np.ndarray  # (N,4) float32 (w,x,y,z)
    colors: np.ndarray  # (N,3) float32 in [0,1]
    opacity: np.ndarray  # (N,) float32 in [0,1]
    camera_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 3.0])
    camera_look_at: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    bounds: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    def summary(self) -> dict:
        if len(self) == 0:
            return {"n": 0}
        return {
            "n": len(self),
            "bounds": self.bounds,
            "opacity_mean": round(float(self.opacity.mean()), 4),
            "opacity_p95": round(float(np.percentile(self.opacity, 95)), 4),
            "scale_median_m": round(float(np.median(self.scales[:, 0])), 5),
            "scale_p95_m": round(float(np.percentile(self.scales[:, 0], 95)), 5),
            "camera_position": [round(float(v), 4) for v in self.camera_position],
        }

    def subsample(self, max_n: int, seed: int = 0) -> GaussianCloud:
        if len(self) <= max_n:
            return self
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(self), size=max_n, replace=False))
        return GaussianCloud(
            self.xyz[idx],
            self.scales[idx],
            self.quats[idx],
            self.colors[idx],
            self.opacity[idx],
            self.camera_position,
            self.camera_look_at,
            self.bounds,
            self.meta,
        )


def _grid_for(h: int, w: int, hard_cap: int) -> tuple[int, np.ndarray, np.ndarray]:
    """Choose an integer sampling stride that keeps the raw grid near the budget."""
    stride = max(1, int(np.ceil(np.sqrt((h * w) / (hard_cap * 1.35)))))
    vv = np.arange(0, h, stride)
    uu = np.arange(0, w, stride)
    uu, vv = np.meshgrid(uu, vv)
    return stride, uu.ravel(), vv.ravel()


def _environment_shell(
    image: np.ndarray, camera: CameraModel, cfg: GaussianConfig, origin: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A very sparse, very faint shell so turning your head in VR is not the void.

    Directions are sampled in the *final* three.js frame (Y up, -Z forward), then
    back-projected into the keyframe for colour. Outside the frame we clamp to the
    border, so the shell reads as peripheral memory haze rather than geometry.
    """
    h, w = image.shape[:2]
    az = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    el = np.linspace(-np.pi / 2 + 0.06, np.pi / 2 - 0.06, 28)
    A, E = np.meshgrid(az, el)
    d3 = np.stack(
        [np.cos(E) * np.sin(A), np.sin(E), -np.cos(E) * np.cos(A)], axis=-1
    ).reshape(-1, 3)
    d_cam = d3 @ _TO_THREE.T  # three -> camera frame (the matrix is its own inverse)

    z = d_cam[:, 2]
    safe = np.maximum(np.abs(z), 0.2)
    u = np.clip((camera.fx * d_cam[:, 0] / safe + camera.cx).astype(int), 0, w - 1)
    v = np.clip((camera.fy * d_cam[:, 1] / safe + camera.cy).astype(int), 0, h - 1)
    colors = image[v, u, :3].astype(np.float32) / 255.0

    horizon = np.clip(0.5 + 0.5 * np.sin(E).ravel(), 0, 1)[:, None]
    colors = np.clip(colors * (0.5 + 0.55 * horizon) + 0.015, 0, 1)

    pos = origin[None, :] + d3 * cfg.shell_radius_m
    # each shell splat must cover roughly one angular cell (~0.1 rad), not the
    # whole sky: 0.06 * radius ~= 0.45 m at 7.5 m
    scale = np.full((len(d3), 3), 0.045 * cfg.shell_radius_m, dtype=np.float32)
    q = quat_from_matrix(np.einsum("ij,njk->nik", _TO_THREE, basis_facing(-d_cam)))
    alpha = np.full(len(d3), cfg.shell_alpha, dtype=np.float32)
    return pos.astype(np.float32), scale, q, colors.astype(np.float32), alpha


def init_from_frame(
    image: np.ndarray,
    depth: np.ndarray,
    field: AttentionField,
    camera: CameraModel,
    cfg: GaussianConfig | None = None,
    global_salience: float = 0.5,
) -> GaussianCloud:
    """Unproject a keyframe and attach cognitive opacity/scale from the attention field.

    Parameters
    ----------
    image
        ``(h, w, 3)`` uint8 RGB keyframe.
    depth
        ``(h, w)`` metric depth in metres, same resolution as ``image``.
    field
        EEG-conditioned attention field. Shape may differ; it is resized.
    """
    cfg = cfg or GaussianConfig()
    image = np.asarray(image)
    depth = np.asarray(depth, dtype=np.float32)
    if image.shape[:2] != depth.shape:
        raise ValueError(f"image {image.shape[:2]} and depth {depth.shape} must match")
    h, w = depth.shape

    stride, uu, vv = _grid_for(h, w, cfg.hard_cap)
    z = depth[vv, uu]
    rgb = image[vv, uu, :3]

    M = resize_bilinear(field.map, (h, w))[vv, uu].astype(np.float32)
    M = np.clip(M, 0.0, 1.0)

    valid = np.isfinite(z) & (z > cfg.min_depth_m) & (z < cfg.max_depth_m)
    if valid.sum() < 16:
        raise ValueError("depth map has no usable pixels for lifting")

    z = z[valid]
    rgb = rgb[valid]
    M = M[valid]
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)

    x = (u - camera.cx) * z / camera.fx
    y = (v - camera.cy) * z / camera.fy

    # pixel footprint at this depth -> physically correct splat size
    footprint = (z / camera.fx) * stride
    # peripheral splats spread; a low-vividness epoch spreads the *whole* volume
    spread = 1.0 + cfg.diffuse_gain * (1.0 - M) + 0.45 * (1.0 - global_salience)
    s_xy = np.clip(cfg.surfel_scale * footprint * spread, 1e-4, cfg.max_scale_m)
    s_z = np.clip(s_xy * cfg.thickness, 1e-4, cfg.max_scale_m)

    # cognitive density: vivid memory = opaque and tight, peripheral = translucent, wide
    base_alpha = clamp(0.48 + 0.48 * global_salience, 0.45, 0.97)
    gamma_alpha = clamp(field.contrast, 1.0, 3.0)
    alpha = base_alpha * (cfg.opacity_floor + (1.0 - cfg.opacity_floor) * np.power(M, gamma_alpha))

    # global axis conversion + orient each splat as a surfel facing the camera
    xyz_cam = np.stack([x, y, z], axis=-1)
    forward = xyz_cam / np.maximum(np.linalg.norm(xyz_cam, axis=1, keepdims=True), 1e-9)
    R = basis_facing(forward)  # (N,3,3) columns right/up/forward
    R = np.einsum("ij,njk->nik", _TO_THREE, R)
    quats = quat_from_matrix(R)
    xyz = xyz_cam @ _TO_THREE.T  # (N,3)

    keep = alpha >= cfg.prune_alpha
    if keep.sum() < 16:
        keep = np.ones_like(alpha, dtype=bool)

    xyz, s_xy, s_z, quats, alpha, rgb, M = (
        xyz[keep],
        s_xy[keep],
        s_z[keep],
        quats[keep],
        alpha[keep],
        rgb[keep],
        M[keep],
    )

    # deterministic salience-weighted cap: keep the focused geometry dense, thin the haze
    rng = np.random.default_rng(cfg.seed)
    n = len(alpha)
    if n > cfg.target_splats:
        p = 0.35 + 0.65 * M
        target_p = cfg.target_splats / max(1, int((p).sum()))
        sel = rng.random(n) < np.minimum(1.0, p * target_p)
        if sel.sum() > cfg.target_splats:
            idx = rng.choice(np.flatnonzero(sel), size=cfg.target_splats, replace=False)
            sel = np.zeros(n, dtype=bool)
            sel[idx] = True
        if sel.sum() < 64:
            sel = np.ones(n, dtype=bool)
        xyz, s_xy, s_z, quats, alpha, rgb, M = (
            xyz[sel],
            s_xy[sel],
            s_z[sel],
            quats[sel],
            alpha[sel],
            rgb[sel],
            M[sel],
        )

    # Recentre the volume on the origin; the virtual camera moves back accordingly.
    # The camera sits at -centroid in camera space, which after the axis conversion is
    # (0, ~0, +median depth) in three.js space: the default three camera position.
    # `xyz` is already in the three.js frame here, so the camera is simply at -centroid.
    centroid = xyz.mean(axis=0)
    xyz = xyz - centroid
    cam_pos = (-centroid).tolist()

    scales = np.stack([s_xy, s_xy, s_z], axis=-1).astype(np.float32)
    colors = np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)

    # bounds are for the *memory volume*, not the environment shell
    bounds = {
        "min": [round(float(v), 4) for v in xyz.min(axis=0)],
        "max": [round(float(v), 4) for v in xyz.max(axis=0)],
    }

    if cfg.environment_shell:
        sx, ss, sq, sc, sa = _environment_shell(
            image, camera, cfg, np.asarray(cam_pos, dtype=np.float32)
        )
        xyz = np.concatenate([xyz, sx.astype(np.float32)], axis=0)
        scales = np.concatenate([scales, ss.astype(np.float32)], axis=0)
        quats = np.concatenate([quats, sq.astype(np.float32)], axis=0)
        colors = np.concatenate([colors, sc.astype(np.float32)], axis=0)
        alpha = np.concatenate([alpha, sa.astype(np.float32)], axis=0)

    return GaussianCloud(
        xyz=xyz.astype(np.float32),
        scales=scales,
        quats=quats.astype(np.float32),
        colors=colors.astype(np.float32),
        opacity=np.clip(alpha, 0.0, 1.0).astype(np.float32),
        camera_position=cam_pos,
        camera_look_at=[0.0, 0.0, 0.0],
        bounds=bounds,
        meta={
            "stride": stride,
            "raw_pixels": int(h * w),
            "kept_before_cap": int(n),
            "gaussians": int(xyz.shape[0]),
            "global_salience": round(float(global_salience), 4),
            "config": asdict(cfg),
            "frame": "threejs-y-up",
            "seed": hashlib.sha1(str(cfg.seed).encode()).hexdigest()[:8],
        },
    )
