"""Camera maths and quaternion helpers (numpy only)."""

from __future__ import annotations

import numpy as np

from ..schemas import CameraModel

__all__ = [
    "CameraModel",
    "basis_facing",
    "look_at_basis",
    "quat_from_matrix",
    "unproject_pixels",
]


def unproject_pixels(camera: CameraModel, u, v, z) -> np.ndarray:
    """(u,v,z) pixel coords -> (N,3) camera-space points, Y down, Z forward."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    x = (u - camera.cx) * z / camera.fx
    y = (v - camera.cy) * z / camera.fy
    return np.stack([x, y, z], axis=-1)


def basis_facing(forward: np.ndarray, up_hint: np.ndarray | None = None) -> np.ndarray:
    """Orthonormal basis whose local +Z points along ``forward``.

    Used to orient each Gaussian as a surfel facing the camera. Without this, an
    isotropic cloud of single-view splats looks like fog when you move your head.
    """
    f = np.asarray(forward, dtype=np.float64)
    n = np.linalg.norm(f, axis=-1, keepdims=True)
    f = f / np.maximum(n, 1e-12)
    up = np.array([0.0, -1.0, 0.0]) if up_hint is None else np.asarray(up_hint, dtype=np.float64)

    # degenerate when looking straight up/down: fall back to a lateral reference
    dot = np.abs(f @ up)
    fallback = np.array([0.0, 0.0, 1.0])
    up = np.where(dot[..., None] > 0.98, fallback, up)

    r = np.cross(up, f)
    r /= np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-12)
    u = np.cross(f, r)
    # columns: right, up, forward
    return np.stack([r, u, f], axis=-1)


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """(N,3,3) rotation matrices -> (N,4) quaternions in (w, x, y, z) order."""
    R = np.asarray(R, dtype=np.float64)
    n = R.shape[0]
    q = np.zeros((n, 4), dtype=np.float64)
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    tr = m00 + m11 + m22

    mask = tr > 0
    s = np.sqrt(np.maximum(tr[mask] + 1.0, 1e-12)) * 2.0
    q[mask, 0] = 0.25 * s
    q[mask, 1] = (m21[mask] - m12[mask]) / s
    q[mask, 2] = (m02[mask] - m20[mask]) / s
    q[mask, 3] = (m10[mask] - m01[mask]) / s

    m2 = (~mask) & (m00 >= m11) & (m00 >= m22)
    if m2.any():
        s2 = np.sqrt(np.maximum(1.0 + m00[m2] - m11[m2] - m22[m2], 1e-12)) * 2.0
        q[m2, 0] = (m21[m2] - m12[m2]) / s2
        q[m2, 1] = 0.25 * s2
        q[m2, 2] = (m01[m2] + m10[m2]) / s2
        q[m2, 3] = (m02[m2] + m20[m2]) / s2

    m3 = (~mask) & (~m2) & (m11 > m22)
    if m3.any():
        s3 = np.sqrt(np.maximum(1.0 + m11[m3] - m00[m3] - m22[m3], 1e-12)) * 2.0
        q[m3, 0] = (m02[m3] - m20[m3]) / s3
        q[m3, 1] = (m01[m3] + m10[m3]) / s3
        q[m3, 2] = 0.25 * s3
        q[m3, 3] = (m12[m3] + m21[m3]) / s3

    m4 = (~mask) & (~m2) & (~m3)
    if m4.any():
        s4 = np.sqrt(np.maximum(1.0 + m22[m4] - m00[m4] - m11[m4], 1e-12)) * 2.0
        q[m4, 0] = (m10[m4] - m01[m4]) / s4
        q[m4, 1] = (m02[m4] + m20[m4]) / s4
        q[m4, 2] = (m12[m4] + m21[m4]) / s4
        q[m4, 3] = 0.25 * s4

    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    return q.astype(np.float32)


def look_at_basis(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = target - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= max(np.linalg.norm(r), 1e-9)
    u = np.cross(r, f)
    return np.stack([r, u, f], axis=-1)
