"""Camera maths, quaternion validity and the single-view lift contract."""

from __future__ import annotations

import numpy as np
import pytest

from mns.fusion.attention import AttentionField
from mns.geometry.camera import basis_facing, quat_from_matrix, unproject_pixels
from mns.geometry.gaussians import GaussianConfig, init_from_frame
from mns.schemas import CameraModel


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_unproject_project_round_trip():
    camera = CameraModel.from_fov(160, 120, hfov_deg=62.0)
    rng = np.random.default_rng(1)
    u = rng.uniform(0, 159, size=50)
    v = rng.uniform(0, 119, size=50)
    z = rng.uniform(0.5, 8.0, size=50)
    xyz = unproject_pixels(camera, u, v, z)
    u2, v2 = camera.project(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    assert np.allclose(u, u2, atol=1e-6)
    assert np.allclose(v, v2, atol=1e-6)


def test_camera_unproject_matches_grid():
    # odd dimensions put the optical axis exactly on a pixel
    camera = CameraModel.from_fov(33, 25)
    depth = np.full((25, 33), 2.5, dtype=np.float32)
    x, y, z = camera.unproject(depth)
    assert z.shape == (25, 33)
    assert np.allclose(z, 2.5)
    assert abs(x[12, 16]) < 1e-6
    assert abs(y[12, 16]) < 1e-6
    with pytest.raises(ValueError):
        camera.unproject(np.ones((10, 10), dtype=np.float32))


def test_quaternions_are_unit_and_reconstruct_rotation():
    rng = np.random.default_rng(7)
    forward = rng.normal(size=(200, 3))
    R = basis_facing(forward)
    q = quat_from_matrix(R)
    assert q.shape == (200, 4)
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-5)
    for i in range(0, 200, 37):
        assert np.allclose(_quat_to_matrix(q[i].astype(np.float64)), R[i], atol=1e-4)

    identity = quat_from_matrix(np.eye(3)[None, :, :])[0]
    assert np.allclose(np.abs(identity), [1.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_basis_facing_is_orthonormal_and_faces_forward():
    forward = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    R = basis_facing(forward)
    assert R.shape == (3, 3, 3)
    assert np.allclose(np.einsum("nij,nkj->nik", R, R), np.eye(3)[None, :, :], atol=1e-6)
    assert np.allclose(R[:, :, 2], forward, atol=1e-6)


def _field(shape=(64, 64), value=0.7, contrast=1.6) -> AttentionField:
    m = np.full(shape, value, dtype=np.float32)
    return AttentionField(map=m, blobs=[], salience={}, floor=0.1, contrast=contrast)


def test_lift_has_no_shell_when_disabled_and_respects_cap():
    h = w = 64
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[: h // 2] = (220, 40, 30)
    image[h // 2 :] = (30, 40, 220)
    depth = np.linspace(1.5, 4.0, h, dtype=np.float32)[:, None].repeat(w, axis=1)
    camera = CameraModel.from_fov(w, h)

    cfg = GaussianConfig(
        target_splats=500,
        hard_cap=10_000,
        environment_shell=False,
        seed=2,
    )
    cloud = init_from_frame(image, depth, _field(), camera, cfg=cfg, global_salience=0.8)

    assert 64 <= len(cloud) <= 500
    assert np.isfinite(cloud.xyz).all()
    assert np.all(cloud.scales > 0)
    assert np.all(cloud.scales <= cfg.max_scale_m + 1e-6)
    assert np.all((cloud.opacity >= 0) & (cloud.opacity <= 1))
    assert np.allclose(np.linalg.norm(cloud.quats, axis=1), 1.0, atol=1e-5)
    assert cloud.meta["frame"] == "threejs-y-up"


def test_lift_flips_camera_y_and_recentres_volume():
    h = w = 64
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[: h // 2] = (255, 0, 0)  # top half, camera y-down
    image[h // 2 :] = (0, 0, 255)  # bottom half
    # constant depth keeps every pixel at the same plane
    depth = np.full((h, w), 2.0, dtype=np.float32)
    camera = CameraModel.from_fov(w, h)

    cfg = GaussianConfig(
        target_splats=50_000, hard_cap=50_000, environment_shell=False, prune_alpha=0.0
    )
    cloud = init_from_frame(image, depth, _field(value=0.9), camera, cfg=cfg, global_salience=0.9)

    red = cloud.colors[:, 0] > 0.8
    blue = cloud.colors[:, 2] > 0.8
    assert red.any() and blue.any()
    # camera space is Y-down; three.js is Y-up. Red (top) must be +Y after export.
    assert cloud.xyz[red, 1].mean() > 0
    assert cloud.xyz[blue, 1].mean() < 0
    # volume is recentred; camera sits in front of the origin looking at it
    assert np.allclose(cloud.xyz.mean(axis=0), 0.0, atol=1e-5)
    assert 1.8 < cloud.camera_position[2] < 2.2


def test_environment_shell_adds_peripheral_geometry():
    h = w = 32
    image = np.full((h, w, 3), 120, dtype=np.uint8)
    depth = np.full((h, w), 2.0, dtype=np.float32)
    camera = CameraModel.from_fov(w, h)
    base = dict(target_splats=2_000, hard_cap=5_000, shell_alpha=0.14, seed=3)

    without = init_from_frame(
        image, depth, _field((h, w)), camera, cfg=GaussianConfig(**base, environment_shell=False)
    )
    with_shell = init_from_frame(
        image, depth, _field((h, w)), camera, cfg=GaussianConfig(**base, environment_shell=True)
    )
    assert len(with_shell) > len(without)
    # The shell is intentionally faint but never invisible.
    assert np.isclose(with_shell.opacity.max(), 1.0) or with_shell.opacity.min() <= 0.14 + 1e-6
