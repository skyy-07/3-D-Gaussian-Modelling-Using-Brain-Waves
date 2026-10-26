"""Round-trips through the two on-disk formats the demo ships."""

from __future__ import annotations

import numpy as np

from mns.export.ply import (
    apply_activations,
    export_cloud,
    read_ply,
    read_splat,
    write_ply,
    write_splat,
)
from mns.geometry.gaussians import GaussianCloud


def _cloud(n: int = 37, seed: int = 4) -> GaussianCloud:
    rng = np.random.default_rng(seed)
    xyz = rng.normal(size=(n, 3)).astype(np.float32)
    scales = rng.uniform(0.005, 0.08, size=(n, 3)).astype(np.float32)
    quats = rng.normal(size=(n, 4)).astype(np.float32)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    colors = rng.uniform(0.05, 0.95, size=(n, 3)).astype(np.float32)
    opacity = rng.uniform(0.06, 0.96, size=n).astype(np.float32)
    return GaussianCloud(
        xyz=xyz,
        scales=scales,
        quats=quats,
        colors=colors,
        opacity=opacity,
        camera_position=[0.1, -0.2, 3.4],
        bounds={"min": xyz.min(axis=0).tolist(), "max": xyz.max(axis=0).tolist()},
    )


def test_splat_round_trip(tmp_path):
    cloud = _cloud()
    path = write_splat(tmp_path / "frame.splat", cloud)
    assert path.stat().st_size == len(cloud) * 32

    data = read_splat(path)
    assert data["xyz"].shape == (len(cloud), 3)
    assert np.allclose(data["xyz"], cloud.xyz, atol=1e-6)
    assert np.allclose(data["scales"], cloud.scales, atol=1e-6)
    assert np.allclose(data["colors"], cloud.colors, atol=1.0 / 255.0 + 1e-6)
    assert np.allclose(data["opacity"], cloud.opacity, atol=1.0 / 255.0 + 1e-6)
    # 8-bit quaternions: compare direction, not component signs or norm.
    a = data["quats"] / np.linalg.norm(data["quats"], axis=1, keepdims=True)
    b = cloud.quats / np.linalg.norm(cloud.quats, axis=1, keepdims=True)
    dots = np.abs(np.sum(a * b, axis=1))
    assert np.all(dots > 0.999)


def test_ply_round_trip_and_activations(tmp_path):
    cloud = _cloud()
    path = write_ply(tmp_path / "frame.ply", cloud)
    meta, rec = read_ply(path)
    assert meta["n"] == len(cloud)
    assert rec.dtype.names and {"x", "f_dc_0", "opacity", "rot_3"} <= set(rec.dtype.names)

    decoded = apply_activations(rec)
    assert np.allclose(decoded["xyz"], cloud.xyz, atol=1e-6)
    assert np.allclose(decoded["colors"], cloud.colors, atol=1e-4)
    assert np.allclose(decoded["opacity"], cloud.opacity, atol=1e-3)
    assert np.allclose(decoded["scales"], cloud.scales, rtol=1e-4)
    assert np.allclose(np.linalg.norm(decoded["quats"], axis=1), 1.0, atol=1e-5)


def test_export_cloud_writes_both_and_reports_bounds(tmp_path):
    cloud = _cloud()
    report = export_cloud(cloud, tmp_path / "frames", stem="000")
    assert (tmp_path / "frames" / "000.ply").exists()
    assert (tmp_path / "frames" / "000.splat").exists()
    assert report["n_splats"] == len(cloud)
    assert report["splat_bytes"] == len(cloud) * 32
    assert set(report["bounds"]) == {"min", "max"}
    assert report["camera_position"] == cloud.camera_position


def test_read_splat_rejects_misaligned_file(tmp_path):
    path = tmp_path / "broken.splat"
    path.write_bytes(b"not-a-splat")
    try:
        read_splat(path)
    except ValueError as exc:
        assert "multiple of 32" in str(exc)
    else:  # pragma: no cover - a regression would be silent otherwise
        raise AssertionError("read_splat accepted a truncated file")
