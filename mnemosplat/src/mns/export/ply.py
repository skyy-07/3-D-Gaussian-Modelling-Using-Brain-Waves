"""Exporters: standard 3DGS ``.ply`` and antimatter15 ``.splat``.

Both formats are written from the same array set, so the WebXR client can load the
8 MB ``.splat`` on a Quest while researchers open the ``.ply`` in gsplat/Blender.

Activation conventions (INRIA 3DGS):
    colour     f_dc = (rgb - 0.5) / C0,  C0 = 0.28209479177387814
    opacity    stored as logit(alpha)
    scale      stored as log(scale)
    rotation   unit quaternion, (w, x, y, z)
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ..geometry.gaussians import GaussianCloud

C0 = 0.28209479177387814
SH_C0 = C0


def _logit(a: np.ndarray) -> np.ndarray:
    a = np.clip(a.astype(np.float64), 1e-4, 1 - 1e-4)
    return np.log(a / (1 - a))


def _norm_quat(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    out = q / np.maximum(n, 1e-12)
    out[n[:, 0] < 1e-9] = [1.0, 0.0, 0.0, 0.0]
    return out


def ply_dtype() -> np.dtype:
    fields = [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ]
    return np.dtype(fields)


def ply_header(n: int) -> str:
    props = [
        "x", "y", "z",
        "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment MnemoSplat cognitive gaussian volume",
        "comment frame: three.js (Y up, -Z forward); units: metres",
        f"element vertex {n}",
    ]
    lines += [f"property float {p}" for p in props]
    lines.append("end_header")
    return "\n".join(lines) + "\n"


def write_ply(path: str | Path, cloud: GaussianCloud) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(cloud)
    rec = np.zeros(n, dtype=ply_dtype())
    rec["x"], rec["y"], rec["z"] = cloud.xyz[:, 0], cloud.xyz[:, 1], cloud.xyz[:, 2]
    rgb = np.clip(cloud.colors, 0.0, 1.0)
    rec["f_dc_0"] = (rgb[:, 0] - 0.5) / SH_C0
    rec["f_dc_1"] = (rgb[:, 1] - 0.5) / SH_C0
    rec["f_dc_2"] = (rgb[:, 2] - 0.5) / SH_C0
    rec["opacity"] = _logit(cloud.opacity)
    rec["scale_0"], rec["scale_1"], rec["scale_2"] = (
        np.log(np.maximum(cloud.scales[:, 0], 1e-8)),
        np.log(np.maximum(cloud.scales[:, 1], 1e-8)),
        np.log(np.maximum(cloud.scales[:, 2], 1e-8)),
    )
    q = _norm_quat(cloud.quats)
    rec["rot_0"], rec["rot_1"], rec["rot_2"], rec["rot_3"] = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    with open(path, "wb") as fh:
        fh.write(ply_header(n).encode("ascii"))
        fh.write(rec.tobytes())
    return path


def write_splat(path: str | Path, cloud: GaussianCloud) -> Path:
    """32 bytes per splat: pos(3f) scale(3f) rgba(4u8) quat(4u8)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(cloud)
    buf = np.zeros(n, dtype=np.dtype([
        ("pos", "<f4", 3),
        ("scale", "<f4", 3),
        ("rgba", "u1", 4),
        ("rot", "u1", 4),
    ]))
    buf["pos"] = cloud.xyz.astype(np.float32)
    buf["scale"] = cloud.scales.astype(np.float32)
    rgba = np.empty((n, 4), dtype=np.uint8)
    rgba[:, :3] = (np.clip(cloud.colors, 0, 1) * 255).astype(np.uint8)
    rgba[:, 3] = (np.clip(cloud.opacity, 0, 1) * 255).astype(np.uint8)
    buf["rgba"] = rgba
    q = _norm_quat(cloud.quats)
    # round-to-nearest, not truncate: the 8-bit quaternion is the only rotation
    # precision the headset gets, and truncation biases every component downward.
    buf["rot"] = np.clip(np.rint(q * 128.0 + 128.0), 0, 255).astype(np.uint8)
    path.write_bytes(buf.tobytes())
    return path


def read_splat(path: str | Path) -> dict[str, np.ndarray]:
    """Inverse of :func:`write_splat`; used by tests and the manifest validator."""
    raw = Path(path).read_bytes()
    if len(raw) % 32 != 0:
        raise ValueError(f"{path} is not a multiple of 32 bytes")
    dt = np.dtype([
        ("pos", "<f4", 3),
        ("scale", "<f4", 3),
        ("rgba", "u1", 4),
        ("rot", "u1", 4),
    ])
    buf = np.frombuffer(raw, dtype=dt)
    return {
        "xyz": np.asarray(buf["pos"], dtype=np.float32).reshape(-1, 3),
        "scales": np.asarray(buf["scale"], dtype=np.float32).reshape(-1, 3),
        "colors": np.asarray(buf["rgba"], dtype=np.float32)[:, :3] / 255.0,
        "opacity": np.asarray(buf["rgba"], dtype=np.float32)[:, 3] / 255.0,
        "quats": (np.asarray(buf["rot"], dtype=np.float32) - 128.0) / 128.0,
    }


def apply_activations(rec: np.ndarray) -> dict[str, np.ndarray]:
    """Decode a 3DGS ply record back to physical values (for QA / tests)."""
    return {
        "xyz": np.stack([rec["x"], rec["y"], rec["z"]], axis=-1),
        "colors": np.stack([rec["f_dc_0"], rec["f_dc_1"], rec["f_dc_2"]], axis=-1) * SH_C0 + 0.5,
        "opacity": 1.0 / (1.0 + np.exp(-rec["opacity"])),
        "scales": np.exp(np.stack([rec["scale_0"], rec["scale_1"], rec["scale_2"]], axis=-1)),
        "quats": np.stack([rec["rot_0"], rec["rot_1"], rec["rot_2"], rec["rot_3"]], axis=-1),
    }


def read_ply(path: str | Path) -> tuple[dict, np.ndarray]:
    """Minimal reader for our own writer (header + payload)."""
    data = Path(path).read_bytes()
    marker = b"end_header\n"
    idx = data.find(marker)
    if idx < 0:
        raise ValueError("not a binary ply")
    header = data[:idx].decode("ascii")
    n = int([ln for ln in header.splitlines() if ln.startswith("element vertex")][0].split()[-1])
    payload = np.frombuffer(data[idx + len(marker) :], dtype=ply_dtype(), count=n)
    return {"header": header, "n": n, "bytes": len(data)}, payload


def export_cloud(cloud: GaussianCloud, out_dir: str | Path, stem: str = "frame") -> dict:
    out = Path(out_dir)
    ply_path = write_ply(out / f"{stem}.ply", cloud)
    splat_path = write_splat(out / f"{stem}.splat", cloud)
    return {
        "ply": ply_path.name,
        "splat": splat_path.name,
        "ply_bytes": ply_path.stat().st_size,
        "splat_bytes": splat_path.stat().st_size,
        "n_splats": len(cloud),
        "bounds": cloud.bounds,
        "camera_position": cloud.camera_position,
    }


def splat_payload_prefix(n: int) -> bytes:
    """Tiny helper for streaming headers / tests."""
    return struct.pack("<I", n)
