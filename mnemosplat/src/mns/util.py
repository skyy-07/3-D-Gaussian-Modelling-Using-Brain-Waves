"""Small shared helpers: hashing, IO, clamping, logging."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("mns")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def slugify(value: object, fallback: str = "object") -> str:
    s = str(value).strip().lower().replace(" ", "-")
    out = "".join(ch for ch in s if ch.isalnum() or ch in "-_")[:48]
    return out or fallback


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def stable_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_dir(parent: Path, prefix: str = "tmp-") -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))


def robust_unit_scale(x: np.ndarray, lo_pct: float = 5.0, hi_pct: float = 95.0) -> np.ndarray:
    """Percentile-normalise to [0, 1]; flat input maps to 0.5."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.percentile(x, [lo_pct, hi_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        return np.full_like(x, 0.5)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)
