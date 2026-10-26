"""Typed contracts shared by the pipeline, the HTTP API and the WebXR client.

Every number that ends up on the headset travels through one of these models, so
this file is the single source of truth for the wire format (``manifest.json``).
"""

from __future__ import annotations

import math
import time
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .util import clamp, slugify


class BBox(BaseModel):
    """Normalised screen bounds, origin top-left, ``0 <= u,v <= 1``."""

    model_config = ConfigDict(extra="ignore")

    u0: float
    v0: float
    u1: float
    v1: float

    @field_validator("u0", "u1", "v0", "v1", mode="before")
    @classmethod
    def _finite(cls, v: float) -> float:
        v = float(v)
        if not math.isfinite(v):
            raise ValueError("bbox coordinate must be finite")
        return v

    def ordered(self) -> BBox:
        return BBox(
            u0=clamp(min(self.u0, self.u1), 0.0, 1.0),
            v0=clamp(min(self.v0, self.v1), 0.0, 1.0),
            u1=clamp(max(self.u0, self.u1), 0.0, 1.0),
            v1=clamp(max(self.v0, self.v1), 0.0, 1.0),
        )

    @classmethod
    def from_xywh(cls, u: float, v: float, w: float, h: float) -> BBox:
        return cls(u0=u, v0=v, u1=u + w, v1=v + h).ordered()

    @classmethod
    def from_pixels(cls, x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> BBox:
        return cls(u0=x1 / w, v0=y1 / h, u1=x2 / w, v1=y2 / h).ordered()

    @property
    def center(self) -> tuple[float, float]:
        b = self.ordered()
        return (0.5 * (b.u0 + b.u1), 0.5 * (b.v0 + b.v1))

    @property
    def size(self) -> tuple[float, float]:
        b = self.ordered()
        return (max(b.u1 - b.u0, 1e-3), max(b.v1 - b.v0, 1e-3))

    @property
    def area(self) -> float:
        w, h = self.size
        return w * h

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        b = self.ordered()
        x1 = int(round(b.u0 * width))
        y1 = int(round(b.v0 * height))
        x2 = int(round(b.u1 * width))
        y2 = int(round(b.v1 * height))
        return (
            max(0, min(x1, width - 1)),
            max(0, min(y1, height - 1)),
            max(1, min(x2, width)),
            max(1, min(y2, height)),
        )

    def shifted(self, du: float, dv: float) -> BBox:
        b = self.ordered()
        return BBox(u0=b.u0 + du, v0=b.v0 + dv, u1=b.u1 + du, v1=b.v1 + dv).ordered()


class SceneObject(BaseModel):
    """One semantic entity from the episodic scene graph."""

    model_config = ConfigDict(extra="ignore")

    id: str
    label: str
    bbox: BBox
    depth_hint: float = Field(0.5, ge=0.0, le=1.0, description="0 = near, 1 = far")
    base_salience: float = Field(0.55, ge=0.0, le=1.0)
    prompt: str = ""
    color: str = "#8a8f98"
    confidence: float = Field(1.0, ge=0.0, le=1.0)

    @field_validator("id", mode="before")
    @classmethod
    def _slug(cls, v: object) -> str:
        return slugify(v)


class Beat(BaseModel):
    """A narrative time slice. One beat -> one EEG epoch -> one splat frame."""

    model_config = ConfigDict(extra="ignore")

    index: int
    t0: float
    t1: float
    label: str
    focus: list[str] = Field(default_factory=list)
    narrative: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.t1 - self.t0)


class SceneGraph(BaseModel):
    """Output of the cognitive director (Muse Spark, or the offline fallback)."""

    model_config = ConfigDict(extra="ignore")

    title: str
    narrative: str
    style: str = "photorealistic memory, 35mm, shallow depth of field"
    lighting: str = "soft late-afternoon window light"
    palette: list[str] = Field(default_factory=list)
    objects: list[SceneObject] = Field(default_factory=list)
    beats: list[Beat] = Field(default_factory=list)
    source: str = "heuristic"
    model: str | None = None
    raw: dict | None = None

    def object_map(self) -> dict[str, SceneObject]:
        return {o.id: o for o in self.objects}


class EpochFeatures(BaseModel):
    """Per-window EEG telemetry. All ratios are dimensionless and unit-free."""

    index: int
    t0: float
    t1: float
    theta: float
    alpha: float
    beta: float
    gamma: float
    gamma_theta: float
    lateralization: float = Field(0.0, ge=-1.0, le=1.0)
    certainty: float = Field(0.5, ge=0.0, le=1.0)
    salience: float = Field(0.5, ge=0.1, le=1.0)
    focus_index: float = Field(0.5, ge=0.0, le=1.0)
    channels_used: list[str] = Field(default_factory=list)


class SalienceFrame(BaseModel):
    t: float
    epoch: EpochFeatures
    object_salience: dict[str, float] = Field(default_factory=dict)


class CameraModel(BaseModel):
    width: int
    height: int
    hfov_deg: float = 60.0
    fx: float
    fy: float
    cx: float
    cy: float
    #: World placement of the virtual camera in the exported splat frame.
    position: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    look_at: list[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0])

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float = 60.0) -> CameraModel:
        hfov = math.radians(hfov_deg)
        fx = (width * 0.5) / math.tan(hfov * 0.5)
        return cls(
            width=width,
            height=height,
            hfov_deg=hfov_deg,
            fx=fx,
            fy=fx,  # square pixels
            cx=(width - 1) * 0.5,
            cy=(height - 1) * 0.5,
        )

    def unproject(self, depth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pixel grid + metric depth -> camera-space X, Y (down), Z (forward)."""
        h, w = depth.shape
        if (h, w) != (self.height, self.width):
            raise ValueError(f"depth shape {depth.shape} != camera {(self.height, self.width)}")
        v, u = np.mgrid[0:h, 0:w].astype(np.float64)
        z = depth.astype(np.float64)
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return x, y, z

    def project(self, x, y, z) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        return self.fx * x / z + self.cx, self.fy * y / z + self.cy


class FrameManifest(BaseModel):
    index: int
    t: float
    label: str
    narrative: str = ""
    splat: str | None = None
    ply: str | None = None
    keyframe: str | None = None
    attention: str | None = None
    depth_preview: str | None = None
    camera: CameraModel
    epoch: EpochFeatures
    n_splats: int = 0
    object_salience: dict[str, float] = Field(default_factory=dict)
    objects: list[SceneObject] = Field(default_factory=list)
    depth_source: str = "unknown"
    bounds: dict[str, list[float]] = Field(default_factory=dict)


class Manifest(BaseModel):
    id: str
    title: str
    subject: str = "synthetic-subject"
    scenario: str = "focus"
    created: float = Field(default_factory=time.time)
    scene_graph: SceneGraph
    frames: list[FrameManifest] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    stats: dict[str, float | int | str] = Field(default_factory=dict)
    objects: list[dict] = Field(default_factory=list)
    upstream: dict[str, str] = Field(default_factory=dict)
    world_scale_m: float = 1.0


Scenario = Literal["focus", "distracted", "neutral"]
