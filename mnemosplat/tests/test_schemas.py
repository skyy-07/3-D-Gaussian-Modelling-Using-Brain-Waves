"""Wire contracts: the manifest is the only thing the headset trusts."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from mns.schemas import (
    BBox,
    Beat,
    CameraModel,
    EpochFeatures,
    FrameManifest,
    Manifest,
    SceneGraph,
    SceneObject,
)


def _epoch() -> EpochFeatures:
    return EpochFeatures(
        index=0,
        t0=0.0,
        t1=0.75,
        theta=1.0,
        alpha=2.0,
        beta=3.0,
        gamma=4.0,
        gamma_theta=4.0,
        salience=0.6,
    )


def test_bbox_helpers_are_consistent():
    b = BBox.from_xywh(0.2, 0.3, 0.4, 0.2)
    assert (b.u0, b.v0, b.u1, b.v1) == pytest.approx((0.2, 0.3, 0.6, 0.5))
    assert b.shifted(0.1, -0.1).u0 == pytest.approx(0.3)
    assert b.shifted(0.7, 0.7).u1 == 1.0
    assert b.size == (pytest.approx(0.4), pytest.approx(0.2))
    assert b.area == pytest.approx(0.08)
    with pytest.raises(ValidationError):
        BBox(u0=float("nan"), v0=0, u1=1, v1=1)


def test_epoch_ranges_are_enforced():
    with pytest.raises(ValidationError):
        EpochFeatures(
            index=0,
            t0=0,
            t1=1,
            theta=1,
            alpha=1,
            beta=1,
            gamma=1,
            gamma_theta=1,
            salience=1.5,
        )
    with pytest.raises(ValidationError):
        EpochFeatures(
            index=0,
            t0=0,
            t1=1,
            theta=1,
            alpha=1,
            beta=1,
            gamma=1,
            gamma_theta=1,
            lateralization=2.0,
        )


def test_scene_object_ids_are_slugified():
    obj = SceneObject(
        id="  Ceramic Mug  ",
        label="mug",
        bbox=BBox(u0=0.1, v0=0.1, u1=0.2, v1=0.2),
    )
    assert obj.id == "ceramic-mug"


def test_manifest_round_trips_through_json():
    graph = SceneGraph(
        title="room",
        narrative="a room",
        objects=[
            SceneObject(id="mug", label="mug", bbox=BBox(u0=0.1, v0=0.1, u1=0.2, v1=0.2))
        ],
        beats=[Beat(index=0, t0=0.0, t1=1.0, label="one", focus=["mug"], narrative="m")],
    )
    camera = CameraModel.from_fov(512, 512)
    frame = FrameManifest(
        index=0,
        t=0.0,
        label="one",
        camera=camera,
        epoch=_epoch(),
        n_splats=120,
        object_salience={"mug": 0.8},
        objects=graph.objects,
        bounds={"min": [0, 0, 0], "max": [1, 1, 1]},
    )
    manifest = Manifest(id="abc", title="room", scene_graph=graph, frames=[frame])

    payload = json.loads(json.dumps(manifest.model_dump()))
    again = Manifest(**payload)
    assert again.frames[0].camera.fx == pytest.approx(camera.fx)
    assert again.frames[0].epoch.salience == pytest.approx(0.6)
    assert again.scene_graph.objects[0].id == "mug"
    assert again.stats == {}


def test_camera_fov_maths():
    cam = CameraModel.from_fov(512, 512, 60.0)
    assert cam.cx == pytest.approx(255.5)
    assert cam.cy == pytest.approx(255.5)
    assert cam.fx == pytest.approx(256 / np.tan(np.radians(30)))
    assert cam.fy == pytest.approx(cam.fx)
    x, y, z = cam.unproject(np.ones((512, 512), dtype=np.float32))
    assert x.shape == y.shape == z.shape == (512, 512)
