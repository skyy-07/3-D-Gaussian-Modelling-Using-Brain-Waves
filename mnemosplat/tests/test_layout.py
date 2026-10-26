"""Scene-graph and bbox sanitation: every graph passes through this gate."""

from __future__ import annotations

import numpy as np
import pytest

from mns.director.layout import MIN_SIDE, beat_index_for_time, sanitize_graph
from mns.director.recipes import GENERIC, heuristic_graph, pick_recipe
from mns.scenarios import STREET, STUDY
from mns.schemas import BBox, Beat, SceneGraph, SceneObject


def _graph(*objects: SceneObject, beats: list[Beat] | None = None) -> SceneGraph:
    return SceneGraph(
        title="test",
        narrative="test",
        objects=list(objects),
        beats=beats or [],
    )


def _obj(oid: str, bbox: tuple[float, float, float, float], depth: float = 0.5) -> SceneObject:
    return SceneObject(
        id=oid,
        label=oid,
        bbox=BBox(u0=bbox[0], v0=bbox[1], u1=bbox[2], v1=bbox[3]),
        depth_hint=depth,
    )


def test_bbox_ordering_and_pixels():
    b = BBox(u0=0.8, v0=0.9, u1=0.2, v1=0.1).ordered()
    assert (b.u0, b.v0, b.u1, b.v1) == (0.2, 0.1, 0.8, 0.9)
    assert b.center == (0.5, 0.5)
    assert b.area > 0
    x1, y1, x2, y2 = b.to_pixels(100, 100)
    assert (x1, y1, x2, y2) == (20, 10, 80, 90)
    assert BBox.from_xywh(0.1, 0.1, 0.2, 0.3).u1 == pytest.approx(0.3)
    assert BBox.from_pixels(10, 20, 110, 220, 200, 400).u0 == pytest.approx(0.05)


def test_out_of_frame_boxes_are_clamped_to_minimum_side():
    graph = _graph(
        _obj("outside", (-0.4, 0.95, 0.05, 1.4), depth=0.4),
        _obj("tiny", (0.5, 0.5, 0.5001, 0.5001), depth=0.4),
    )
    clean, report = sanitize_graph(graph, 512, 512)
    assert report.resized >= 2
    for obj in clean.objects:
        b = obj.bbox
        assert 0.0 <= b.u0 <= b.u1 <= 1.0
        assert 0.0 <= b.v0 <= b.v1 <= 1.0
        assert b.u1 - b.u0 >= MIN_SIDE - 1e-6
        assert b.v1 - b.v0 >= MIN_SIDE - 1e-6


def test_duplicate_ids_are_dropped():
    graph = _graph(
        _obj("mug", (0.2, 0.2, 0.3, 0.3)),
        _obj("mug", (0.6, 0.6, 0.8, 0.8)),
    )
    clean, report = sanitize_graph(graph)
    assert len(clean.objects) == 1
    assert report.dropped == ["mug"]


def test_beats_are_normalised_to_a_contiguous_timeline():
    graph = _graph(
        _obj("a", (0.1, 0.1, 0.2, 0.2)),
        beats=[
            Beat(index=3, t0=2.0, t1=4.0, label="one", focus=["a", "ghost"]),
            Beat(index=4, t0=4.0, t1=6.0, label="two", focus=["a"]),
        ],
    )
    clean, _ = sanitize_graph(graph)
    assert [b.index for b in clean.beats] == [0, 1]
    assert clean.beats[0].t0 == 0.0
    assert abs(clean.beats[-1].t1 - 1.0) < 1e-6
    assert clean.beats[0].focus == ["a"]
    assert beat_index_for_time(clean, 0.1) == 0
    assert beat_index_for_time(clean, 0.99) == 1


def test_bottom_clutter_cannot_be_a_far_wall():
    graph = _graph(_obj("box", (0.1, 0.8, 0.2, 0.95), depth=0.95))
    clean, _ = sanitize_graph(graph)
    assert clean.objects[0].depth_hint <= 0.55 + 1e-9


def test_default_palette_and_empty_beats_are_repaired():
    graph = _graph(_obj("a", (0.1, 0.1, 0.3, 0.3)))
    clean, _ = sanitize_graph(graph)
    assert len(clean.palette) == 5
    assert len(clean.beats) == 1
    assert clean.beats[0].focus


def test_recipe_matching():
    assert pick_recipe(STUDY).key == "study"
    assert pick_recipe(STREET).key == "street"
    assert pick_recipe("completely unrelated words").key == GENERIC.key
    graph = heuristic_graph(STUDY)
    assert graph.source == "heuristic"
    assert len(graph.objects) >= 5


def test_sanitizer_never_mutates_left_and_right_edges():
    graph = heuristic_graph(STUDY)
    clean, _ = sanitize_graph(graph)
    for obj in clean.objects:
        assert np.isfinite([obj.bbox.u0, obj.bbox.v0, obj.bbox.u1, obj.bbox.v1]).all()
