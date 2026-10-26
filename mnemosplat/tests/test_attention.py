"""EEG -> attention field: shape, contrast and entity assignment."""

from __future__ import annotations

import numpy as np

from mns.director.recipes import heuristic_graph
from mns.fusion.attention import (
    assign_object_salience,
    bbox_mask,
    build_attention_field,
    default_masks,
    resize_bilinear,
)
from mns.scenarios import STUDY
from mns.schemas import EpochFeatures


def _epoch(salience: float, certainty: float = 0.5, lateralization: float = 0.0) -> EpochFeatures:
    return EpochFeatures(
        index=0,
        t0=0.0,
        t1=0.75,
        theta=8.0,
        alpha=10.0,
        beta=12.0,
        gamma=6.0,
        gamma_theta=0.75,
        lateralization=lateralization,
        certainty=certainty,
        salience=salience,
        focus_index=salience,
    )


def _salience(graph, value: float) -> dict[str, float]:
    return {o.id: value for o in graph.objects}


def test_attention_field_range_and_peak():
    graph = heuristic_graph(STUDY)
    sal = _salience(graph, 0.8)
    sal["mug"] = 0.98
    field = build_attention_field(graph, sal, _epoch(0.85), shape=(96, 96))

    assert field.map.shape == (96, 96)
    assert field.map.dtype == np.float32
    assert field.map.min() >= field.floor - 1e-6
    assert field.map.max() <= 1.0 + 1e-6

    mug = next(o for o in graph.objects if o.id == "mug")
    u, v, peak = field.peak()
    assert peak > 0.9
    assert abs(u - mug.bbox.center[0]) < 0.12
    assert abs(v - mug.bbox.center[1]) < 0.12
    assert field.object_density("mug") > field.object_density("plant")


def test_low_salience_is_flatter_than_vivid():
    graph = heuristic_graph(STUDY)
    sal = _salience(graph, 0.6)
    vivid = build_attention_field(graph, sal, _epoch(0.95), shape=(80, 80))
    diffuse = build_attention_field(graph, sal, _epoch(0.12), shape=(80, 80))

    # The peak is normalised in both, so absolute vividness has to show up as
    # contrast: a vivid field is darker on average and has a higher peak/mean.
    assert diffuse.map.mean() > vivid.map.mean()
    assert diffuse.contrast < vivid.contrast
    assert vivid.map.max() / vivid.map.mean() > diffuse.map.max() / diffuse.map.mean()


def test_object_salience_tracks_vividness_and_focus_prior():
    graph = heuristic_graph(STUDY)
    epochs = [_epoch(0.15), _epoch(0.9)]
    low, high = assign_object_salience(graph, epochs)

    assert set(low) == {o.id for o in graph.objects}
    assert np.mean(list(high.values())) > np.mean(list(low.values()))
    assert all(0.1 <= v <= 1.0 for v in low.values())
    assert all(0.1 <= v <= 1.0 for v in high.values())
    assert all(isinstance(v, float) for v in high.values())


def test_hemifield_prior_moves_the_peak():
    graph = heuristic_graph(STUDY)
    sal = _salience(graph, 0.7)
    left = build_attention_field(graph, sal, _epoch(0.7, lateralization=-0.9), shape=(64, 64))
    right = build_attention_field(graph, sal, _epoch(0.7, lateralization=0.9), shape=(64, 64))
    assert left.hemifield < 0 < right.hemifield
    assert any(b.object_id == "__hemifield__" for b in left.blobs)
    assert any(b.object_id == "__hemifield__" for b in right.blobs)


def test_masks_and_resize_helpers():
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert np.allclose(resize_bilinear(arr, (4, 4)), arr)
    assert resize_bilinear(arr, (8, 8)).shape == (8, 8)
    assert resize_bilinear(arr, (8, 8)).max() <= 15.0

    graph = heuristic_graph(STUDY)
    masks = default_masks(graph, (48, 48))
    assert set(masks) == {o.id for o in graph.objects}
    assert all(m.shape == (48, 48) and m.min() >= 0.0 and m.max() <= 1.0 for m in masks.values())

    m = bbox_mask(graph.objects[2].bbox, (32, 32), feather=2)
    assert m.shape == (32, 32) and m.max() > 0
