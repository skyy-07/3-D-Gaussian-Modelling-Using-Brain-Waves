"""Offline director + defensive Meta API parsing (no network in tests)."""

from __future__ import annotations

import base64
import zlib

import numpy as np
import pytest

from mns.director.meta_client import (
    MetaAPIError,
    MetaModelAPI,
    _extract_json,
    decode_mask,
    parse_sam_output,
)
from mns.director.prompts import scene_graph_schema
from mns.director.scene_graph import build_scene_graph, refine_with_sam
from mns.scenarios import STUDY, get_scenario


def test_client_without_key_is_offline_and_never_calls_network():
    api = MetaModelAPI(api_key="")
    assert api.available is False
    with pytest.raises(MetaAPIError):
        api.chat_json("system", "user", scene_graph_schema())
    with pytest.raises(MetaAPIError):
        api.segment("", "mug")


def test_offline_director_produces_a_usable_graph():
    result = build_scene_graph(get_scenario("focus").narrative, api=None)
    graph = result.graph
    assert result.used_model is False
    assert result.model is None
    assert graph.source == "heuristic"
    assert 3 <= len(graph.objects) <= 8
    assert 3 <= len(graph.beats) <= 6
    assert graph.title
    assert all(0.0 <= o.bbox.u0 <= o.bbox.u1 <= 1.0 for o in graph.objects)
    assert all(f in {x.id for x in graph.objects} for b in graph.beats for f in b.focus)


def test_schema_is_strict_and_complete():
    schema = scene_graph_schema()
    assert schema["strict"] is True
    assert schema["schema"]["required"] == [
        "title",
        "style",
        "lighting",
        "palette",
        "objects",
        "beats",
    ]
    props = schema["schema"]["properties"]
    assert props["objects"]["minItems"] == 3
    assert props["beats"]["maxItems"] == 6


def test_parse_sam_boxes_from_wire_grammar():
    text = (
        "0<|box;x1=10;y1=20;x2=110;y2=120;w=512;h=512;c=0.91|>"
        "1<|box;x1=300;y1=200;x2=180;y2=100;w=512;h=512|>"
        "2<|box;x1=40;y1=60;x2=200;y2=260;w=1000;h=1000;c=0.42|>"
    )
    segments = parse_sam_output(text)
    # The reversed second box is rejected rather than silently normalised.
    assert len(segments) == 2
    first, third = segments
    assert first.object_id == 0
    assert first.bbox == pytest.approx((10 / 512, 20 / 512, 110 / 512, 120 / 512))
    assert first.confidence == pytest.approx(0.91)
    assert third.object_id == 2
    assert third.bbox == pytest.approx((0.04, 0.06, 0.2, 0.26))


def test_decode_mask_accepts_compressed_raster_and_rejects_junk():
    h, w = 8, 6
    raster = (np.arange(h * w).reshape(h, w) % 2).astype(np.uint8)
    payload = base64.a85encode(zlib.compress(raster.tobytes())).decode("ascii")
    decoded = decode_mask({"h": h, "w": w, "payload": payload})
    assert decoded is not None
    assert decoded.shape == (h, w)
    assert np.array_equal(decoded, raster)
    assert decode_mask({"h": 4, "w": 4, "payload": "not-valid-a85"}) is None
    assert decode_mask({"h": 4, "w": 4, "payload": base64.a85encode(b"short").decode()}) is None


def test_extract_json_handles_fences_and_preamble():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('Here you go: {"a": [1, 2]} — enjoy') == {"a": [1, 2]}
    with pytest.raises(MetaAPIError):
        _extract_json("no json here")


def test_refine_with_sam_degrades_when_no_segments(monkeypatch):
    graph = build_scene_graph(STUDY, api=None).graph
    api = MetaModelAPI(api_key="fake")
    # No network: return an empty segmentation for every object.
    monkeypatch.setattr(api, "segment", lambda *a, **k: [])
    refined, masks, diag = refine_with_sam(graph, api, "unused")
    assert masks == {}
    assert len(diag) == min(5, len(graph.objects))
    assert all(d["matched"] is False for d in diag)
    assert len(refined.objects) == len(graph.objects)
