"""Experimental BrainVis reconstruction client (GPU worker is mocked)."""

from __future__ import annotations

import base64
import io
import json

import httpx
import numpy as np
import pytest
from PIL import Image

from mns.reconstruct import BrainVisClient, ReconstructError


def _png_bytes() -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (16, 16), (220, 150, 60)).save(out, format="PNG")
    return out.getvalue()


def test_unconfigured_client_refuses_without_a_url():
    client = BrainVisClient(url="")
    status = client.status()
    assert status.configured is False
    assert status.ready is False
    assert "MNS_BRAINVIS_WORKER_URL" in status.reason
    with pytest.raises(ReconstructError, match="not configured"):
        client.reconstruct(np.zeros((128, 110), dtype=np.float32), 250.0, tuple(range(128)))


def test_status_reports_worker_truth():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        assert request.headers["Authorization"] == "Bearer secret"
        return httpx.Response(
            200,
            json={
                "ready": True,
                "backend": "brainvis",
                "cuda": True,
                "checkpoint": True,
                "stable_diffusion": True,
                "version": "0.1",
            },
        )

    client = BrainVisClient(
        "http://gpu.local:8000",
        token="secret",
        transport=httpx.MockTransport(handler),
    )
    status = client.status()
    assert status.ready is True
    assert status.cuda is True
    assert status.checkpoint is True
    assert status.worker_version == "0.1"


def test_reconstruct_decodes_png_frames_and_metadata():
    png = _png_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/reconstruct"
        assert request.headers["Content-Type"].startswith("multipart/form-data")
        return httpx.Response(
            200,
            json={
                "frames": [
                    {
                        "index": 0,
                        "png_b64": base64.b64encode(png).decode("ascii"),
                        "predicted_class": 5,
                        "label": "coffee mug",
                        "prompt": "coffee mug",
                    }
                ],
                "meta": {"engine": "brainvis", "window_s": 0.44},
            },
        )

    client = BrainVisClient(
        "http://gpu.local:8000",
        transport=httpx.MockTransport(handler),
    )
    result = client.reconstruct(
        np.zeros((1, 128, 110), dtype=np.float32),
        250.0,
        tuple(f"ch{i}" for i in range(128)),
        seed=7,
    )
    assert len(result.frames) == 1
    assert result.frames[0].label == "coffee mug"
    assert result.frames[0].png == png
    assert result.meta["window_s"] == 0.44


def test_worker_error_is_surfaced():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "checkpoint missing"})

    client = BrainVisClient(
        "http://gpu.local:8000",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ReconstructError, match="checkpoint missing"):
        client.reconstruct(np.zeros((128, 110), dtype=np.float32), 250.0, ["ch"] * 128)


def test_malformed_frame_is_rejected():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"frames": [{"index": 0, "png_b64": base64.b64encode(b"not-png").decode()}]},
        )

    client = BrainVisClient(
        "http://gpu.local:8000",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ReconstructError, match="invalid PNG"):
        client.reconstruct(np.zeros((128, 110), dtype=np.float32), 250.0, ["ch"] * 128)


def test_request_payload_contains_channel_metadata():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode("latin-1", errors="ignore")
        seen["body"] = body
        return httpx.Response(
            200,
            json={
                "frames": [
                    {
                        "index": 0,
                        "png_b64": base64.b64encode(_png_bytes()).decode("ascii"),
                    }
                ]
            },
        )

    client = BrainVisClient(
        "http://gpu.local:8000",
        transport=httpx.MockTransport(handler),
    )
    client.reconstruct(
        np.zeros((128, 110), dtype=np.float32),
        250.0,
        ["O1"] * 128,
    )
    assert "channels" in seen["body"]
    assert json.dumps(["O1"] * 128) in seen["body"]


def test_mirror_status_and_pipeline_are_decoded():
    png = _png_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "healthy",
                    "service": "BrainVis EEG & Image Reconstruction API",
                    "version": "1.0.0",
                    "device": "cpu",
                },
            )
        assert request.url.path == "/api/pipeline"
        payload = json.loads(request.content)
        assert len(payload["eeg_data"]) == 128
        assert len(payload["eeg_data"][0]) == 440
        return httpx.Response(
            200,
            json={
                "analysis": {
                    "predicted_stimulus": {
                        "class_index": 8,
                        "label": "panda",
                        "category": "animal",
                    }
                },
                "reconstruction": {
                    "reconstructed_image_base64": (
                        "data:image/png;base64,"
                        + base64.b64encode(png).decode("ascii")
                    ),
                    "engine": "built-in-neural-synth",
                    "prompt": "reconstructed panda (animal)",
                },
            },
        )

    from mns.reconstruct import BrainVisMirrorClient

    client = BrainVisMirrorClient(
        "http://mirror.local",
        transport=httpx.MockTransport(handler),
    )
    status = client.status()
    assert status.ready is True
    assert status.backend == "brainvis-mirror"
    assert status.checkpoint is None

    result = client.reconstruct(
        np.zeros((1, 128, 110), dtype=np.float32),
        250.0,
        [f"ch{i}" for i in range(128)],
    )
    assert len(result.frames) == 1
    assert result.frames[0].label == "panda"
    assert result.frames[0].predicted_class == 8
    assert result.frames[0].png == png
    assert result.meta["provider"] == "brainvis-mirror"
    assert "not the trained" in result.meta["note"]


def test_provider_factory_selects_mirror(monkeypatch):
    monkeypatch.setenv("MNS_RECON_PROVIDER", "mirror")
    monkeypatch.setenv("MNS_BRAINVIS_WORKER_URL", "http://mirror.local")
    from mns.reconstruct import make_reconstruction_client

    client = make_reconstruction_client()
    assert client.provider == "mirror"
    assert client.url == "http://mirror.local"


def test_scripted_fixture_returns_watermarked_frames():
    from mns.reconstruct import ScriptedFixtureClient

    client = ScriptedFixtureClient()
    status = client.status()
    assert status.configured is True
    assert status.ready is True
    assert status.backend == "scripted-demo-fixture"
    assert status.checkpoint is False

    result = client.reconstruct(
        np.zeros((2, 128, 110), dtype=np.float32),
        250.0,
        [f"ch{i}" for i in range(128)],
    )
    assert len(result.frames) == 2
    assert result.frames[0].label == "running shoe"
    assert result.frames[0].predicted_class == 13
    assert result.meta["kind"] == "scripted_demo_fixture"
    assert result.meta["scripted"] is True
    assert "not EEG reconstruction" in result.meta["note"]
    with Image.open(io.BytesIO(result.frames[0].png)) as image:
        assert image.size == (512, 512)
        assert image.format == "PNG"


def test_provider_factory_selects_fixture(monkeypatch):
    monkeypatch.setenv("MNS_RECON_PROVIDER", "fixture")
    from mns.reconstruct import ScriptedFixtureClient, make_reconstruction_client

    client = make_reconstruction_client()
    assert isinstance(client, ScriptedFixtureClient)
    assert client.provider == "fixture"
