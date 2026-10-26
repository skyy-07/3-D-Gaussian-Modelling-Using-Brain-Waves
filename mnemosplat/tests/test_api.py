"""HTTP surface: health, isolation, public guards, and recorded-EEG uploads."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

import mns.api as api
from mns.director.meta_client import MetaModelAPI
from mns.pipeline import PipelineConfig, run_episode


class _FakeRequest:
    def __init__(self, token: str | None = None) -> None:
        self.headers = {} if token is None else {"x-mns-token": token}


def _csv(seconds: float = 4.0, sfreq: float = 250.0, channels: int = 4) -> bytes:
    t = np.arange(int(round(seconds * sfreq))) / sfreq
    names = ["O1", "O2", "Oz", "P3", "P4", "Pz"][:channels]
    rows = ["time," + ",".join(names)]
    for stamp in t:
        values = [
            8.0 * np.sin(2 * np.pi * (9.5 + c) * stamp)
            + 4.0 * np.sin(2 * np.pi * (35.0 + c) * stamp)
            for c in range(channels)
        ]
        rows.append(",".join([f"{stamp:.6f}", *[f"{value:.6f}" for value in values]]))
    return "\n".join(rows).encode("utf-8")


@pytest.fixture()
def eeg_env(tmp_path, monkeypatch):
    """A tiny baked base episode plus an inline EEG worker."""
    out = tmp_path / "out"
    base = out / "focus"
    cfg = PipelineConfig(
        width=128,
        height=128,
        target_splats=3_000,
        use_meta=False,
        use_muse_image=False,
        use_sam=False,
        preview=False,
        turntable=False,
        force=True,
        write_ply_frames=0,
    )
    run_episode(base, cfg, api=MetaModelAPI(api_key=""))

    class InlineExecutor:
        def submit(self, fn, *args, **kwargs):
            fn(*args, **kwargs)

    monkeypatch.setattr(api, "OUT", out)
    monkeypatch.setattr(api, "_eeg_jobs", {})
    monkeypatch.setattr(api, "_eeg_rate", {})
    monkeypatch.setattr(api, "_eeg_executor", InlineExecutor())
    monkeypatch.setattr(api, "EEG_TARGET_SPLATS", 3_000)
    monkeypatch.setattr(api, "EEG_MAX_ACTIVE", 2)
    monkeypatch.setattr(api, "EEG_RATE_MAX", 50)
    monkeypatch.setattr(api, "EEG_RATE_WINDOW_S", 60)
    monkeypatch.setattr(api, "PUBLIC_DEMO", True)
    monkeypatch.setattr(api, "ADMIN_TOKEN", "")
    return out


def test_health_reports_deployment_state():
    with TestClient(api.app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert "public_demo" in body
    assert "generation_enabled" in body
    assert body["models"]["text"]


def test_isolation_headers_are_present():
    with TestClient(api.app) as client:
        response = client.get("/api/health")
    assert response.headers["cross-origin-opener-policy"] == "same-origin"
    assert response.headers["cross-origin-embedder-policy"] == "require-corp"
    assert response.headers["x-content-type-options"] == "nosniff"


class _IPRequest:
    def __init__(self, forwarded: str | None) -> None:
        self.headers = {} if forwarded is None else {"x-forwarded-for": forwarded}
        self.client = None


def test_client_ip_uses_the_proxy_appended_entry():
    # Render appends the real peer; the leftmost value is client-controlled.
    assert api._client_ip(_IPRequest("1.2.3.4, 5.6.7.8")) == "5.6.7.8"
    assert api._client_ip(_IPRequest("1.2.3.4, ")) == "1.2.3.4"
    assert api._client_ip(_IPRequest(None)) == "unknown"


def test_public_demo_blocks_generation(monkeypatch):
    monkeypatch.setattr(api, "PUBLIC_DEMO", True)
    monkeypatch.setattr(api, "ADMIN_TOKEN", "")
    with TestClient(api.app) as client:
        response = client.post("/api/episode", json={})
    assert response.status_code == 403
    assert "disabled" in response.json()["detail"]


def test_public_demo_blocks_before_body_validation(monkeypatch):
    monkeypatch.setattr(api, "PUBLIC_DEMO", True)
    monkeypatch.setattr(api, "ADMIN_TOKEN", "")
    with TestClient(api.app) as client:
        # No multipart body: the early guard must win over a 422 validation error.
        assert client.post("/api/segment").status_code == 403
        assert client.post("/api/transcribe").status_code == 403


def test_admin_token_is_required_when_configured(monkeypatch):
    monkeypatch.setattr(api, "PUBLIC_DEMO", False)
    monkeypatch.setattr(api, "ADMIN_TOKEN", "s3cret")
    with pytest.raises(HTTPException) as missing:
        api._require_write_access(_FakeRequest())
    assert missing.value.status_code == 403
    with pytest.raises(HTTPException):
        api._require_write_access(_FakeRequest("wrong"))
    # The correct token passes; no request is made here.
    api._require_write_access(_FakeRequest("s3cret"))


def test_local_mode_allows_generation(monkeypatch):
    monkeypatch.setattr(api, "PUBLIC_DEMO", False)
    monkeypatch.setattr(api, "ADMIN_TOKEN", "")
    api._require_write_access(_FakeRequest())


def test_public_demo_accepts_recorded_eeg_and_preserves_image_inputs(eeg_env):
    with TestClient(api.app) as client:
        # Paid generation stays blocked on the same public deployment.
        assert client.post("/api/episode", json={}).status_code == 403

        response = client.post(
            "/api/eeg",
            files={"file": ("memory.csv", _csv(), "text/csv")},
            data={"sfreq": "250"},
        )
        assert response.status_code == 202, response.text
        payload = response.json()
        assert payload["key"].startswith("eeg-")

        job = client.get(f"/api/eeg/jobs/{payload['job']}").json()
        assert job["state"] == "done", job
        assert job["n_epochs"] >= 3
        assert job["n_beats"] == 4
        assert job["scenario"]["recorded"] is True

    manifest_path = eeg_env / job["manifest"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["scenario"] == "recorded"
    assert manifest["upstream"]["eeg"] == "file"
    assert manifest["upstream"]["keyframe"] == "shared between conditions"
    assert manifest["upstream"]["depth"] == "shared"
    assert manifest["stats"]["eeg_epochs"] >= 3
    assert len(manifest["frames"]) == 4

    uploaded = manifest_path.parent
    base = eeg_env / "focus"
    assert np.array_equal(
        np.asarray(Image.open(base / "keyframe.png").convert("RGB")),
        np.asarray(Image.open(uploaded / "keyframe.png").convert("RGB")),
    )
    assert np.array_equal(
        np.load(base / "depth.npy"),
        np.load(uploaded / "depth.npy"),
    )


def test_recorded_eeg_rejects_unsupported_files(eeg_env):
    with TestClient(api.app) as client:
        response = client.post(
            "/api/eeg",
            files={"file": ("memory.edf", b"not an edf", "application/octet-stream")},
            data={"sfreq": "250"},
        )
    assert response.status_code == 400
    assert "not supported" in response.json()["detail"]
    assert api._eeg_jobs == {}


def test_recorded_eeg_rejects_too_short_recordings(eeg_env):
    with TestClient(api.app) as client:
        response = client.post(
            "/api/eeg",
            files={"file": ("short.csv", _csv(seconds=1.0), "text/csv")},
            data={"sfreq": "250"},
        )
    assert response.status_code == 400
    assert "three" in response.json()["detail"] or "windows" in response.json()["detail"]


def test_recorded_eeg_body_size_guard(eeg_env, monkeypatch):
    monkeypatch.setattr(api, "EEG_MAX_BYTES", 1)
    with TestClient(api.app) as client:
        response = client.post(
            "/api/eeg",
            files={"file": ("big.csv", b"x" * (300 * 1024), "text/csv")},
            data={"sfreq": "250"},
        )
    assert response.status_code == 413
    assert "MB" in response.json()["detail"] or "limit" in response.json()["detail"]


def test_recorded_eeg_is_rate_limited(eeg_env, monkeypatch):
    monkeypatch.setattr(api, "EEG_RATE_MAX", 1)
    with TestClient(api.app) as client:
        first = client.post(
            "/api/eeg",
            files={"file": ("bad.edf", b"x", "application/octet-stream")},
            data={"sfreq": "250"},
        )
        second = client.post(
            "/api/eeg",
            files={"file": ("bad.edf", b"x", "application/octet-stream")},
            data={"sfreq": "250"},
        )
    assert first.status_code == 400
    assert second.status_code == 429
    assert "Too many" in second.json()["detail"]
