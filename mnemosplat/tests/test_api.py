"""HTTP surface: health, cross-origin isolation, and public-deployment guards."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import mns.api as api


class _FakeRequest:
    def __init__(self, token: str | None = None) -> None:
        self.headers = {} if token is None else {"x-mns-token": token}


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
