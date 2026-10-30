"""Experimental BrainVis reconstruction client.

This is deliberately separate from :mod:`mns.eeg.brainvis`. The salience bridge uses
BrainVis embeddings to rank scene entities; this module talks to an optional GPU worker
that runs the authors' image-reconstruction cascade and returns static frames.

The CPU demo must never silently fall back to a generated image. If no worker is
configured, :class:`BrainVisClient` reports that fact and the API refuses the request.
"""

from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np
from PIL import Image

DEFAULT_TIMEOUT_S = 900.0
DEFAULT_PROBE_TIMEOUT_S = 4.0


class ReconstructError(RuntimeError):
    """A worker or protocol failure that should be shown to the caller."""


@dataclass(frozen=True)
class WorkerStatus:
    configured: bool
    ready: bool
    reason: str
    backend: str = "brainvis"
    cuda: bool | None = None
    checkpoint: bool | None = None
    stable_diffusion: bool | None = None
    worker_version: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "ready": self.ready,
            "reason": self.reason,
            "backend": self.backend,
            "cuda": self.cuda,
            "checkpoint": self.checkpoint,
            "stable_diffusion": self.stable_diffusion,
            "worker_version": self.worker_version,
        }


@dataclass
class ReconFrame:
    index: int
    png: bytes
    predicted_class: int | None = None
    label: str | None = None
    prompt: str | None = None


@dataclass
class ReconResult:
    frames: list[ReconFrame] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class BrainVisClient:
    """HTTP client for the separate CUDA worker.

    ``transport`` exists for tests and for a local in-process worker; production uses
    the default httpx transport.
    """

    def __init__(
        self,
        url: str,
        token: str = "",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        probe_timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.url = (url or "").strip().rstrip("/")
        self.token = token or ""
        self.timeout_s = max(30.0, timeout_s)
        self.probe_timeout_s = max(1.0, probe_timeout_s)
        self._transport = transport

    @classmethod
    def from_env(cls) -> BrainVisClient:
        return cls(
            url=os.environ.get("MNS_BRAINVIS_WORKER_URL", ""),
            token=os.environ.get("MNS_BRAINVIS_WORKER_TOKEN", ""),
            timeout_s=_env_float("MNS_BRAINVIS_TIMEOUT_S", DEFAULT_TIMEOUT_S),
        )

    @property
    def configured(self) -> bool:
        return bool(self.url)

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _client(self, timeout_s: float) -> httpx.Client:
        return httpx.Client(
            headers=self._headers(),
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            transport=self._transport,
        )

    def status(self) -> WorkerStatus:
        if not self.configured:
            return WorkerStatus(
                configured=False,
                ready=False,
                reason=(
                    "MNS_BRAINVIS_WORKER_URL is unset. The CPU demo cannot run "
                    "BrainVis or Stable Diffusion; point it at a CUDA worker."
                ),
            )
        try:
            with self._client(self.probe_timeout_s) as client:
                response = client.get(f"{self.url}/health")
                response.raise_for_status()
                body = response.json()
        except Exception as exc:  # noqa: BLE001 - status must never crash the API
            return WorkerStatus(
                configured=True,
                ready=False,
                reason=f"BrainVis worker is unreachable: {exc}",
            )

        ready = bool(body.get("ready"))
        reason = str(body.get("reason") or "").strip()
        if not reason:
            reason = "worker is ready" if ready else "worker reported not ready"
        return WorkerStatus(
            configured=True,
            ready=ready,
            reason=reason,
            backend=str(body.get("backend") or "brainvis"),
            cuda=body.get("cuda"),
            checkpoint=body.get("checkpoint"),
            stable_diffusion=body.get("stable_diffusion"),
            worker_version=body.get("version"),
        )

    def reconstruct(
        self,
        signal: np.ndarray,
        sfreq: float,
        channels: tuple[str, ...] | list[str],
        *,
        prompt: str | None = None,
        seed: int | None = None,
        normalize: bool = True,
    ) -> ReconResult:
        """Send one or more ``(channels, samples)`` windows to the GPU worker.

        The worker owns BrainVis preprocessing, channel validation, checkpoint loading
        and diffusion. This process never imports torch.
        """
        if not self.configured:
            raise ReconstructError("MNS_BRAINVIS_WORKER_URL is not configured")
        arr = np.asarray(signal, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ReconstructError(
                "signal must be (channels, samples) or (windows, channels, samples)"
            )
        if not np.isfinite(arr).all():
            raise ReconstructError("signal contains NaN or infinite values")
        if arr.shape[1] != len(channels):
            raise ReconstructError("channel count does not match the signal array")

        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        data: dict[str, str] = {
            "sfreq": str(float(sfreq)),
            "channels": json.dumps(list(channels)),
            "normalize": "1" if normalize else "0",
        }
        if prompt:
            data["prompt"] = prompt
        if seed is not None:
            data["seed"] = str(int(seed))

        try:
            with self._client(self.timeout_s) as client:
                response = client.post(
                    f"{self.url}/reconstruct",
                    files={"file": ("eeg.npy", buf.getvalue(), "application/octet-stream")},
                    data=data,
                )
        except Exception as exc:  # noqa: BLE001 - wrap transport failures with context
            raise ReconstructError(f"BrainVis worker request failed: {exc}") from exc

        if response.status_code >= 400:
            detail = response.text[:600]
            try:
                parsed = response.json()
                detail = str(parsed.get("detail") or parsed.get("error") or detail)
            except Exception:  # noqa: BLE001
                pass
            raise ReconstructError(
                f"BrainVis worker returned HTTP {response.status_code}: {detail}"
            )

        try:
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            raise ReconstructError("BrainVis worker returned invalid JSON") from exc

        frames: list[ReconFrame] = []
        for item in body.get("frames") or []:
            encoded = item.get("png_b64") or item.get("png_base64")
            if not encoded:
                continue
            try:
                png = base64.b64decode(encoded, validate=True)
                with Image.open(io.BytesIO(png)) as image:
                    image.verify()
                    if image.format != "PNG":
                        raise ValueError("not PNG")
            except Exception as exc:  # noqa: BLE001
                raise ReconstructError(f"worker returned an invalid PNG frame: {exc}") from exc
            frames.append(
                ReconFrame(
                    index=int(item.get("index", len(frames))),
                    png=png,
                    predicted_class=(
                        int(item["predicted_class"])
                        if item.get("predicted_class") is not None
                        else None
                    ),
                    label=str(item["label"]) if item.get("label") else None,
                    prompt=str(item["prompt"]) if item.get("prompt") else None,
                )
            )
        if not frames:
            raise ReconstructError("BrainVis worker returned no image frames")
        return ReconResult(frames=frames, meta=dict(body.get("meta") or {}))


__all__ = [
    "BrainVisClient",
    "ReconFrame",
    "ReconResult",
    "ReconstructError",
    "WorkerStatus",
]
