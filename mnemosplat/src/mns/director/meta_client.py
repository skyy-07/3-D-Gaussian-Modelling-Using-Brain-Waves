"""Thin, defensive client for the Meta Model API (``https://api.meta.ai/v1``).

Model map used by MnemoSplat
----------------------------
* ``muse-spark-1.3``        structured scene graph from a memory transcript
* ``muse-wonder-1.0``       optional: harder narratives, long transcripts
* ``muse-image-1.0``        the remembered keyframe + the "distractor" variant
* ``sam-3.1``               object masks/boxes to sharpen the attention field
* ``muse-voice-transcribe-1.0``  speak the memory instead of typing it

Notes
-----
* The retired ``api.llama.com`` preview is *not* used; it was shut down in July 2026.
* Everything degrades: if a call fails, the pipeline falls back to the offline
  provider and records the failure in ``manifest.upstream`` so the demo never
  pretends a model ran when it did not.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from ..util import clamp

log = logging.getLogger("mns.meta")

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
TEXT_MODEL = os.environ.get("MNS_TEXT_MODEL", "muse-spark-1.3")
IMAGE_MODEL = os.environ.get("MNS_IMAGE_MODEL", "muse-image-1.0")
SEG_MODEL = os.environ.get("MNS_SEG_MODEL", "sam-3.1")
ASR_MODEL = os.environ.get("MNS_ASR_MODEL", "muse-voice-transcribe-1.0")

_BOX_RE = re.compile(
    r"(?:(\d+)\s*)?<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)"
    r"(?:;c=([0-9.]+))?\|>"
)
_MASK_RE = re.compile(r"<\|mask;x=0;y=0(?:;c=[0-9.]+)?;data=(\d+),(\d+),([~!])([^|]*)\|>")


class MetaAPIError(RuntimeError):
    """Any non-recoverable failure talking to the Meta Model API."""


@dataclass
class SegmentResult:
    object_id: int
    bbox: tuple[float, float, float, float]  # normalised u0,v0,u1,v1
    confidence: float
    mask: dict | None = None  # {"h","w","data"} raw, decoded lazily


class MetaModelAPI:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 3,
    ):
        self.api_key = api_key or os.environ.get("MODEL_API_KEY", "")
        self.base_url = (base_url or os.environ.get("MNS_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client: httpx.Client | None = None
        self.calls: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ infra
    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "mnemosplat/0.1",
                },
                timeout=httpx.Timeout(self.timeout_s, connect=10.0),
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        if not self.available:
            raise MetaAPIError("MODEL_API_KEY is not set")
        last: Exception | None = None
        for attempt in range(self.max_retries):
            started = time.perf_counter()
            try:
                resp = self._http().request(method, path, **kwargs)
                elapsed = round((time.perf_counter() - started) * 1000)
                self.calls.append(
                    {
                        "path": path,
                        "status": resp.status_code,
                        "ms": elapsed,
                        "attempt": attempt + 1,
                    }
                )
                if resp.status_code in (429,) or 500 <= resp.status_code < 600:
                    raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
                if resp.status_code >= 400:
                    raise MetaAPIError(f"{path} -> HTTP {resp.status_code}: {resp.text[:300]}")
                return resp
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last = exc
                sleep = 0.6 * (2**attempt)
                log.warning("Meta API %s failed (%s), retrying in %.1fs", path, exc, sleep)
                time.sleep(sleep)
        raise MetaAPIError(f"{path} failed after {self.max_retries} attempts: {last}")

    # ------------------------------------------------------- text / structure
    def chat_json(
        self,
        system: str,
        user: str,
        schema: dict,
        model: str = TEXT_MODEL,
        max_tokens: int = 3000,
        temperature: float = 0.6,
    ) -> tuple[dict, str]:
        """Structured-output chat. Returns ``(parsed_json, model_id)``."""
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_schema", "json_schema": schema},
        }
        try:
            resp = self._request("POST", "/chat/completions", json=body)
        except MetaAPIError as exc:
            # older/hosted variants may not accept json_schema; retry plain
            log.warning("json_schema rejected (%s); retrying with json_object", exc)
            body.pop("response_format", None)
            body["response_format"] = {"type": "json_object"}
            resp = self._request("POST", "/chat/completions", json=body)

        payload = resp.json()
        content = _first_message_text(payload)
        return _extract_json(content), payload.get("model", model)

    # ------------------------------------------------------------------ image
    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        n: int = 1,
        output_format: str = "png",
        reasoning_strength: str = "high",
        model: str = IMAGE_MODEL,
    ) -> list[bytes]:
        """Muse Image: flat $0.01/image, returns PNG/WebP bytes."""
        body = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": "b64_json",
            "output_format": output_format,
            "reasoning_strength": reasoning_strength,
            "tool_enablement": {"enable_web_search": False, "enable_shell": False},
        }
        resp = self._request("POST", "/images/generations", json=body)
        data = resp.json().get("data", [])
        out: list[bytes] = []
        for item in data:
            b64 = item.get("b64_json")
            if b64:
                out.append(base64.b64decode(b64))
        if not out:
            raise MetaAPIError("Muse Image returned no images (possibly safety-filtered)")
        return out

    # -------------------------------------------------------------- segments
    def segment(
        self,
        image_b64: str,
        prompt: str,
        include_confidence: bool = True,
        model: str = SEG_MODEL,
    ) -> list[SegmentResult]:
        """SAM 3.1 over the Responses API; returns normalised boxes.

        Mask payloads are returned raw. Decoding is best-effort and validated by
        exact raster length, so a format change degrades to boxes instead of
        corrupting the attention map.
        """
        if not self.available:
            raise MetaAPIError("MODEL_API_KEY is not set")
        text = prompt
        if include_confidence:
            text = f"{prompt} include_confidence=true"
        body = {
            "model": model,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": text},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{image_b64}",
                        },
                    ],
                }
            ],
            "stream": True,
        }
        chunks: list[str] = []
        with self._http().stream("POST", "/responses", json=body) as stream:
            for line in stream.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw in ("", "[DONE]"):
                    continue
                try:
                    evt = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if evt.get("type") == "response.output_text.delta":
                    chunks.append(evt.get("delta", ""))
        return parse_sam_output("".join(chunks))


# --------------------------------------------------------------------- utils
def _first_message_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    for item in payload.get("output", []) or []:
        if item.get("type") == "message":
            for c in item.get("content", []) or []:
                if c.get("type") in ("output_text", "text"):
                    return c.get("text", "")
    choices = payload.get("choices") or []
    if choices:
        return choices[0].get("message", {}).get("content", "") or ""
    return ""


def _extract_json(text: str) -> dict:
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            return json.loads(t[start : end + 1])
        raise MetaAPIError(f"model did not return JSON: {t[:200]!r}") from None


def parse_sam_output(text: str) -> list[SegmentResult]:
    """Parse the SAM 3.1 wire grammar (boxes always, masks when present)."""
    results: list[SegmentResult] = []
    for m in _BOX_RE.finditer(text):
        oid = int(m.group(1)) if m.group(1) else len(results)
        x1, y1, x2, y2, w, h = (int(m.group(i)) for i in range(2, 8))
        conf = float(m.group(8)) if m.group(8) else 0.9
        if w <= 0 or h <= 0 or x2 <= x1 or y2 <= y1:
            continue
        mask = None
        tail = text[m.end() : m.end() + 4096]
        mm = _MASK_RE.match(tail)
        if mm:
            mask = {
                "h": int(mm.group(1)),
                "w": int(mm.group(2)),
                "enc": mm.group(3),
                "payload": mm.group(4),
            }
        results.append(
            SegmentResult(
                object_id=oid,
                bbox=(
                    clamp(x1 / w, 0.0, 1.0),
                    clamp(y1 / h, 0.0, 1.0),
                    clamp(x2 / w, 0.0, 1.0),
                    clamp(y2 / h, 0.0, 1.0),
                ),
                confidence=clamp(conf, 0.0, 1.0),
                mask=mask,
            )
        )
    return results


def decode_mask(mask: dict) -> Any | None:
    """Best-effort SAM mask decode. Returns uint8 (h,w) in {0,1} or ``None``.

    Validated against the declared raster size, so an encoding change cannot silently
    corrupt the attention field - we simply fall back to box masks.
    """
    try:
        import zlib

        import numpy as np

        h, w = int(mask["h"]), int(mask["w"])
        raw = base64.a85decode(mask["payload"].encode("ascii"), adobe=False)
        candidates: list[bytes] = []
        if len(raw) == h * w:
            candidates.append(raw)
        else:
            candidates.append(zlib.decompress(raw))
        for buf in candidates:
            if len(buf) == h * w:
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
                return (arr > 0).astype("uint8")
    except Exception:  # noqa: BLE001 - decode is optional by design
        return None
    return None


def transcript_from_audio(path: str, api_key: str | None = None) -> tuple[str, str]:
    """Muse Voice Transcribe. Reads a media file, returns ``(text, model)``."""
    key = api_key or os.environ.get("MODEL_API_KEY", "")
    if not key:
        raise MetaAPIError("MODEL_API_KEY is not set")
    with open(path, "rb") as fh:
        blob = fh.read()
    filename = os.path.basename(path)
    base = os.environ.get("MNS_BASE_URL", DEFAULT_BASE_URL)
    with httpx.Client(
        base_url=base,
        headers={"Authorization": f"Bearer {key}"},
        timeout=httpx.Timeout(300.0, connect=10.0),
    ) as client:
        for endpoint in ("/asr/transcribe", "/audio/transcriptions"):
            resp = client.post(
                endpoint,
                data={"model": ASR_MODEL},
                files={"file": (filename, blob)},
            )
            if resp.status_code < 400:
                payload = resp.json()
                return payload.get("text", ""), payload.get("model", ASR_MODEL)
            log.warning("transcribe %s -> HTTP %s", endpoint, resp.status_code)
    raise MetaAPIError("transcription failed on every known endpoint")

