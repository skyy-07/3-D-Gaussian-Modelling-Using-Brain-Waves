"""FastAPI surface for the demo.

Endpoints
---------
GET  /api/health                 liveness + capability report
GET  /api/scenarios              built-in narratives
GET  /api/scenes                 generated episodes (reads out/index.json)
POST /api/episode                start a generation job  -> {"job": id}
GET  /api/jobs/{id}              poll job state
POST /api/segment                SAM 3.1 on an uploaded frame
POST /api/transcribe             Muse Voice Transcribe on an uploaded clip
GET  /assets/*                   splat/keyframe/manifest assets
GET  /                           built WebXR client (web/dist) when present

Generation is asynchronous because Muse Image can take 30-60 s; a stage demo should
never sit on a blocked request.
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import logging
import os
import secrets
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from .depth import DepthEstimator
from .director.meta_client import MetaAPIError, MetaModelAPI, transcript_from_audio
from .eeg.brainvis import resolve_brainvis
from .pipeline import PipelineConfig, run_episode, run_pair, selftest
from .scenarios import list_scenarios
from .util import configure_logging, read_json, write_json

log = logging.getLogger("mns.api")

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("MNS_OUT", ROOT / "out" / "demo"))
WEB_DIST = Path(os.environ.get("MNS_WEB_DIST", ROOT / "web" / "dist"))

# Public deployments serve pre-generated assets read-only. Generation, segmentation and
# transcription spend money and consume a single worker, so they are opt-in.
PUBLIC_DEMO = os.environ.get("MNS_PUBLIC_DEMO", "").strip().lower() in ("1", "true", "yes")
ADMIN_TOKEN = os.environ.get("MNS_ADMIN_TOKEN", "")
ALLOW_ORIGINS = [
    o.strip() for o in os.environ.get("MNS_ALLOW_ORIGINS", "*").split(",") if o.strip()
]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging(os.environ.get("MNS_LOG", "INFO"))
    OUT.mkdir(parents=True, exist_ok=True)
    log.info(
        "assets root: %s | web dist: %s | public demo: %s",
        OUT,
        WEB_DIST,
        PUBLIC_DEMO,
    )
    yield


app = FastAPI(
    title="MnemoSplat",
    version="0.1.0",
    description="EEG-salience 3D memory volumes",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _require_write_access(request: Request) -> None:
    """Gate money-spending endpoints on a public deployment.

    ``MNS_PUBLIC_DEMO=1`` disables generation unless ``MNS_ADMIN_TOKEN`` is supplied in
    ``X-MNS-Token``. If the token is configured, it is required even outside public mode.
    """
    if PUBLIC_DEMO and not ADMIN_TOKEN:
        raise HTTPException(403, "generation is disabled on this public demo")
    if ADMIN_TOKEN:
        supplied = request.headers.get("x-mns-token", "")
        if not secrets.compare_digest(supplied, ADMIN_TOKEN):
            raise HTTPException(403, "invalid or missing X-MNS-Token")


_WRITE_PATHS = frozenset({"/api/episode", "/api/selftest", "/api/segment", "/api/transcribe"})


@app.middleware("http")
async def _write_guard(request: Request, call_next):
    """Reject write requests before FastAPI parses uploads or JSON bodies."""
    if request.method == "POST" and request.url.path in _WRITE_PATHS:
        try:
            _require_write_access(request)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return await call_next(request)


@app.middleware("http")
async def _cross_origin_isolation(request, call_next):
    """Enable SharedArrayBuffer for the WebXR splat sorter.

    GaussianSplats3D uses a SharedArrayBuffer to talk to its sort worker. Browsers
    only expose it on cross-origin-isolated pages, so the demo server (and Vite in
    dev) must send these headers. Without them the client falls back to copying.
    """
    response = await call_next(request)
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    return response


@app.middleware("http")
async def _cache_headers(request: Request, call_next):
    """Cache headers for the two asset classes: hashed bundles and demo frames."""
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/"):
        response.headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
    elif path == "/" or path.endswith(("index.html", "index.json", "manifest.json")):
        response.headers.setdefault("Cache-Control", "no-cache")
    elif path.startswith("/assets/"):
        # Frame filenames are stable across rebuilds. Keep the edge cache short so a
        # redeploy does not pair a fresh manifest with stale splats.
        response.headers.setdefault("Cache-Control", "public, max-age=300")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    return response


_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="mns")
_jobs: dict[str, dict] = {}
_static_mounted = False


class EpisodeRequest(BaseModel):
    narrative: str | None = None
    scenario: str = "focus"
    use_muse_image: bool = Field(False, description="call muse-image-1.0 ($0.01/frame)")
    use_sam: bool = True
    use_meta: bool = True
    target_splats: int = Field(120_000, ge=2_000, le=262_144)
    width: int = Field(512, ge=256, le=1024)
    height: int = Field(512, ge=256, le=1024)
    compare: bool = Field(False, description="also render the paired condition")
    subject: str = "synthetic-subject"


def _mount_static() -> None:
    global _static_mounted
    if _static_mounted:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    app.mount("/assets", StaticFiles(directory=str(OUT)), name="assets")
    if WEB_DIST.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
    _static_mounted = True


@app.get("/api/health")
def health() -> dict:
    api = MetaModelAPI()
    return {
        "ok": True,
        "version": app.version,
        "model_api_key": bool(api.available),
        "models": {
            "text": os.environ.get("MNS_TEXT_MODEL", "muse-spark-1.3"),
            "image": os.environ.get("MNS_IMAGE_MODEL", "muse-image-1.0"),
            "segmentation": os.environ.get("MNS_SEG_MODEL", "sam-3.1"),
            "asr": os.environ.get("MNS_ASR_MODEL", "muse-voice-transcribe-1.0"),
        },
        "depth_learned": DepthEstimator().learned_available,
        "brainvis": resolve_brainvis(),
        "web_built": WEB_DIST.exists(),
        "assets": str(OUT),
        "jobs": len(_jobs),
        "public_demo": PUBLIC_DEMO,
        "generation_enabled": (not PUBLIC_DEMO) or bool(ADMIN_TOKEN),
    }


@app.get("/api/scenarios")
def scenarios() -> dict:
    return {"scenarios": list_scenarios()}


@app.get("/api/scenes")
def scenes() -> dict:
    index = OUT / "index.json"
    if index.exists():
        return read_json(index)
    manifests = sorted(OUT.glob("*/manifest.json"))
    return {
        "scenarios": [
            {
                "key": p.parent.name,
                "label": p.parent.name,
                "manifest": f"{p.parent.name}/manifest.json",
                "caption": "",
            }
            for p in manifests
        ]
    }


def _job_runner(job_id: str, req: EpisodeRequest) -> None:
    job = _jobs[job_id]
    job["state"] = "running"
    job["started"] = time.time()
    try:
        api = MetaModelAPI()
        cfg = PipelineConfig(
            width=req.width,
            height=req.height,
            target_splats=req.target_splats,
            scenario=req.scenario,
            subject=req.subject,
            use_meta=req.use_meta,
            use_muse_image=req.use_muse_image,
            use_sam=req.use_sam,
            force=True,
        )
        if req.compare:
            keys = ("focus", "distracted") if req.scenario != "street" else ("street",)
            target = OUT / "live" / job_id
            run_pair(target, cfg, keys=keys, api=api)
            job.update(
                state="done",
                base=f"live/{job_id}",
                index=f"live/{job_id}/index.json",
                compare=True,
            )
        else:
            target = OUT / "live" / job_id
            res = run_episode(target, cfg, narrative=req.narrative, api=api)
            write_json(
                target / "job.json",
                {"request": req.model_dump(), "warnings": res.warnings, "elapsed_s": res.elapsed_s},
            )
            job.update(
                state="done",
                base=f"live/{job_id}",
                manifest=f"live/{job_id}/manifest.json",
                warnings=res.warnings,
                elapsed_s=res.elapsed_s,
            )
        job["finished"] = time.time()
        log.info("job %s done in %.1fs", job_id, job["finished"] - job["started"])
    except Exception as exc:  # noqa: BLE001 - surface the failure to the UI
        log.exception("job %s failed", job_id)
        job.update(state="error", error=str(exc), finished=time.time())


@app.post("/api/episode")
def create_episode(req: EpisodeRequest, request: Request) -> dict:
    _require_write_access(request)
    _mount_static()
    job_id = uuid.uuid4().hex[:10]
    _jobs[job_id] = {"id": job_id, "state": "queued", "request": req.model_dump()}
    _executor.submit(_job_runner, job_id, req)
    return {"job": job_id, "state": "queued"}


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(404, "unknown job")
    return _jobs[job_id]


@app.post("/api/segment")
async def segment(
    request: Request, image: UploadFile = File(...), prompt: str = Form("mug")
) -> dict:
    _require_write_access(request)
    api = MetaModelAPI()
    if not api.available:
        raise HTTPException(503, "MODEL_API_KEY not set")
    raw = await image.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    b64 = base64.b64encode(_png(img)).decode("ascii")
    try:
        segs = api.segment(b64, prompt)
    except MetaAPIError as exc:
        raise HTTPException(502, str(exc)) from exc
    return {
        "prompt": prompt,
        "segments": [
            {
                "id": s.object_id,
                "bbox": s.bbox,
                "confidence": s.confidence,
                "has_mask": bool(s.mask),
            }
            for s in segs
        ],
    }


@app.post("/api/transcribe")
async def transcribe(request: Request, audio: UploadFile = File(...)) -> dict:
    _require_write_access(request)
    raw = await audio.read()
    suffix = Path(audio.filename or "memory.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
        fh.write(raw)
        path = fh.name
    try:
        text, model = transcript_from_audio(path)
    except MetaAPIError as exc:
        raise HTTPException(502, str(exc)) from exc
    finally:
        Path(path).unlink(missing_ok=True)
    return {"text": text, "model": model}


@app.post("/api/selftest")
def selftest_endpoint(request: Request) -> dict:
    _require_write_access(request)
    return selftest(OUT / "selftest")


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_mount_static()


@app.exception_handler(MetaAPIError)
def _meta_error(_request, exc: MetaAPIError) -> JSONResponse:  # pragma: no cover
    return JSONResponse(status_code=502, content={"error": str(exc)})


def main() -> None:
    import uvicorn

    configure_logging(os.environ.get("MNS_LOG", "INFO"))
    uvicorn.run(
        "mns.api:app",
        host=os.environ.get("MNS_HOST", "0.0.0.0"),
        port=int(os.environ.get("MNS_PORT", os.environ.get("PORT", "8000"))),
        reload=False,
    )


__all__ = ["app", "main"]
