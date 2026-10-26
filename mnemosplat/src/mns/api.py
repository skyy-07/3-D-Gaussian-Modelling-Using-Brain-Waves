"""FastAPI surface for the demo.

Endpoints
---------
GET  /api/health                 liveness + capability report
GET  /api/scenarios              built-in narratives
GET  /api/scenes                 generated episodes (reads out/index.json)
POST /api/episode                start a generation job  -> {"job": id}
GET  /api/jobs/{id}              poll job state
POST /api/eeg                    upload a recorded EEG -> {"job": id} (public-safe)
GET  /api/eeg/jobs/{id}          poll a recorded-EEG re-lift job
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
import shutil
import tempfile
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from .depth import DepthEstimator
from .director.meta_client import MetaAPIError, MetaModelAPI, transcript_from_audio
from .eeg import (
    EEGFormatError,
    aggregate_epochs,
    analyze_epochs,
    parse_eeg_upload,
    safe_filename,
)
from .eeg.brainvis import resolve_brainvis
from .pipeline import PipelineConfig, run_episode, run_pair, selftest
from .scenarios import get_scenario, list_scenarios
from .schemas import Manifest
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


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Recorded-EEG uploads are CPU-only and never call a paid API, so they are available
# on the public demo. They still need abuse bounds: one re-lift at a time, a small file
# limit, a short per-IP cooldown, and a lower splat budget than the baked 120k demo.
EEG_MAX_BYTES = max(
    256 * 1024,
    min(32 * 1024 * 1024, _env_int("MNS_EEG_MAX_MB", 12) * 1024 * 1024),
)
EEG_TARGET_SPLATS = max(2_000, min(120_000, _env_int("MNS_EEG_TARGET_SPLATS", 60_000)))
EEG_MAX_ACTIVE = max(1, _env_int("MNS_EEG_MAX_ACTIVE", 1))
EEG_RATE_MAX = max(1, _env_int("MNS_EEG_RATE_MAX", 4))
EEG_RATE_WINDOW_S = max(30, _env_int("MNS_EEG_RATE_WINDOW_S", 900))
EEG_KEEP_JOBS = max(1, _env_int("MNS_EEG_KEEP_JOBS", 4))
EEG_EPOCH_S = 0.75  # matches the baked 4-beat narrative; not operator-tunable


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
    """Reject write requests before FastAPI parses uploads or JSON bodies.

    ``/api/eeg`` is intentionally not behind the paid-generation guard, but its body
    size is rejected here before multipart parsing so a public URL cannot be made to
    spool a huge upload.
    """
    if request.method == "POST" and request.url.path in _WRITE_PATHS:
        try:
            _require_write_access(request)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    if request.method == "POST" and request.url.path == "/api/eeg":
        declared = request.headers.get("content-length")
        if declared:
            try:
                too_large = int(declared) > EEG_MAX_BYTES + 256 * 1024
            except ValueError:
                too_large = False
            if too_large:
                limit_mb = EEG_MAX_BYTES // (1024 * 1024)
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"EEG upload is limited to {limit_mb} MB"},
                )
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
_eeg_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="mns-eeg")
_eeg_jobs: dict[str, dict] = {}
_eeg_rate: dict[str, deque[float]] = {}
_eeg_lock = threading.Lock()
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


# --------------------------------------------------------------------------- #
# recorded-EEG upload helpers
# --------------------------------------------------------------------------- #
def _base_episode(preferred: str = "focus") -> tuple[Path, Manifest]:
    """Find a baked episode whose keyframe/depth can be re-lifted.

    The upload path never synthesises a new image. It reuses one of the assets that
    already exist in the image, so the only changed variable is the uploaded EEG.
    """
    wanted = Path(str(preferred or "focus")).name
    candidates: list[Path] = [OUT / wanted, OUT / "focus", OUT / "distracted"]
    candidates.extend(sorted({path.parent for path in OUT.glob("*/manifest.json")}))

    seen: set[Path] = set()
    for folder in candidates:
        try:
            folder = folder.resolve()
        except OSError:
            continue
        if folder in seen:
            continue
        seen.add(folder)
        manifest_path = folder / "manifest.json"
        keyframe = folder / "keyframe.png"
        depth = folder / "depth.npy"
        if not (manifest_path.exists() and keyframe.exists() and depth.exists()):
            continue
        try:
            manifest = Manifest(**read_json(manifest_path))
            if not manifest.frames:
                continue
            with Image.open(keyframe) as image:
                width, height = image.size
            depth_array = np.load(depth)
            if depth_array.shape != (height, width):
                log.warning(
                    "base episode %s has depth %s but keyframe %sx%s",
                    folder,
                    depth_array.shape,
                    width,
                    height,
                )
                continue
        except Exception as exc:  # noqa: BLE001 - try the next baked episode
            log.warning("ignoring unusable base episode %s: %s", folder, exc)
            continue
        return folder, manifest

    raise HTTPException(
        503,
        "No baked episode with keyframe.png and depth.npy is available. "
        "Generate the demo assets before using recorded-EEG upload.",
    )


def _client_ip(request: Request) -> str:
    # Render appends the address it saw to X-Forwarded-For, so the rightmost entry is
    # the one we can trust. Reading the leftmost entry would let a client pick its own
    # rate-limit bucket by sending a spoofed header.
    forwarded = request.headers.get("x-forwarded-for", "")
    parts = [part.strip() for part in forwarded.split(",") if part.strip()]
    if parts:
        return parts[-1][:64]
    return (request.client.host if request.client else "unknown")[:64]


def _check_eeg_access(request: Request) -> None:
    """Per-IP cooldown + one active public re-lift at a time."""
    ip = _client_ip(request)
    now = time.time()
    with _eeg_lock:
        # Bound the per-IP map on a long-lived public instance. Only do the sweep when
        # it has grown unusually large; normal traffic never pays for it.
        if len(_eeg_rate) > 2048:
            for key, seen in list(_eeg_rate.items()):
                while seen and now - seen[0] > EEG_RATE_WINDOW_S:
                    seen.popleft()
                if not seen:
                    _eeg_rate.pop(key, None)
        hits = _eeg_rate.setdefault(ip, deque())
        while hits and now - hits[0] > EEG_RATE_WINDOW_S:
            hits.popleft()
        if len(hits) >= EEG_RATE_MAX:
            retry = int(EEG_RATE_WINDOW_S - (now - hits[0])) if hits else EEG_RATE_WINDOW_S
            raise HTTPException(
                429,
                f"Too many EEG uploads from this address. Try again in {max(1, retry)}s.",
            )
        active = sum(
            1 for job in _eeg_jobs.values() if job.get("state") in ("queued", "running")
        )
        if active >= EEG_MAX_ACTIVE:
            raise HTTPException(
                429,
                "The EEG worker is busy with another upload. Try again when it finishes.",
            )
        hits.append(now)


def _prune_eeg_jobs() -> None:
    """Keep only the newest completed uploads; Render's disk is ephemeral anyway."""
    with _eeg_lock:
        finished = [
            (job_id, job)
            for job_id, job in _eeg_jobs.items()
            if job.get("state") in ("done", "error")
        ]
        finished.sort(key=lambda item: item[1].get("finished") or item[1].get("created") or 0)
        stale = finished[:-EEG_KEEP_JOBS] if len(finished) > EEG_KEEP_JOBS else []
        for job_id, _job in stale:
            _eeg_jobs.pop(job_id, None)
    for job_id, _job in stale:
        shutil.rmtree(OUT / "live" / f"eeg-{job_id}", ignore_errors=True)


async def _read_limited(upload: UploadFile) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(256 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > EEG_MAX_BYTES:
            limit_mb = EEG_MAX_BYTES // (1024 * 1024)
            raise HTTPException(413, f"EEG upload is limited to {limit_mb} MB")
        chunks.append(chunk)
    return b"".join(chunks)


def _eeg_runner(
    job_id: str,
    recording,
    info: dict,
    base_dir: Path,
    base_manifest: Manifest,
) -> None:
    """Analyse the recording, then re-lift the baked base with those features only."""
    job = _eeg_jobs.get(job_id)
    if job is None:
        return
    target = OUT / "live" / f"eeg-{job_id}"
    started = time.time()
    with _eeg_lock:
        job.update(
            state="running",
            stage="analyzing",
            progress=0.15,
            message="Welch PSD · band power · population calibration…",
            started=started,
        )
    try:
        feats, calibrator, band_table = analyze_epochs(
            recording,
            epoch_s=EEG_EPOCH_S,
            calibration="population",
        )
        if len(feats) < 3:
            raise ValueError(
                "the recording produced fewer than three usable 0.75 s windows; "
                "upload at least 2.25 s of EEG"
            )

        n_beats = max(1, min(len(base_manifest.scene_graph.beats) or 4, 6))
        aggregated = aggregate_epochs(feats, n_beats, epoch_s=EEG_EPOCH_S)
        with Image.open(base_dir / "keyframe.png") as image:
            width, height = image.size

        try:
            base_scenario = get_scenario(base_manifest.scenario)
        except KeyError:
            base_scenario = get_scenario("focus")

        cfg = PipelineConfig(
            width=width,
            height=height,
            epoch_s=EEG_EPOCH_S,
            target_splats=EEG_TARGET_SPLATS,
            scenario=base_scenario.key,
            subject=info["filename"],
            calibration="population",
            use_meta=False,
            use_muse_image=False,
            use_sam=False,
            per_beat_keyframes=False,
            write_ply_frames=0,
            preview=False,
            turntable=False,
            force=True,
        )

        with _eeg_lock:
            job.update(
                stage="lifting",
                progress=0.55,
                message=f"Re-lifting the shared keyframe · {len(aggregated)} beats…",
            )

        # ``api_key=""`` is an extra guarantee: even a build environment with a key
        # cannot turn this public endpoint into a paid model call.
        result = run_episode(
            target,
            cfg,
            narrative=base_manifest.scene_graph.narrative,
            recording=recording,
            scenario=base_scenario,
            api=MetaModelAPI(api_key=""),
            graph_override=base_manifest.scene_graph,
            keyframe_override=base_dir / "keyframe.png",
            depth_override=base_dir / "depth.npy",
            features_override=aggregated,
            calibrator_override=calibrator,
            band_table_override=band_table,
        )

        manifest = result.manifest
        manifest.id = f"eeg-{job_id}"
        manifest.scenario = "recorded"
        manifest.subject = info["filename"]
        manifest.upstream.update(
            {
                "eeg": "file",
                "eeg_file": info["filename"],
                "eeg_upload": f"{len(feats)} windows -> {len(aggregated)} beats",
            }
        )
        manifest.notes = [
            *manifest.notes,
            (
                f"Recorded EEG: {info['filename']} ({len(recording.channels)} channels, "
                f"{recording.sfreq:g} Hz, {recording.duration_s:.1f}s). "
                f"{len(feats)} Welch windows were reduced to {len(aggregated)} "
                "consolidation beats."
            ),
            (
                "The uploaded EEG changed only density, opacity, covariance and sampling; "
                "the keyframe, depth map and scene graph are the baked base episode."
            ),
            (
                "Uploaded recordings are research data, not clinical data, and are not "
                "used for diagnosis."
            ),
        ]
        manifest.stats.update(
            {
                "eeg_epochs": len(feats),
                "eeg_beats": len(aggregated),
                "eeg_duration_s": round(float(recording.duration_s), 2),
                "eeg_sfreq": round(float(recording.sfreq), 2),
                "eeg_channels": len(recording.channels),
            }
        )
        write_json(target / "manifest.json", manifest.model_dump())

        scenario = {
            "key": f"eeg-{job_id}",
            "label": "Recorded EEG",
            "caption": (
                f"{info['filename']} · {len(feats)} windows → "
                f"{len(aggregated)} beats"
            ),
            "manifest": f"live/eeg-{job_id}/manifest.json",
            "salience_mean": manifest.stats.get("salience_mean"),
            "salience_peak": manifest.stats.get("salience_peak"),
            "splat_mb": manifest.stats.get("splat_mb"),
            "recorded": True,
            "file": info["filename"],
        }
        write_json(
            target / "job.json",
            {
                "request": info,
                "warnings": result.warnings,
                "elapsed_s": result.elapsed_s,
                "n_epochs": len(feats),
                "n_beats": len(aggregated),
            },
        )
        with _eeg_lock:
            job.update(
                state="done",
                stage="done",
                progress=1.0,
                message="Recorded EEG loaded. Only density changed.",
                manifest=scenario["manifest"],
                scenario=scenario,
                warnings=result.warnings,
                elapsed_s=result.elapsed_s,
                n_epochs=len(feats),
                n_beats=len(aggregated),
                duration_s=round(float(recording.duration_s), 2),
                sfreq=round(float(recording.sfreq), 2),
                channels=len(recording.channels),
                finished=time.time(),
            )
        log.info(
            "eeg job %s done in %.1fs: %d windows -> %d beats, %.1f MB",
            job_id,
            job["finished"] - started,
            len(feats),
            len(aggregated),
            float(manifest.stats.get("splat_mb") or 0),
        )
    except Exception as exc:  # noqa: BLE001 - surface the failure to the polling client
        log.exception("eeg job %s failed", job_id)
        shutil.rmtree(target, ignore_errors=True)
        with _eeg_lock:
            job.update(
                state="error",
                stage="error",
                progress=0.0,
                message="The upload could not be processed.",
                error=str(exc),
                finished=time.time(),
            )
    finally:
        _prune_eeg_jobs()


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
        "eeg_jobs": len(_eeg_jobs),
        "public_demo": PUBLIC_DEMO,
        "generation_enabled": (not PUBLIC_DEMO) or bool(ADMIN_TOKEN),
        "eeg_upload": {
            "enabled": True,
            "max_mb": EEG_MAX_BYTES // (1024 * 1024),
            "target_splats": EEG_TARGET_SPLATS,
            "max_active": EEG_MAX_ACTIVE,
            "rate_max": EEG_RATE_MAX,
            "rate_window_s": EEG_RATE_WINDOW_S,
            "epoch_s": EEG_EPOCH_S,
        },
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


@app.post("/api/eeg", status_code=202)
async def upload_eeg(
    request: Request,
    file: UploadFile = File(...),
    sfreq: float = Form(250.0),
    base: str = Form("focus"),
) -> dict:
    """Condition the baked memory volume from a recorded EEG file.

    Public-safe by design: no Meta model call, no image synthesis, no new depth map.
    The uploaded signal only changes the salience field used to re-lift the existing
    keyframe/depth pair. Work runs asynchronously because the re-lift is CPU-bound on
    Render's free plan.
    """
    _check_eeg_access(request)
    base_dir, base_manifest = _base_episode(base)
    raw = await _read_limited(file)
    filename = safe_filename(file.filename)

    try:
        recording = parse_eeg_upload(raw, filename=filename, sfreq=sfreq)
    except (EEGFormatError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc

    step = max(16, int(EEG_EPOCH_S * recording.sfreq))
    usable = recording.signal.shape[1] // step
    if usable < 3:
        minimum = 3 * EEG_EPOCH_S
        raise HTTPException(
            400,
            f"the recording is {recording.duration_s:.2f}s; at least three "
            f"{EEG_EPOCH_S:g}s windows ({minimum:.2f}s) are needed",
        )

    job_id = uuid.uuid4().hex[:10]
    info = {
        "filename": filename,
        "base": base_dir.name,
        "sfreq": round(float(recording.sfreq), 2),
        "channels": len(recording.channels),
        "duration_s": round(float(recording.duration_s), 2),
        "bytes": len(raw),
        "windows": usable,
    }
    job_state = {
        "id": job_id,
        "state": "queued",
        "stage": "queued",
        "progress": 0.0,
        "message": "Queued. The shared keyframe will be re-lifted from your EEG.",
        "created": time.time(),
        "filename": filename,
        "sfreq": info["sfreq"],
        "channels": info["channels"],
        "duration_s": info["duration_s"],
        "base": base_dir.name,
    }
    with _eeg_lock:
        _eeg_jobs[job_id] = job_state
    _eeg_executor.submit(_eeg_runner, job_id, recording, info, base_dir, base_manifest)
    return {"job": job_id, "state": job_state["state"], "key": f"eeg-{job_id}"}


@app.get("/api/eeg/jobs/{job_id}")
def eeg_job(job_id: str) -> dict:
    if job_id not in _eeg_jobs:
        raise HTTPException(404, "unknown EEG job")
    return _eeg_jobs[job_id]


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
