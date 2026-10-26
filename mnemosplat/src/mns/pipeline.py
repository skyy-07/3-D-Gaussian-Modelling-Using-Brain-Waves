"""End-to-end orchestration: transcript + EEG -> WebXR-ready memory volume.

Deliberate product decision
---------------------------
We synthesise **one** keyframe per episode and re-lift it once per EEG epoch, changing
only the cognitive density. The timeline therefore scrubs *consolidation*, not camera
motion. Consequences:

* the geometry never tears between timeline frames (the #1 way these demos look fake);
* Muse Image is called once per memory ($0.01) instead of once per frame;
* the only thing that changes across the slider is exactly the thing we claim the EEG
  controls - density, opacity and spread.

Set ``per_beat_keyframes=True`` for the experimental multi-keyframe mode.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .depth import DepthEstimator
from .director.meta_client import MetaModelAPI
from .director.scene_graph import build_scene_graph, refine_with_sam
from .eeg import EEGRecording, analyze_epochs, generate_eeg
from .export.ply import export_cloud
from .fusion.attention import (
    assign_object_salience,
    build_attention_field,
    colorize,
    object_summary,
    overlay,
)
from .geometry.gaussians import GaussianCloud, GaussianConfig, init_from_frame
from .geometry.preview import rasterize, turntable
from .render import KeyframeError, KeyframeResult, muse_keyframe, procedural_keyframe
from .render.procedural import depth_preview
from .scenarios import Scenario, get_scenario
from .schemas import CameraModel, FrameManifest, Manifest
from .util import configure_logging, read_json, stable_hash, write_json

log = logging.getLogger("mns.pipeline")


@dataclass
class PipelineConfig:
    width: int = 512
    height: int = 512
    hfov_deg: float = 60.0
    epoch_s: float = 0.75
    target_splats: int = 120_000
    hard_cap: int = 262_144
    scenario: str = "focus"
    subject: str = "synthetic-subject"
    calibration: str = "population"
    use_meta: bool = True
    use_muse_image: bool = False
    use_sam: bool = True
    per_beat_keyframes: bool = False
    write_ply_frames: int = 1
    preview: bool = True
    turntable: bool = True
    force: bool = False
    seed: int = 7
    #: knobs that must change the output hash
    version: int = 3

    def cache_key(self, narrative: str) -> str:
        return stable_hash(
            narrative,
            self.scenario,
            self.width,
            self.height,
            self.target_splats,
            self.use_muse_image,
            self.per_beat_keyframes,
            self.seed,
            self.version,
        )


@dataclass
class EpisodeResult:
    manifest: Manifest
    out_dir: Path
    cloud: GaussianCloud | None = None
    elapsed_s: float = 0.0
    cached: bool = False
    warnings: list[str] = field(default_factory=list)


def _save_png(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, optimize=True)


def _b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def run_episode(
    out_dir: str | Path,
    cfg: PipelineConfig | None = None,
    narrative: str | None = None,
    recording: EEGRecording | None = None,
    scenario: Scenario | None = None,
    api: MetaModelAPI | None = None,
    graph_override=None,
    keyframe_override: str | Path | None = None,
    depth_override: str | Path | None = None,
) -> EpisodeResult:
    cfg = cfg or PipelineConfig()
    started = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    scen = scenario or get_scenario(cfg.scenario)
    narrative = (narrative or scen.narrative).strip()
    manifest_path = out / "manifest.json"
    key = cfg.cache_key(narrative) + "-" + stable_hash(scen.focus, scen.lateralization, scen.seed)

    entry = read_json(out / ".episode.json") if (out / ".episode.json").exists() else {}
    if manifest_path.exists() and entry.get("key") == key and not cfg.force:
        log.info("reusing cached episode at %s", out)
        return EpisodeResult(
            manifest=Manifest(**read_json(manifest_path)),
            out_dir=out,
            elapsed_s=round(time.perf_counter() - started, 3),
            cached=True,
        )

    api = api or MetaModelAPI()
    notes: list[str] = []
    upstream: dict[str, str] = {}

    # ---------------------------------------------------------------- 1. director
    director = build_scene_graph(
        narrative,
        api=api if (cfg.use_meta and api.available) else None,
        width=cfg.width,
        height=cfg.height,
    )
    graph = graph_override or director.graph
    upstream["scene_graph"] = (
        f"meta:{director.model}" if director.used_model else "heuristic:recipe"
    )
    if keyframe_override is not None:
        upstream["scene_graph"] = "shared:paired-condition"
    if director.warning:
        warnings.append(director.warning)
    if graph.source == "heuristic":
        notes.append(
            "Scene graph authored offline (no MODEL_API_KEY or model unavailable); "
            "set MODEL_API_KEY to let Muse Spark parse the transcript."
        )

    # -------------------------------------------------------------------- 2. EEG
    n_epochs = max(3, len(graph.beats))
    if recording is None:
        rec = generate_eeg(
            duration_s=n_epochs * cfg.epoch_s,
            epoch_s=cfg.epoch_s,
            focus_schedule=np.array(scen.focus),
            lateralization_schedule=np.array(scen.lateralization),
            seed=scen.seed,
        )
        upstream["eeg"] = "synthetic:v1"
        notes.append(
            "EEG is synthetic (mns.eeg.synthetic): spectrally realistic, but not recorded. "
            "Swap in THINGS-EEG2 / BrainVis features for a live study."
        )
    else:
        rec = recording
        upstream["eeg"] = "file" if not rec.synthetic else "synthetic:v1"

    feats, calibrator, band_table = analyze_epochs(
        rec,
        epoch_s=cfg.epoch_s,
        calibration=cfg.calibration,
    )
    upstream["eeg_calibration"] = f"{calibrator.mode}:center={calibrator.center:.3f}"

    # ------------------------------------------------------ 3. keyframe + depth
    camera = CameraModel.from_fov(cfg.width, cfg.height, cfg.hfov_deg)
    kf: KeyframeResult
    render = None
    use_muse = cfg.use_muse_image and api.available
    t0 = time.perf_counter()
    if keyframe_override is not None and Path(keyframe_override).exists():
        img = Image.open(keyframe_override).convert("RGB")
        kf = KeyframeResult(
            image=img,
            provider="shared",
            prompt="(shared keyframe for paired-condition comparison)",
            meta={"shared_from": str(keyframe_override)},
        )
        upstream["keyframe"] = "shared between conditions"
    elif use_muse:
        try:
            kf = muse_keyframe(graph, api, cfg.width, cfg.height, beat_label=graph.beats[0].label)
            upstream["keyframe"] = "muse-image-1.0"
        except KeyframeError as exc:  # pragma: no cover - network dependent
            warnings.append(f"Muse Image failed ({exc}); using procedural keyframe")
            kf, render = procedural_keyframe(graph, cfg.width, cfg.height, seed=cfg.seed)
            upstream["keyframe"] = "procedural(v1) [muse fallback]"
    else:
        kf, render = procedural_keyframe(graph, cfg.width, cfg.height, seed=cfg.seed)
        upstream["keyframe"] = "procedural(v1)"
        notes.append(
            "Keyframe rendered procedurally from the scene graph. Enable MNS_USE_MUSE_IMAGE=1 "
            "with MODEL_API_KEY for muse-image-1.0 synthesis."
        )
    keyframe_ms = round((time.perf_counter() - t0) * 1000)
    _save_png(kf.image, out / "keyframe.png")

    estimator = DepthEstimator()
    if depth_override is not None and Path(depth_override).exists():
        from .depth import DepthResult

        depth_res = DepthResult(
            depth=np.load(depth_override).astype(np.float32),
            source="shared",
            meta={"shared_from": str(depth_override)},
        )
    else:
        depth_res = estimator.estimate(
            kf.image, cfg.width, cfg.height, oracle=(render.depth if render else None)
        )
    upstream["depth"] = depth_res.source
    np.save(out / "depth.npy", depth_res.depth)
    _save_png(depth_preview(depth_res.depth), out / "depth.png")

    # -------------------------------------------------------- 4. masks (SAM 3.1)
    masks: dict[str, np.ndarray] = dict(render.masks) if render else {}
    sam_diag: list[dict] = []
    if use_muse and cfg.use_sam and api.available and not masks:
        try:
            graph, masks, sam_diag = refine_with_sam(graph, api, _b64_png(kf.image))
            upstream["segmentation"] = "sam-3.1"
        except Exception as exc:  # noqa: BLE001 - non-fatal by design
            warnings.append(f"SAM refine skipped: {exc}")
            upstream["segmentation"] = "box-fallback"
    elif masks:
        upstream["segmentation"] = "scene-graph masks (exact)"

    # ------------------------------------------------------------- 5. salience
    per_epoch = assign_object_salience(graph, feats)
    gcfg = GaussianConfig(
        target_splats=cfg.target_splats,
        hard_cap=cfg.hard_cap,
        seed=cfg.seed,
    )

    frames: list[FrameManifest] = []
    clouds: list[GaussianCloud] = []
    attention_dir = out / "frames"
    attention_dir.mkdir(parents=True, exist_ok=True)

    # The lifter sees a 3x3 median-filtered copy of the keyframe. Procedural grain is
    # lovely in the photo and terrible as per-splat colour noise; a median removes
    # speckle while keeping window bars and object edges intact.
    lift_rgb = np.asarray(
        kf.image.convert("RGB")
        .filter(ImageFilter.MedianFilter(3))
        .filter(ImageFilter.GaussianBlur(0.55)),
        dtype=np.uint8,
    )
    for i, (ep, sal) in enumerate(zip(feats, per_epoch, strict=True)):
        beat = graph.beats[min(i, len(graph.beats) - 1)]
        field = build_attention_field(
            graph,
            sal,
            ep,
            shape=(cfg.height, cfg.width),
            masks=masks or None,
        )
        cloud = init_from_frame(
            lift_rgb,
            depth_res.depth,
            field,
            camera,
            cfg=gcfg,
            global_salience=ep.salience,
        )
        stem = f"{i:03d}"
        exported = export_cloud(cloud, attention_dir, stem)
        if i < cfg.write_ply_frames:
            pass
        else:
            (attention_dir / f"{stem}.ply").unlink(missing_ok=True)
        _save_png(colorize(field), attention_dir / f"{stem}.attention.png")
        if i == 0:
            _save_png(overlay(kf.image, field, 0.5), out / "keyframe.attention.png")

        cam = cloud.camera_position
        frames.append(
            FrameManifest(
                index=i,
                t=round(ep.t0, 3),
                label=f"{beat.label} · {ep.t0:.2f}s",
                narrative=beat.narrative,
                splat=f"frames/{exported['splat']}",
                ply=f"frames/{exported['ply']}" if i < cfg.write_ply_frames else None,
                keyframe="keyframe.png",
                attention=f"frames/{stem}.attention.png",
                depth_preview="depth.png",
                camera=camera.model_copy(
                    update={
                        "position": [round(float(v), 4) for v in cam],
                        "look_at": [0.0, 0.0, 0.0],
                    }
                ),
                epoch=ep,
                n_splats=exported["n_splats"],
                object_salience=sal,
                objects=graph.objects,
                depth_source=depth_res.source,
                bounds=exported["bounds"],
            )
        )
        clouds.append(cloud)

    # ------------------------------------------------------------------ 6. QA renders
    if cfg.preview and clouds:
        sals = [f.epoch.salience for f in frames]
        lo_idx, hi_idx = int(np.argmin(sals)), int(np.argmax(sals))
        try:
            combo = Image.new("RGB", (1024, 512))
            from PIL import ImageDraw

            for slot, idx in enumerate((lo_idx, hi_idx)):
                img = rasterize(clouds[idx], size=(512, 512), max_splats=110_000)
                combo.paste(img, (512 * slot, 0))
                d = ImageDraw.Draw(combo)
                label = f"{frames[idx].label}   salience {sals[idx]:.2f}"
                d.rectangle((512 * slot + 8, 8, 512 * slot + 300, 30), fill=(8, 9, 12))
                d.text((512 * slot + 14, 14), label, fill=(235, 230, 220))
            _save_png(combo, out / "preview.png")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"preview render failed: {exc}")
    if cfg.turntable and clouds:
        try:
            best = int(np.argmax([f.epoch.salience for f in frames]))
            turntable(clouds[best], out / "turntable", views=5)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"turntable failed: {exc}")

    # --------------------------------------------------------------- 7. manifest
    summary_obj = object_summary(graph, per_epoch[int(np.argmax([e.salience for e in feats]))])
    total_bytes = sum(
        (attention_dir / Path(f.splat).name).stat().st_size for f in frames if f.splat
    )
    manifest = Manifest(
        id=key,
        title=graph.title,
        subject=cfg.subject,
        scenario=scen.key,
        scene_graph=graph,
        frames=frames,
        notes=notes,
        upstream=upstream,
        stats={
            "n_frames": len(frames),
            "n_splats_peak": max(f.n_splats for f in frames),
            "n_splats_min": min(f.n_splats for f in frames),
            "splat_mb": round(total_bytes / (1024 * 1024), 2),
            "salience_mean": round(float(np.mean([e.salience for e in feats])), 3),
            "salience_peak": round(float(max(e.salience for e in feats)), 3),
            "focus_index_mean": round(float(np.mean([e.focus_index for e in feats])), 3),
            "gamma_theta_mean": round(band_table["gamma_theta_mean"], 3),
            "calibration_mode": calibrator.mode,
            "keyframe_ms": keyframe_ms,
            "director_used_model": int(director.used_model),
        },
        objects=summary_obj,
        world_scale_m=1.0,
    )
    write_json(manifest_path, manifest.model_dump())
    write_json(out / ".episode.json", {"key": key, "cfg": asdict(cfg), "warnings": warnings})

    elapsed = time.perf_counter() - started
    log.info(
        "episode done in %.1fs: %d frames, peak %s splats, %.1f MB",
        elapsed,
        len(frames),
        f"{manifest.stats['n_splats_peak']:,}",
        manifest.stats["splat_mb"],
    )
    return EpisodeResult(
        manifest=manifest,
        out_dir=out,
        cloud=clouds[int(np.argmax([f.epoch.salience for f in frames]))] if clouds else None,
        elapsed_s=round(elapsed, 2),
        warnings=warnings,
    )


def run_pair(
    base_dir: str | Path,
    cfg: PipelineConfig | None = None,
    keys: tuple[str, ...] = ("focus", "distracted"),
    api: MetaModelAPI | None = None,
) -> dict:
    """Render the two-condition comparison plus an index the web client can read."""
    cfg = cfg or PipelineConfig()
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    api = api or MetaModelAPI()
    index = []
    shared_narrative = get_scenario(keys[0]).narrative
    shared_graph = None
    shared_keyframe: Path | None = None
    shared_depth: Path | None = None
    for k in keys:
        c = PipelineConfig(**{**asdict(cfg), "scenario": k})
        # identical scene graph, keyframe and depth across conditions; only EEG differs
        res = run_episode(
            base / k,
            c,
            narrative=shared_narrative,
            api=api,
            graph_override=shared_graph,
            keyframe_override=shared_keyframe,
            depth_override=shared_depth,
        )
        if shared_graph is None:
            shared_graph = res.manifest.scene_graph
            shared_keyframe = res.out_dir / "keyframe.png"
            shared_depth = res.out_dir / "depth.npy"
        index.append(
            {
                "key": k,
                "label": get_scenario(k).label,
                "caption": get_scenario(k).caption,
                "manifest": f"{k}/manifest.json",
                "salience_mean": res.manifest.stats["salience_mean"],
                "salience_peak": res.manifest.stats["salience_peak"],
                "splat_mb": res.manifest.stats["splat_mb"],
                "cached": res.cached,
            }
        )
    payload = {
        "title": "MnemoSplat · same words, different memory",
        "generated": time.time(),
        "scenarios": index,
        "note": "Both conditions share one narrative, one scene graph and one keyframe.",
    }
    write_json(base / "index.json", payload)
    return payload


def selftest(tmp_dir: str | Path | None = None) -> dict:
    """Fast CPU-only smoke test used by `mnemosplat selftest` and CI."""
    configure_logging("WARNING")
    out = Path(tmp_dir or "out/selftest")
    cfg = PipelineConfig(
        width=256,
        height=256,
        target_splats=18_000,
        preview=True,
        turntable=False,
        force=True,
        use_meta=False,
        write_ply_frames=1,
    )
    res = run_episode(out, cfg, api=MetaModelAPI(api_key=""))
    m = res.manifest
    return {
        "ok": True,
        "frames": len(m.frames),
        "splats": [f.n_splats for f in m.frames],
        "salience": [f.epoch.salience for f in m.frames],
        "splat_mb": m.stats["splat_mb"],
        "out": str(out),
        "upstream": m.upstream,
    }
