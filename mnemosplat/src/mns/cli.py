"""Command line entry point: ``mnemosplat <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .util import configure_logging


def _print_json(payload) -> None:
    print(json.dumps(payload, indent=2))


def cmd_demo(args) -> int:
    from .director.meta_client import MetaModelAPI
    from .pipeline import PipelineConfig, run_episode, run_pair
    from .scenarios import SCENARIOS

    configure_logging(args.log)
    api = MetaModelAPI()
    cfg = PipelineConfig(
        width=args.width,
        height=args.height,
        target_splats=args.splats,
        use_meta=not args.offline,
        use_muse_image=args.muse,
        use_sam=not args.no_sam,
        force=args.force,
        preview=not args.no_preview,
        turntable=not args.no_preview,
    )
    out = Path(args.out)
    if args.scenario == "both":
        payload = run_pair(out, cfg, keys=("focus", "distracted"), api=api)
        _print_json(payload)
        print(f"\n  open: http://{args.host}:{args.port}  (run `mnemosplat serve`)")
        return 0

    if args.scenario not in SCENARIOS:
        print(f"unknown scenario {args.scenario!r}; choose: {', '.join(SCENARIOS)} or both")
        return 2
    cfg.scenario = args.scenario
    res = run_episode(out, cfg, narrative=args.narrative, api=api)
    m = res.manifest
    print(f"\nMnemoSplat · {m.title}  [{m.scenario}]")
    print(f"  frames      {len(m.frames)}")
    print(
        f"  splats      {m.stats['n_splats_peak']:,} peak / "
        f"{m.stats['n_splats_min']:,} min"
    )
    print(
        f"  salience    mean {m.stats['salience_mean']:.2f} · "
        f"peak {m.stats['salience_peak']:.2f}"
    )
    print(f"  size        {m.stats['splat_mb']} MB of .splat data")
    print(f"  upstream    {', '.join(f'{k}={v}' for k, v in m.upstream.items())}")
    print(f"  manifest    {res.out_dir / 'manifest.json'}")
    if res.warnings:
        print("  warnings:")
        for w in res.warnings:
            print(f"    - {w}")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .api import app  # noqa: F401 - imported for the uvicorn factory path

    configure_logging(args.log)
    uvicorn.run(
        "mns.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log.lower(),
    )
    return 0


def cmd_selftest(args) -> int:
    from .pipeline import selftest

    result = selftest(args.out)
    _print_json(result)
    return 0 if result.get("ok") else 1


def cmd_doctor(args) -> int:
    from .depth import DepthEstimator
    from .director.meta_client import MetaModelAPI
    from .eeg.brainvis import resolve_brainvis

    api = MetaModelAPI()
    checks: dict = {
        "python": sys.version.split()[0],
        "model_api_key": bool(api.available),
        "base_url": api.base_url,
        "models": {
            "text": os.environ.get("MNS_TEXT_MODEL", "muse-spark-1.3"),
            "image": os.environ.get("MNS_IMAGE_MODEL", "muse-image-1.0"),
            "segmentation": os.environ.get("MNS_SEG_MODEL", "sam-3.1"),
            "asr": os.environ.get("MNS_ASR_MODEL", "muse-voice-transcribe-1.0"),
        },
        "depth_learned": DepthEstimator().learned_available,
        "brainvis": resolve_brainvis(),
        "web_dist": Path("web/dist/index.html").exists(),
    }
    if args.probe and api.available:
        try:
            resp = api._request("GET", "/models")
            models = resp.json()
            checks["model_probe"] = [
                m.get("id") or m.get("name")
                for m in (models.get("data") or models.get("models") or [])
            ][:20]
        except Exception as exc:  # noqa: BLE001
            checks["model_probe_error"] = str(exc)
    _print_json(checks)

    ready = checks["model_api_key"]
    print("\n" + ("Meta Model API key detected - full pipeline enabled." if ready else
                  "No MODEL_API_KEY: running in offline mode (recipes + procedural keyframe)."))
    print("Install learned depth with:  uv pip install -e '.[depth]'")
    print("Build the headset client with:  cd web && npm install && npm run build")
    return 0


def cmd_export(args) -> int:
    """Re-export a .splat as .ply (or vice versa) without rerunning the pipeline."""
    import numpy as np

    from .export.ply import read_splat, write_ply
    from .geometry.gaussians import GaussianCloud

    src = Path(args.input)
    data = read_splat(src)
    cloud = GaussianCloud(
        xyz=data["xyz"],
        scales=data["scales"],
        quats=data["quats"],
        colors=data["colors"],
        opacity=data["opacity"],
        bounds={
            "min": data["xyz"].min(axis=0).round(4).tolist(),
            "max": data["xyz"].max(axis=0).round(4).tolist(),
        },
    )
    dst = Path(args.output or src.with_suffix(".ply"))
    if dst.suffix == ".ply":
        write_ply(dst, cloud)
    else:
        from .export.ply import write_splat

        write_splat(dst, cloud)
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.2f} MB, {len(cloud):,} splats)")
    _ = np
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mnemosplat",
        description="EEG-salience conditioned 3D Gaussian memory volumes for WebXR",
    )
    p.add_argument("--log", default="INFO", help="log level")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="render an episode (or the focus/distracted pair)")
    d.add_argument("--scenario", default="focus", help="focus | distracted | street | both")
    d.add_argument("--narrative", default=None, help="custom transcript (overrides scenario text)")
    d.add_argument("--out", default="out/demo")
    d.add_argument("--width", type=int, default=512)
    d.add_argument("--height", type=int, default=512)
    d.add_argument("--splats", type=int, default=120_000)
    d.add_argument("--muse", action="store_true", help="use muse-image-1.0 for the keyframe")
    d.add_argument("--no-sam", action="store_true", help="skip SAM 3.1 box refinement")
    d.add_argument("--offline", action="store_true", help="never call the Meta API")
    d.add_argument("--force", action="store_true", help="ignore the cache")
    d.add_argument("--no-preview", action="store_true")
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--port", type=int, default=8000)
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("serve", help="run the HTTP API + web client")
    s.add_argument("--host", default=os.environ.get("MNS_HOST", "0.0.0.0"))
    s.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MNS_PORT", os.environ.get("PORT", "8000"))),
    )
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)

    t = sub.add_parser("selftest", help="CPU-only end-to-end smoke test")
    t.add_argument("--out", default="out/selftest")
    t.set_defaults(func=cmd_selftest)

    doc = sub.add_parser("doctor", help="environment and capability report")
    doc.add_argument("--probe", action="store_true", help="call GET /models to verify the key")
    doc.set_defaults(func=cmd_doctor)

    e = sub.add_parser("export", help="convert .splat <-> .ply")
    e.add_argument("input")
    e.add_argument("output", nargs="?")
    e.set_defaults(func=cmd_export)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
