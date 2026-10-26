"""End-to-end CPU pipeline: small episode, paired-condition honesty, cache."""

from __future__ import annotations

import json

import numpy as np

from mns.director.meta_client import MetaModelAPI
from mns.pipeline import PipelineConfig, run_episode, run_pair


def _cfg(**overrides) -> PipelineConfig:
    base = dict(
        width=128,
        height=128,
        target_splats=4_000,
        use_meta=False,
        use_muse_image=False,
        use_sam=False,
        preview=False,
        turntable=False,
        force=True,
        write_ply_frames=1,
        seed=5,
    )
    base.update(overrides)
    return PipelineConfig(**base)


def test_small_episode_is_manifest_complete(tmp_path):
    result = run_episode(tmp_path / "episode", _cfg(), api=MetaModelAPI(api_key=""))
    manifest = result.manifest

    assert len(manifest.frames) >= 3
    assert result.out_dir.joinpath("manifest.json").exists()
    assert result.out_dir.joinpath("keyframe.png").exists()
    assert result.out_dir.joinpath("depth.npy").exists()
    assert manifest.upstream["eeg"] == "synthetic:v1"
    assert "keyframe" in manifest.upstream and "depth" in manifest.upstream

    for frame in manifest.frames:
        assert frame.splat and (result.out_dir / frame.splat).exists()
        assert frame.keyframe and (result.out_dir / frame.keyframe).exists()
        assert frame.attention and (result.out_dir / frame.attention).exists()
        assert frame.n_splats > 0
        # target cap is applied to the lifted surface; the faint environment shell
        # is added afterwards, so the manifest can exceed the target by one shell.
        assert frame.n_splats <= 4_000 + 72 * 28
        assert len(frame.camera.position) == 3
        assert np.isfinite(frame.camera.position).all()
        assert set(frame.object_salience) == {o.id for o in frame.objects}
        assert 0.1 <= frame.epoch.salience <= 1.0

    stats = manifest.stats
    assert stats["n_frames"] == len(manifest.frames)
    assert stats["n_splats_peak"] >= stats["n_splats_min"]
    assert stats["salience_peak"] >= stats["salience_mean"]
    assert stats["calibration_mode"] == "population"


def test_episode_cache_reuses_manifest(tmp_path):
    cfg = _cfg(force=False)
    first = run_episode(tmp_path / "cached", cfg, api=MetaModelAPI(api_key=""))
    assert first.cached is False
    second = run_episode(tmp_path / "cached", cfg, api=MetaModelAPI(api_key=""))
    assert second.cached is True
    assert second.manifest.id == first.manifest.id
    assert second.elapsed_s <= first.elapsed_s + 1.0


def test_pair_shares_graph_keyframe_and_depth(tmp_path):
    payload = run_pair(tmp_path / "pair", _cfg(), api=MetaModelAPI(api_key=""))
    assert [s["key"] for s in payload["scenarios"]] == ["focus", "distracted"]
    assert "share one" in payload["note"].lower()

    focus = json.loads((tmp_path / "pair" / "focus" / "manifest.json").read_text())
    distracted = json.loads((tmp_path / "pair" / "distracted" / "manifest.json").read_text())

    # The scientific control: the image/geometry inputs are byte-identical and the
    # neural condition is the only changed variable.
    focus_kf = (tmp_path / "pair" / "focus" / "keyframe.png").read_bytes()
    distracted_kf = (tmp_path / "pair" / "distracted" / "keyframe.png").read_bytes()
    assert focus_kf == distracted_kf
    assert np.array_equal(
        np.load(tmp_path / "pair" / "focus" / "depth.npy"),
        np.load(tmp_path / "pair" / "distracted" / "depth.npy"),
    )
    assert focus["scene_graph"]["objects"] == distracted["scene_graph"]["objects"]
    assert focus["scene_graph"]["title"] == distracted["scene_graph"]["title"]

    # Only the EEG-derived channel differs.
    assert focus["stats"]["salience_mean"] > distracted["stats"]["salience_mean"]
    assert focus["upstream"]["eeg"] == distracted["upstream"]["eeg"] == "synthetic:v1"
    assert focus["upstream"]["eeg_calibration"] == distracted["upstream"]["eeg_calibration"]


def test_manifest_reports_shared_upstream_for_second_condition(tmp_path):
    run_pair(tmp_path / "pair2", _cfg(), api=MetaModelAPI(api_key=""))
    distracted = json.loads((tmp_path / "pair2" / "distracted" / "manifest.json").read_text())
    assert distracted["upstream"]["scene_graph"] == "shared:paired-condition"
    assert distracted["upstream"]["keyframe"] == "shared between conditions"
    assert distracted["upstream"]["depth"] == "shared"
