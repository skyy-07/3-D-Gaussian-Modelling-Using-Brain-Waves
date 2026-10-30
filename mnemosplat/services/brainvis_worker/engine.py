"""Native BrainVis inference adapter for the separate GPU worker.

This module is intentionally not imported by the CPU web app. It needs a CUDA
machine, the BrainVis checkout, the authors' alignment checkpoint and the Stable
Diffusion 1.5 checkpoint. The code mirrors the reconstruction path in BrainVis'
``cascade_diffusion.py`` but accepts a single batched EEG array instead of the
benchmark dataloader.

Because the checkpoint is request-only, this adapter cannot be exercised in CI. The
worker health endpoint reports exactly which artifact is missing; it never falls back
to random weights or a text-to-image model.
"""

from __future__ import annotations

import io
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

# BrainVis class index -> prompt used by the released cascade. Keep this list ordered
# by the integer class id in the authors' ``lable_number_dict``; do not reorder it.
PROMPTS = [
    "sorrel horse",
    "parachute",
    "smoothing iron",
    "anemone fish",
    "espresso maker",
    "coffee mug",
    "mountain bike, all-terrain bike",
    "revolver, six-shooter",
    "panda",
    "daisy",
    "canoe or kayak",
    "lycaenid butterfly",
    "german shepherd dog",
    "running shoe",
    "Orange jack-o’-lantern",
    "cellular telephone",
    "golf ball",
    "desktop computer",
    "broom",
    "pizza",
    "missile",
    "Cebus capucinus",
    "pool table, billiard table, snooker table",
    "mailbag",
    "old convertible",
    "folding chair",
    "pajama",
    "mitten,glove",
    "electric guitar",
    "eflex camera",
    "grand piano",
    "mountain tent",
    "banana",
    "bolete",
    "digital watch",
    "African elephant",
    "airliner",
    "electric locomotive",
    "radio telescope",
    "cat",
]

WINDOW_SAMPLES = 440
CHANNELS = 128


@dataclass
class GeneratedFrame:
    png: bytes
    predicted_class: int
    label: str
    prompt: str


class BrainVisEngine:
    def __init__(
        self,
        root: str | os.PathLike | None = None,
        checkpoint: str | os.PathLike | None = None,
        sd_checkpoint: str | os.PathLike | None = None,
        device: str = "cuda",
        steps: int = 50,
        stage2_steps: int = 100,
        num_samples: int = 1,
        seed: int = 3407,
    ) -> None:
        self.root = Path(root or os.environ.get("BRAINVIS_ROOT", "")).expanduser()
        self.checkpoint = Path(
            checkpoint
            or os.environ.get("MNS_BRAINVIS_CHECKPOINT", "")
            or (self.root / "exp" / "epilepsy" / "test" / "clipfinetune_model.pkl")
        ).expanduser()
        self.sd_checkpoint = Path(
            sd_checkpoint
            or os.environ.get("MNS_BRAINVIS_SD_CKPT", "")
            or (self.root / "pretrained_model" / "v1-5-pruned-emaonly.ckpt")
        ).expanduser()
        self.config_path = self.root / "pretrained_model" / "config15.yaml"
        self.device = device
        self.steps = max(10, int(steps))
        self.stage2_steps = max(10, int(stage2_steps))
        self.num_samples = max(1, int(num_samples))
        self.seed = int(seed)
        self._lock = threading.Lock()
        self._loaded = False
        self._alignnet = None
        self._timefreq = None
        self._ldm = None
        self._sampler = None
        self._img2img = None
        self._torch = None
        self._load_error: str | None = None

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        reason = None
        cuda = False
        try:
            import torch

            cuda = bool(torch.cuda.is_available())
        except Exception as exc:  # noqa: BLE001
            reason = f"torch is not importable: {exc}"

        checkpoint = self.checkpoint.is_file()
        sd_checkpoint = self.sd_checkpoint.is_file()
        high_level = (
            self.root.is_dir()
            and (self.root / "model" / "BrainVisModels.py").is_file()
            and (self.root / "dc_ldm" / "models" / "diffusion" / "plms.py").is_file()
            and self.config_path.is_file()
        )
        if not high_level:
            reason = f"BrainVis checkout or config15.yaml missing under {self.root}"
        elif not checkpoint:
            reason = f"BrainVis checkpoint missing: {self.checkpoint}"
        elif not sd_checkpoint:
            reason = f"Stable Diffusion 1.5 checkpoint missing: {self.sd_checkpoint}"
        elif not cuda:
            reason = "CUDA is not available to the worker process"
        elif self._load_error:
            reason = self._load_error
        ready = bool(high_level and checkpoint and sd_checkpoint and cuda and not self._load_error)
        return {
            "ready": ready,
            "reason": reason or ("ready" if ready else "worker is not ready"),
            "backend": "brainvis",
            "cuda": cuda,
            "checkpoint": checkpoint,
            "stable_diffusion": sd_checkpoint,
            "brainvis_root": str(self.root),
            "version": "0.1",
        }

    # ------------------------------------------------------------------- load
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            try:
                self._load()
                self._loaded = True
                self._load_error = None
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"{type(exc).__name__}: {exc}"
                raise

    def _load(self) -> None:
        import torch
        from diffusers import StableDiffusionImg2ImgPipeline
        from omegaconf import OmegaConf

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the BrainVis worker")
        if not self.root.is_dir():
            raise FileNotFoundError(f"BrainVis root not found: {self.root}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"BrainVis checkpoint not found: {self.checkpoint}")
        if not self.sd_checkpoint.is_file():
            raise FileNotFoundError(f"Stable Diffusion checkpoint not found: {self.sd_checkpoint}")

        root = str(self.root)
        if root not in sys.path:
            sys.path.insert(0, root)

        from dc_ldm.models.diffusion.plms import PLMSSampler
        from dc_ldm.util import instantiate_from_config
        from model.BrainVisModels import AlignNet, FreqEncoder, TimeEncoder, TimeFreqEncoder

        args = SimpleNamespace(
            d_model=1024,
            dropout=0.2,
            attn_heads=16,
            enable_res_parameter=1,
            layers=8,
            momentum=0.99,
            vocab_size=660,
            wave_length=4,
            mask_ratio=0.75,
            reg_layers=4,
            num_class=40,
            data_shape=(WINDOW_SAMPLES, CHANNELS),
            device=self.device,
        )

        time_model = TimeEncoder(args)
        freq_model = FreqEncoder()
        timefreq = TimeFreqEncoder(time_model, freq_model, args)
        alignnet = AlignNet(1024, 128, 77 * 768, timefreq)

        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {str(k).removeprefix("module."): v for k, v in state.items()}
        missing, unexpected = alignnet.load_state_dict(state, strict=False)
        if missing or unexpected:
            # A partial load is almost always the wrong checkpoint. Refuse instead of
            # producing plausible-looking garbage from randomly initialized layers.
            raise RuntimeError(
                "checkpoint does not match AlignNet: "
                f"{len(missing)} missing / {len(unexpected)} unexpected keys; "
                f"first missing={missing[:3]}, first unexpected={unexpected[:3]}"
            )
        alignnet.to(self.device).eval()

        config = OmegaConf.load(str(self.config_path))
        config.model.params.unet_config.params.use_time_cond = False
        config.model.params.unet_config.params.global_pool = True
        ddim_steps = 50
        ldm = instantiate_from_config(config.model)
        sd_state = torch.load(
            self.sd_checkpoint, map_location="cpu", weights_only=False
        )["state_dict"]
        ldm.load_state_dict(sd_state, strict=False)
        ldm.cond_stage_trainable = False
        ldm.ddim_steps = ddim_steps
        ldm.re_init_ema()
        ldm.p_channels = config.model.params.channels
        ldm.p_image_size = config.model.params.image_size
        ldm.ch_mult = config.model.params.first_stage_config.params.ddconfig.ch_mult
        ldm.clip_tune = False
        ldm.cls_tune = False
        ldm.to(self.device).eval()

        sampler = PLMSSampler(ldm)

        img2img = StableDiffusionImg2ImgPipeline.from_single_file(
            str(self.sd_checkpoint),
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        img2img.enable_attention_slicing()
        img2img.to(self.device)

        self._torch = torch
        self._alignnet = alignnet
        self._timefreq = timefreq
        self._ldm = ldm
        self._sampler = sampler
        self._img2img = img2img
        self._ddim_steps = ddim_steps
        self._shape = (4, 64, 64)

    # -------------------------------------------------------------- inference
    def reconstruct(
        self,
        window: np.ndarray,
        *,
        prompt: str | None = None,
        seed: int | None = None,
    ) -> list[GeneratedFrame]:
        self._ensure_loaded()
        torch = self._torch
        arr = np.asarray(window, dtype=np.float32)
        if arr.shape == (CHANNELS, WINDOW_SAMPLES):
            arr = arr.T
        if arr.shape != (WINDOW_SAMPLES, CHANNELS):
            raise ValueError(
                f"BrainVis expects ({WINDOW_SAMPLES}, {CHANNELS}) after preprocessing; "
                f"got {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise ValueError("EEG window contains NaN or infinite values")

        x = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            clip_encoded = self._alignnet(x)
            _, _, scores = self._timefreq(x)
            class_idx = int(scores.argmax(dim=1)[0].item())
            label = PROMPTS[class_idx]
            use_prompt = (prompt or "").strip() or label

            latent = clip_encoded[0].reshape(1, 77, 768).repeat(self.num_samples, 1, 1)
            samples, _ = self._sampler.sample(
                S=self.steps,
                conditioning=latent,
                batch_size=self.num_samples,
                shape=self._shape,
                verbose=False,
            )
            stage1 = self._ldm.decode_first_stage(samples)
            stage1 = torch.clamp((stage1 + 1.0) / 2.0, 0.0, 1.0)
            arrays = (255.0 * stage1.detach().cpu().numpy()).astype(np.uint8)
            images = [
                Image.fromarray(np.transpose(item, (1, 2, 0)))
                for item in arrays
            ]

            generator = torch.Generator(device=self.device).manual_seed(
                self.seed if seed is None else int(seed)
            )
            generated = self._img2img(
                [use_prompt] * self.num_samples,
                image=images,
                strength=0.85,
                guidance_scale=7.5,
                num_inference_steps=self.stage2_steps,
                generator=generator,
            ).images

            out: list[GeneratedFrame] = []
            for image in generated:
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                out.append(
                    GeneratedFrame(
                        png=buf.getvalue(),
                        predicted_class=class_idx,
                        label=label,
                        prompt=use_prompt,
                    )
                )
            return out


__all__ = ["BrainVisEngine", "GeneratedFrame", "PROMPTS", "CHANNELS", "WINDOW_SAMPLES"]
