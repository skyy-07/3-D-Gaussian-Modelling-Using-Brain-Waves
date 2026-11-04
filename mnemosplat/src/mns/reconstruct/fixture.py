"""Deterministic scripted-demo reconstruction provider.

This module deliberately does **not** decode EEG. It exists so the upload -> job ->
gallery path can be demonstrated with a fixed, repeatable illustration when the real
BrainVis checkpoint is unavailable. Every frame carries a visible watermark and the
API/UI label the provider as a scripted demo fixture, never as reconstruction.
"""

from __future__ import annotations

import io
import os
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPTED_BACKEND = "scripted-demo-fixture"
SCRIPTED_VERSION = "1.0.0"
SCRIPTED_LABEL = "running shoe"
SCRIPTED_CLASS = 13
SCRIPTED_NOTE = (
    "Scripted demo fixture: fixed illustration frames returned for the bundled "
    "128-channel demo EEG. This is not EEG reconstruction, not a decoded memory, "
    "and not evidence of what a person saw."
)


def _env_label() -> str:
    return (os.environ.get("MNS_FIXTURE_LABEL") or SCRIPTED_LABEL).strip() or SCRIPTED_LABEL


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _background() -> Image.Image:
    width = height = 512
    image = Image.new("RGB", (width, height), (9, 13, 21))
    draw = ImageDraw.Draw(image)
    # Vertical gradient with a faint horizon glow.
    for y in range(height):
        t = y / (height - 1)
        r = int(9 + 16 * (1.0 - t) ** 2)
        g = int(13 + 24 * (1.0 - t) ** 2)
        b = int(21 + 34 * (1.0 - t) ** 2)
        draw.line((0, y, width, y), fill=(r, g, b))
    for x in range(0, width, 32):
        draw.line((x, 0, x, height), fill=(24, 31, 44), width=1)
    for y in range(0, height, 32):
        draw.line((0, y, width, y), fill=(24, 31, 44), width=1)
    draw.ellipse((70, 330, 442, 445), fill=(20, 27, 39))
    return image


def _draw_shoe(image: Image.Image, index: int, total: int) -> None:
    """Draw a stylised running shoe with a small per-frame bounce and tilt."""
    phase = (index / max(1, total)) * 2.0 * np.pi
    bounce = int(round(10.0 * np.sin(phase)))
    tilt = 4.0 * np.sin(phase + 0.6)

    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx, cy = 256, 286 + bounce

    # Motion lines behind the shoe.
    for offset, length, alpha in ((0, 92, 90), (26, 68, 70), (52, 48, 50)):
        y = cy + 38 + offset
        draw.line(
            (72, y, 72 + length, y),
            fill=(233, 178, 95, alpha),
            width=5,
        )

    # Ground shadow.
    draw.ellipse((cx - 148, cy + 92, cx + 148, cy + 128), fill=(0, 0, 0, 110))

    # Sole.
    draw.rounded_rectangle(
        (cx - 154, cy + 42, cx + 162, cy + 86),
        radius=22,
        fill=(28, 33, 44, 255),
        outline=(233, 178, 95, 255),
        width=3,
    )
    draw.rounded_rectangle(
        (cx - 146, cy + 74, cx + 154, cy + 88),
        radius=7,
        fill=(58, 65, 80, 255),
    )

    # Upper: heel, collar, tongue, instep and toe.
    upper = [
        (cx - 148, cy + 48),
        (cx - 156, cy - 8),
        (cx - 132, cy - 54),
        (cx - 96, cy - 72),
        (cx - 58, cy - 70),
        (cx - 34, cy - 48),
        (cx - 16, cy - 76),
        (cx + 22, cy - 68),
        (cx + 70, cy - 42),
        (cx + 124, cy - 10),
        (cx + 160, cy + 24),
        (cx + 162, cy + 48),
        (cx + 120, cy + 58),
        (cx - 80, cy + 58),
    ]
    draw.polygon(upper, fill=(232, 232, 238, 255))
    draw.line(upper + [upper[0]], fill=(15, 19, 28, 255), width=3, joint="curve")

    # Heel counter and collar.
    draw.ellipse((cx - 158, cy - 38, cx - 102, cy + 14), fill=(198, 204, 214, 255))
    draw.arc((cx - 158, cy - 38, cx - 102, cy + 14), 90, 270, fill=(15, 19, 28, 255), width=3)
    draw.ellipse((cx - 98, cy - 74, cx - 52, cy - 36), fill=(244, 244, 248, 255))
    draw.arc((cx - 98, cy - 74, cx - 52, cy - 36), 180, 360, fill=(15, 19, 28, 255), width=3)

    # Tongue.
    draw.polygon(
        [
            (cx - 54, cy - 58),
            (cx - 18, cy - 78),
            (cx + 12, cy - 58),
            (cx + 2, cy - 30),
            (cx - 34, cy - 28),
        ],
        fill=(206, 212, 222, 255),
    )

    # Accent stripe.
    draw.line(
        (cx - 120, cy + 8, cx + 128, cy + 36),
        fill=(233, 178, 95, 255),
        width=10,
    )
    draw.line(
        (cx - 120, cy + 8, cx + 128, cy + 36),
        fill=(255, 220, 150, 255),
        width=3,
    )

    # Laces.
    for i in range(4):
        x = cx - 30 + i * 26
        y = cy - 52 + i * 11
        draw.line((x, y, x + 20, y + 10), fill=(15, 19, 28, 255), width=3)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(233, 178, 95, 255))

    # Toe cap.
    draw.arc(
        (cx + 96, cy - 10, cx + 166, cy + 58),
        270,
        90,
        fill=(15, 19, 28, 255),
        width=3,
    )

    layer = layer.rotate(tilt, resample=Image.Resampling.BICUBIC, center=(cx, cy))
    image.paste(layer, (0, 0), layer)


def _draw_labels(image: Image.Image, index: int, total: int, label: str) -> None:
    draw = ImageDraw.Draw(image)
    title_font = _font(34, bold=True)
    sub_font = _font(15, bold=True)
    tiny_font = _font(13)
    badge_font = _font(13, bold=True)

    draw.text((28, 24), label.upper(), font=title_font, fill=(244, 239, 226))
    draw.text((30, 64), "SCRIPTED DEMO FIXTURE", font=sub_font, fill=(233, 178, 95))
    draw.text(
        (30, 84),
        "NOT EEG RECONSTRUCTION",
        font=tiny_font,
        fill=(255, 145, 120),
    )

    # Watermark band.
    draw.rounded_rectangle((24, 452, 488, 488), radius=10, fill=(10, 14, 22))
    draw.text(
        (38, 460),
        "FIXED ILLUSTRATION · NOT A DECODED MEMORY",
        font=tiny_font,
        fill=(220, 224, 232),
    )

    # Frame badge.
    badge = f"FRAME {index + 1:02d}/{total:02d}"
    box = draw.textbbox((0, 0), badge, font=badge_font)
    width = box[2] - box[0]
    draw.rounded_rectangle((512 - 30 - width - 18, 24, 482, 52), radius=8, fill=(233, 178, 95))
    draw.text((512 - 30 - width - 9, 30), badge, font=badge_font, fill=(15, 19, 28))


def render_scripted_frame(index: int, total: int, *, label: str | None = None) -> bytes:
    """Return one deterministic, watermarked PNG frame."""
    total = max(1, int(total))
    index = max(0, min(int(index), total - 1))
    image = _background()
    _draw_shoe(image, index, total)
    _draw_labels(image, index, total, label or _env_label())
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


class ScriptedFixtureClient:
    """Provider that returns a fixed flipbook for the scripted demo.

    It accepts the same 128-channel windows as the real providers so the API contract
    is unchanged, but it never inspects the EEG content. The returned metadata and
    watermark make the lack of reconstruction explicit.
    """

    provider = "fixture"

    def __init__(self, *, label: str | None = None) -> None:
        self.label = (label or _env_label()).strip() or SCRIPTED_LABEL

    @classmethod
    def from_env(cls) -> ScriptedFixtureClient:
        return cls()

    @property
    def configured(self) -> bool:
        return True

    def status(self) -> Any:
        from . import WorkerStatus

        return WorkerStatus(
            configured=True,
            ready=True,
            reason=(
                "Scripted demo fixture is ready. It returns fixed illustration frames, "
                "not EEG reconstruction."
            ),
            backend=SCRIPTED_BACKEND,
            cuda=False,
            checkpoint=False,
            stable_diffusion=False,
            worker_version=SCRIPTED_VERSION,
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
    ) -> Any:
        from . import ReconFrame, ReconResult, ReconstructError

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

        total = max(1, arr.shape[0])
        frames = [
            ReconFrame(
                index=index,
                png=render_scripted_frame(index, total, label=self.label),
                predicted_class=SCRIPTED_CLASS,
                label=self.label,
                prompt=f"scripted demo fixture: {self.label}",
            )
            for index in range(total)
        ]
        return ReconResult(
            frames=frames,
            meta={
                "engine": "hardcoded-scripted-flipbook",
                "provider": SCRIPTED_BACKEND,
                "kind": "scripted_demo_fixture",
                "note": SCRIPTED_NOTE,
                "window_s": 440.0 / float(sfreq) if sfreq else 0.44,
                "sfreq": float(sfreq),
                "channels": len(channels),
                "scripted": True,
                "label": self.label,
                "class_index": SCRIPTED_CLASS,
            },
        )


__all__ = [
    "SCRIPTED_BACKEND",
    "SCRIPTED_CLASS",
    "SCRIPTED_LABEL",
    "SCRIPTED_NOTE",
    "SCRIPTED_VERSION",
    "ScriptedFixtureClient",
    "render_scripted_frame",
]
