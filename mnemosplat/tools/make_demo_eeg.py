#!/usr/bin/env python3
"""Generate the deterministic 128-channel EEG fixture used by the scripted demo.

The fixture is synthetic. It is *not* a recorded BrainVis trial and it does not
encode a semantic label. It exists so the upload -> job -> gallery path can be
replayed with a stable file while the scripted demo provider is enabled.

Usage:
    python tools/make_demo_eeg.py [output.npy]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import signal


def make_fixture(fs: float = 250.0, seconds: float = 7.04) -> np.ndarray:
    n = int(round(fs * seconds))
    t = np.arange(n, dtype=np.float64) / fs
    rng = np.random.default_rng(13)

    noise = rng.standard_normal((128, n))
    b, a = signal.butter(2, [0.5 / (fs / 2), 45.0 / (fs / 2)], btype="band")
    noise = signal.filtfilt(b, a, noise, axis=-1)

    # A 2 Hz cadence envelope gives the synthetic trial a running-like rhythm.
    cadence = 1.0 + 0.22 * np.sin(2 * np.pi * 2.0 * t)
    vep = (
        np.exp(-((t - 0.17) ** 2) / 0.005) * -15.0
        + np.exp(-((t - 0.30) ** 2) / 0.010) * 20.0
    )

    sig = np.zeros((128, n), dtype=np.float32)
    for ch in range(128):
        phase = (ch / 128.0) * np.pi
        sig[ch] = (
            cadence
            * (
                15.0 * np.sin(2 * np.pi * 10.5 * t + phase)
                + 8.0 * np.sin(2 * np.pi * 22.0 * t + 2 * phase)
            )
            + vep
            + 3.0 * noise[ch]
        ).astype(np.float32)
    return sig


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else Path("out/demo/hardcoded_eeg_running_shoe.npy")
    out.parent.mkdir(parents=True, exist_ok=True)
    sig = make_fixture()
    np.save(out, sig, allow_pickle=False)
    print(f"saved {out.resolve()}")
    print(f"shape={sig.shape} dtype={sig.dtype} duration={sig.shape[1] / 250.0:.2f}s")
    print("upload with sfreq=250, mode=reconstruct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
