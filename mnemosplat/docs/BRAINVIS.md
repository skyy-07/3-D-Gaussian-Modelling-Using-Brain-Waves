# BrainVis bridge

`RomGai/BrainVis` is the backbone for the **learned** salience branch. MnemoSplat uses it
for one narrow job:

```text
raw posterior EEG ──► BrainVis freq encoder + time encoder ──► CLIP-aligned embedding
                                                                    │
                                                          small regression head
                                                                    ▼
                                                     per-entity salience weights
```

It does **not** use BrainVis to generate RGB. That restriction is the scientific point of
the project, not an implementation shortcut. See [SCIENCE.md](SCIENCE.md).

The default demo runs on `SpectralSalienceEncoder` (Welch PSD + calibration), which needs
no weights, no GPU and no checkpoint. This document describes how the learned branch
plugs in when the artifacts are available.

---

## 1. Current status

| Item | State |
| --- | --- |
| Data loader for `.npy` / `.npz` / `.pth` / BrainVis block dumps | Implemented (`mns.eeg.brainvis.load_recording`) |
| Zero-dependency spectral encoder | Implemented, used by the demo |
| Artifact/checkout check | Implemented (`resolve_brainvis`, surfaced by `mnemosplat doctor`) |
| Encoder protocol | Implemented (`SalienceEncoder`) |
| TorchScript inference | **Not wired** — `BrainVisSalienceEncoder` raises a clear error |
| Trained regression head | **Not shipped** — requires labels and a GPU run |
| Checkpoints | **Not present** — BrainVis releases them on request from the authors |

This is intentional. An untrained pseudo-encoder would emit plausible-looking noise and
poison the attention field. The code raises rather than pretending.

```bash
BRAINVIS_ROOT=/path/to/BrainVis mnemosplat doctor
```

`resolve_brainvis` checks:

```text
data/EEG/block_splits_by_image_all.pth
data/EEG/eeg_5_95_std.pth
pretrained_model/v1-5-pruned-emaonly.ckpt
```

`ready` is true only when the checkout exists and all three are present.

---

## 2. What BrainVis provides

BrainVis (ICASSP 2025) learns EEG representations aligned to CLIP image embeddings using
frequency and time encoders, then reconstructs images from those embeddings. The pieces
that matter here are the encoders and the aligned embedding, not the generator.

MnemoSplat's hypothesis is that the embedding is a better substrate for **ranking
entities by encoding strength** than for regressing pixels. Ranking is a
low-dimensional, high-signal target; RGB regression is not.

Useful references:

- Repository: <https://github.com/RomGai/BrainVis>
- THINGS-EEG2 benchmark: <https://www.sciencedirect.com/science/article/pii/S1053811922008758>
- Related EEG-to-image work:
  <https://github.com/IsaacVanB/visual-reconstruction-eeg>,
  <https://github.com/desa-lab/EEG-Image-Reconstruction>

---

## 3. Data plumbing

### 3.1 Load a recording

```python
from mns.eeg.brainvis import load_recording

rec = load_recording(
    "subject01_session02.npy",
    sfreq=250.0,
    channels=("O1", "O2", "Oz", "P3", "P4", "Pz"),
)
print(rec.signal.shape, rec.duration_s, rec.synthetic)
```

Behaviour:

- `.npy` → array;
- `.npz` → `key=` or the first array;
- `.pth` / `.pt` → `torch.load` (requires torch); looks for `eeg`, `signal`, `data`, `x`,
  `dataset`; supports BrainVis `dataset[split][i]["eeg"]` via `key="dataset", trial=i`;
- anything else → delimited text.

The loader transposes arrays that look samples-first (`shape[0] > shape[1]`) and falls
back to `ch0…chN` names when the montage length does not match. Recordings loaded from
disk are marked `synthetic=False`; the manifest records `upstream.eeg = "file"`.

### 3.2 Validate before trusting

Before using a file in a demo:

```python
import numpy as np
assert rec.signal.ndim == 2
assert np.isfinite(rec.signal).all()
print("µV range", np.percentile(rec.signal, [1, 99]))
print("duration", rec.duration_s)
```

A 60 Hz notch, a 1–50 Hz band-pass and bad-channel rejection are expected preprocessing
steps for real data. They are not applied automatically because the right choices depend
on the amplifier and montage.

---

## 4. Encoder contract

```python
class SalienceEncoder(Protocol):
    def encode(self, recording: EEGRecording, epoch_s: float = 0.75): ...
```

The spectral implementation returns `(list[EpochFeatures], calibrator, band_table)`,
exactly what the pipeline consumes.

A learned encoder must return the same triple, or a list of `EpochFeatures` plus a
compatible calibrator. The minimum fields are:

- `salience ∈ [0.1, 1.0]` — the vividness gate;
- `certainty ∈ [0, 1]` — kernel spread;
- `lateralization ∈ [−1, 1]` — hemifield prior;
- band powers and `gamma_theta` for the HUD;
- `channels_used`.

### 4.1 TorchScript export

Train or obtain an `nn.Module` that maps `(B, C, T) → (B, D)` CLIP-aligned embeddings and
export it:

```python
import torch

class Encoder(torch.nn.Module):
    def __init__(self, freq_encoder, time_encoder, projection):
        super().__init__()
        self.freq_encoder = freq_encoder
        self.time_encoder = time_encoder
        self.projection = projection

    def forward(self, eeg):           # eeg: (B, C, T), float32
        z = self.projection(self.freq_encoder(eeg) + self.time_encoder(eeg))
        return z                       # (B, D), L2-normalised

model = Encoder(...).eval()
example = torch.zeros(1, 6, 250)
torch.jit.trace(model, example)
torch.jit.save(torch.jit.trace(model, example), "brainvis_encoder.pt")
```

Keep the input contract explicit: channel order, sampling rate, units and epoch length.
A mismatch here is the most likely reason a working checkpoint produces garbage.

### 4.2 Regression head

The embedding must be mapped to per-entity salience. Two viable targets:

1. **Human ratings (preferred).** After each memory, ask the participant to rate how
   vividly each entity was experienced (0–100). Aggregate across raters; keep
   per-subject and per-entity splits.
2. **Behavioural proxy.** Use gaze dwell or free-recall mention as a weaker target, and
   say so. Do not present a proxy as ground-truth salience.

A JSONL row:

```json
{"episode": "study-07", "subject": "s03", "epoch": 2, "object": "mug", "target": 0.91, "split": "train"}
```

Fit a small head (ridge or 2-layer MLP) with leave-one-subject-out splits. Compare
against the spectral baseline on the same folds. Report Spearman correlation and
calibration error, not just MSE.

The BrainVis docstring in `mns/eeg/brainvis.py` describes this contract. There is no
`mnemosplat labels` command in this checkout; label export is deliberately left to the
study code so the target definition is not hidden in a demo CLI.

---

## 5. Wiring it into the pipeline

The current pipeline calls `analyze_epochs` directly. The integration is a small,
explicit change at that boundary, not a rewrite:

```python
from mns.eeg.brainvis import BrainVisSalienceEncoder
from mns.eeg.psd import SalienceCalibrator

encoder = BrainVisSalienceEncoder(encoder_path="brainvis_encoder.pt")
feats, calibrator, table = encoder.encode(rec, epoch_s=cfg.epoch_s)
```

To make that work end to end:

1. Implement `BrainVisSalienceEncoder.encode` to:
   - window `rec.signal` into epochs;
   - normalise (µV, channel order, sample rate) exactly as in training;
   - run the TorchScript encoder under `torch.no_grad()`;
   - run the regression head;
   - calibrate the output with `SalienceCalibrator` (population or reference);
   - derive certainty/lateralisation from the spectral path or a second head;
   - return `EpochFeatures`.
2. Add an `encoder` parameter to `run_episode` (or a small factory keyed by
   `MNS_EEG_ENCODER=spectral|brainvis`).
3. Record `upstream.eeg = "brainvis:<checkpoint-hash>"` so the manifest never confuses
   learned and spectral salience.
4. Add a test that a random-weight encoder still returns bounded features, and a test
   that a missing checkpoint raises.

Do not silently fall back from a requested learned encoder to the spectral one. A
fallback is fine, but it must be visible.

---

## 6. Evaluation plan

| Question | Metric | Baseline |
| --- | --- | --- |
| Does the embedding predict entity salience? | Spearman ρ, per-subject | Spectral salience |
| Is it calibrated? | ECE / reliability curve | Population calibration |
| Does it generalise? | Leave-one-subject-out | Same-subject fit |
| Does it beat a dataset prior? | ρ vs. constant/recipe prior | Recipe `base_salience` |
| Does it help the render? | Blind human vividness preference | Spectral render |

A learned encoder that beats the spectral baseline on held-out subjects is a result. A
learned encoder that only memorises the training scenes is not.

---

## 7. Why the demo does not wait for checkpoints

A hackathon demo must run on a laptop with no GPU and no author email. The spectral
encoder:

- runs in milliseconds on CPU;
- is deterministic and testable;
- exposes the same interface;
- is honest about being a heuristic.

The learned branch is the research upgrade path. Keeping the interface stable means the
demo, the HUD and the manifest do not change when it lands.
