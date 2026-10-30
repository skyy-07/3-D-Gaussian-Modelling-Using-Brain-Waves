# Experimental BrainVis reconstruction branch

> **Status: experimental, GPU-only, and disabled by default.** This branch is separate
> from the controlled 3D salience demo. It does **not** make the chair-and-mug scene
> change, and it does **not** turn EEG into a faithful video of what someone watched.

## 1. Why the current demo always shows the chair and mug

The controlled demo is intentionally fixed:

- `src/mns/scenarios.py` hard-codes the study-room narrative and its objects.
- `src/mns/render/procedural.py` draws one procedural keyframe from that scene graph.
- `src/mns/api.py` reuses the same `keyframe.png` and `depth.npy` for every uploaded
  EEG recording.
- `src/mns/geometry/gaussians.py` samples RGB from that keyframe. EEG changes only
  opacity, covariance and density.

That is the scientific control. Changing the pixels from EEG would be a different
experiment, with different validation requirements.

## 2. What BrainVis is, and what it is not

[`RomGai/BrainVis`](https://github.com/RomGai/BrainVis) (ICASSP 2025) is a **static
image reconstruction** system for an EEG-Image benchmark:

- input: roughly a 440-sample EEG trial from a 128-channel dataset;
- encoder: time-domain Transformer + frequency-domain LSTM → CLIP-aligned embedding;
- generator: latent diffusion + Stable Diffusion 1.5 img2img, conditioned by that
  embedding and a class prompt;
- output: a small set of 512×512 **static images** for one of 40 categories;
- hardware: the authors tested on an A100;
- checkpoint: **not in the repository**. The README asks researchers to request it
  from the authors by email (`hfu006@e.ntu.edu.sg`).

It has no temporal model, no depth output, no 3D representation and no arbitrary-video
decoder. A sequence of BrainVis outputs is a sequence of independent images, not a
video. The class prompt and the training distribution also constrain the result.

## 3. Can Render's free tier run it?

**No, not locally.** Render's free web service is roughly 0.1 CPU and 512 MB RAM, with
no GPU. BrainVis needs PyTorch, CUDA, Stable Diffusion 1.5, CLIP and the authors'
checkpoint; that does not fit or run on the free instance.

The free instance **can** host this FastAPI app and proxy an upload to an external GPU
worker over HTTPS. It can also call a hosted image API. Neither makes the CPU instance
a reconstruction engine.

For a real GPU run you need one of:

1. your own CUDA machine (the worker in [`services/brainvis_worker`](../services/brainvis_worker));
2. a GPU cloud service (RunPod, Lambda, Modal, a GPU Droplet, etc.);
3. a paid hosted inference endpoint.

The live CPU app remains available even when the worker is offline; health reports the
worker as unavailable and the reconstruction mode refuses with a clear error.

## 4. Architecture

```text
browser ──POST /api/eeg?mode=reconstruct──► CPU FastAPI app (Render)
                                                   │  multipart 128×T EEG windows
                                                   ▼
                                        external CUDA worker
                                        BrainVis + SD1.5 + checkpoint
                                                   │  PNG frames + metadata
                                                   ▼
                                        CPU app stores /assets/live/recon-*
                                                   │
                                                   ▼
                                        browser filmstrip (static frames)
```

The CPU app never imports `torch` and never calls Stable Diffusion. It validates the
upload, enforces the admin gate and rate limits, forwards the windows, and stores the
returned PNGs.

## 5. Requirements and limits

### Data

- **Exactly 128 channels** in the EEG-Image training order. The current upload parser
  accepts 128 channels, but it cannot guess a channel permutation from an arbitrary
  montage. A 6-channel posterior cap cannot be mapped honestly to the 128-channel
  model.
- A sampling rate between 100 and 2000 Hz. The worker resamples each 440 ms window to
  440 samples.
- At least one complete 440 ms window. Four evenly spaced windows are used by default,
  up to `MNS_RECON_MAX_FRAMES`.
- The recording must be from a compatible subject/session and preprocessing pipeline.
  BrainVis is subject-specific; out-of-distribution EEG produces category-level
  guesses, not a memory.

### Output

- 512×512 static frames, one per EEG window.
- Independent frames, no temporal consistency, no optical flow, no video encoding.
- The model may return a class label from the 40-class vocabulary. That label is a
  model prediction, not a statement about what the person actually saw.
- Outputs are session-only and deleted when the service restarts.

### Cost and safety

- GPU inference is slow (tens of seconds to minutes per window on a small GPU).
- The public demo requires `MNS_ADMIN_TOKEN` for reconstruction. The existing public
  recorded-EEG conditioning path remains CPU-only and token-free.
- Do not present these frames as clinical evidence, a memory, or a faithful video
  reconstruction.

## 6. Configuration

### CPU app (Render/local)

| Variable | Default | Purpose |
| --- | --- | --- |
| `MNS_BRAINVIS_WORKER_URL` | unset | Base URL of the CUDA worker, e.g. `https://gpu.example.net` |
| `MNS_BRAINVIS_WORKER_TOKEN` | unset | Bearer token shared with the worker |
| `MNS_BRAINVIS_TIMEOUT_S` | `900` | Per-reconstruction HTTP timeout |
| `MNS_RECON_MAX_FRAMES` | `4` | Maximum static frames per upload |
| `MNS_RECON_STATUS_TTL_S` | `20` | Health-probe cache |
| `MNS_ADMIN_TOKEN` | unset | Required to enable reconstruction on a public demo |

### GPU worker

See [`services/brainvis_worker/README.md`](../services/brainvis_worker/README.md).
The worker expects:

| Variable | Purpose |
| --- | --- |
| `BRAINVIS_ROOT` | BrainVis checkout |
| `MNS_BRAINVIS_CHECKPOINT` | `clipfinetune_model.pkl` |
| `MNS_BRAINVIS_SD_CKPT` | `v1-5-pruned-emaonly.ckpt` |
| `MNS_BRAINVIS_WORKER_TOKEN` | Must match the CPU app token |
| `MNS_BRAINVIS_DEVICE` | `cuda` (default) |

## 7. Enabling it

1. Obtain the BrainVis checkpoint from the authors.
2. Put the BrainVis checkout, checkpoint and SD1.5 checkpoint on a CUDA machine.
3. Start the worker and check `GET /health`; `ready` must be `true`.
4. Set `MNS_BRAINVIS_WORKER_URL` and `MNS_BRAINVIS_WORKER_TOKEN` on the CPU app.
5. Set `MNS_ADMIN_TOKEN` if the CPU app is public.
6. Open the EEG dialog, choose **Reconstruct frames (GPU)**, and upload a 128-channel
   recording.

Without the checkpoint, the worker reports `checkpoint: false` and the CPU app refuses
the request. There is intentionally no random-weight or "make something up" fallback.

## 8. What would make this closer to video

BrainVis alone is not enough. A real video path would need:

- per-frame or per-window EEG synchronized to the stimulus;
- a temporal model or explicit motion/consistency constraint;
- validation on held-out videos and subjects;
- a clearly separated evaluation from the fixed-scene control;
- depth/3D lift only if the generated frames are stable enough.

Until then, the honest description is **experimental static-frame reconstruction**.
