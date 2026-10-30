# BrainVis GPU worker

This is the optional CUDA side of the experimental reconstruction branch. It is **not**
part of the Render CPU image. The web app can call it over HTTPS, but it cannot run it
on the free plan.

> BrainVis checkpoints are released by the authors on request; they are not in the
> GitHub repository. Without `clipfinetune_model.pkl`, this worker intentionally refuses
> to generate images. There is no random-weight or text-to-image fallback.

## What it does

```text
POST /reconstruct
  multipart: eeg.npy (N, 128, T), sfreq, channels JSON, prompt?, seed?
    -> resample each 0.44 s window to 440 samples
    -> optional z-score / 5-95 scaling
    -> BrainVis AlignNet (time + frequency encoders)
    -> latent diffusion stage 1
    -> Stable Diffusion 1.5 img2img stage 2
    -> N independent 512x512 PNGs + class labels
```

The response is JSON with base64 PNGs, not a video. There is no temporal model and no
depth/3D output.

## Required artifacts

Place these on the GPU host:

```text
/opt/BrainVis/                                      # pinned public checkout
/opt/BrainVis/pretrained_model/config15.yaml        # included in the checkout
/opt/BrainVis/pretrained_model/v1-5-pruned-emaonly.ckpt
/models/clipfinetune_model.pkl                      # request from the BrainVis authors
```

The worker environment:

```bash
export BRAINVIS_ROOT=/opt/BrainVis
export MNS_BRAINVIS_CHECKPOINT=/models/clipfinetune_model.pkl
export MNS_BRAINVIS_SD_CKPT=/opt/BrainVis/pretrained_model/v1-5-pruned-emaonly.ckpt
export MNS_BRAINVIS_WORKER_TOKEN=change-me
export MNS_BRAINVIS_DEVICE=cuda
# Optional:
export MNS_BRAINVIS_STEPS=50
export MNS_BRAINVIS_STAGE2_STEPS=100
export MNS_BRAINVIS_NUM_SAMPLES=1
export MNS_BRAINVIS_NORMALIZE=zscore   # zscore | minmax | none
```

## Run

```bash
cd services/brainvis_worker
python3.10 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Check readiness:

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

`ready` must be `true`. `checkpoint`, `stable_diffusion` and `cuda` show exactly which
dependency is missing.

Send one window:

```bash
python - <<'PY'
import io, json, httpx, numpy as np
x = np.zeros((1, 128, 110), dtype=np.float32)  # 0.44 s at 250 Hz
buf = io.BytesIO(); np.save(buf, x)
r = httpx.post(
    "http://127.0.0.1:8000/reconstruct",
    headers={"Authorization": "Bearer change-me"},
    files={"file": ("eeg.npy", buf.getvalue(), "application/octet-stream")},
    data={"sfreq": "250", "channels": json.dumps([f"ch{i}" for i in range(128)])},
    timeout=900,
)
print(r.status_code, r.text[:200])
PY
```

## CPU app configuration

Point the web app at the worker:

```bash
export MNS_BRAINVIS_WORKER_URL=http://gpu-host:8000
export MNS_BRAINVIS_WORKER_TOKEN=change-me
```

On a public deployment, also set `MNS_ADMIN_TOKEN`; the reconstruction route requires
the admin token so an anonymous visitor cannot spend GPU time.

## Limitations

- **128 channels only**, in the EEG-Image training order. A normal 6-channel montage is
  not silently remapped.
- **Subject/session-specific**. The checkpoint was fine-tuned for a benchmark subject;
  arbitrary EEG is out of distribution.
- **Static frames**. Four windows produce four independent images, not four video
  frames. Do not call the output a reconstruction of a specific video.
- **Class prompt prior**. The released cascade conditions stage 2 on one of 40 class
  prompts. The class is a model prediction, not ground truth.
- **Memory**. Both the LDM and the img2img pipeline are resident; use an A100-class GPU
  or unload/reload stages deliberately.
- **No evaluation**. The worker does not compute FID/SSIM/CLIP against a stimulus. It
  cannot establish that a frame is faithful.
