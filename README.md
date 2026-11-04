# MnemoSplat: 3D Gaussian Modelling Using Brain Waves

**Same words. Different memory.** EEG-salience-conditioned 3D Gaussian memory volumes for WebXR.

[![Live Demo](https://img.shields.io/badge/Demo-Live%20on%20Render-blue?style=for-the-badge&logo=render)](https://mnemosplat.onrender.com)
[![BrainVis Mirror](https://img.shields.io/badge/BrainVis%20Endpoint-Active-green?style=for-the-badge)](https://brainvis-api.onrender.com)
[![GitHub License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

* **Live Interactive WebXR App:** <https://mnemosplat.onrender.com>  
* **BrainVis Neural Decoding API:** <https://brainvis-api.onrender.com>  
* **Source & Package Docs:** [`mnemosplat/`](mnemosplat/)  
* **BrainVis Mirror Repository:** [`skyy-07/BrainVis`](https://github.com/skyy-07/BrainVis)

> [!NOTE]
> The public demo runs on Render's free tier, which spins down after 15 minutes of inactivity. The initial cold start may take 30–60 seconds while the container initializes.

---

## 🧠 Project Overview

**MnemoSplat** is a multi-modal spatial computing system that translates spoken episodic memories into interactive, immersive **3D Gaussian Splatting volumes** for WebXR (compatible with web browsers and VR headsets like Meta Quest).

When two people experience or recount the exact same verbal story, their subjective memories are never identical. Traditional 3D reconstruction pipelines generate uniform scenes regardless of the human experience. MnemoSplat bridges this gap by conditioning 3D Gaussian memory volumes with **human electroencephalography (EEG)**:

* **The Language Model (LLM)** extracts *what* entities are in the scene from the narrative transcript.
* **Monocular Depth Estimation (Depth-Anything-V2)** computes *where* entities are located in 3D Euclidean space.
* **Brain Signals (EEG / BrainVis)** govern *how vividly those entities consolidated into memory* by modulating the 3D Gaussian density, opacity, kernel scale, and spatial falloff.

> **Key Design Principle:** EEG signals modulate geometric and salience parameters; **EEG never touches RGB pixels directly**, eliminating visual hallucinations while preserving objective spatial geometry.

---

## 🏛️ System Architecture

```text
Spoken Memory Transcript
          │
          ▼
┌───────────────────────────┐
│     LLM Scene Analysis    │ ──► Entity Extractor & Bounding Boxes
└─────────────┬─────────────┘
              ├──► Keyframe Synthesis (Image Generator) ──► Monocular Depth (DA-V2)
              │                                                    │
              │                                                    ▼
┌─────────────┴─────────────┐                           ┌─────────────────────┐
│    Raw Posterior EEG      │                           │  Single-View 3D     │
│ (128 Channels, 440 Samples│                           │  Gaussian Lifting   │
│         @ 250 Hz)         │                           │  (gsplat / .splat)  │
└─────────────┬─────────────┘                           └──────────┬──────────┘
              ▼                                                    │
┌───────────────────────────┐                                      │
│  BrainVis Multi-Domain    │                                      │
│      Neural Encoder       │                                      ▼
│ • TimeEncoder (Transformer│                           ┌─────────────────────┐
│ • FreqEncoder (FFT+LSTM)  │                           │   WebXR Viewer      │
│ • AlignNet (77×768 CLIP)  │ ──► Salience Field M(u,v) │ (GaussianSplats3D   │
└───────────────────────────┘     (Modulates Gaussian   │  Browser / VR HMD)  │
                                   Opacity & Density)   └─────────────────────┘
```

---

## 🔬 The BrainVis Bridge (EEG Decoding & Image Reconstruction)

MnemoSplat integrates with [BrainVis](https://github.com/skyy-07/BrainVis) (*ICASSP 2025*) to decode visual categories and visual features directly from brain signals:

1. **Strict Input Contract**:
   * Expects **128 channels** in standard 10–20 montage across **440 time samples** (samples 20–460 @ 250 Hz, capturing 1.76 seconds of stimulus observation).
2. **Multi-Domain Neural Alignment**:
   * **Time-Domain Transformer**: 8 layers, 16 attention heads, 1024-d model for temporal wave dynamics.
   * **Frequency-Domain LSTM**: Fast Fourier Transform (FFT) amplitude spectrum passed through an LSTM across standard clinical bands (Delta, Theta, Alpha, Beta, Gamma).
   * **AlignNet**: Cross-modal alignment projecting EEG representations into a **`[1, 77, 768]` CLIP text-image latent embedding**.
3. **Cascaded Diffusion Pipeline (GPU Worker)**:
   * **Stage 1 (PLMS Latent Diffusion)**: Synthesizes a coarse visual stimulus conditioned on the `[77, 768]` EEG-CLIP latent vector.
   * **Stage 2 (SD 1.5 Img2Img)**: Refines the coarse image with high-frequency details (`strength=0.85`) based on decoded visual categories.

---

## 🧪 Pre-Baked Test Dataset (`.npy`)

Standardized 128-channel, 440-sample `.npy` test files are available in [`mnemosplat/test_eeg_samples/`](mnemosplat/test_eeg_samples/) to evaluate different neurophysiological states:

| Sample File | Dynamics | Target Scenario |
| :--- | :--- | :--- |
| `scenario_1_visual_stimulus_panda_alpha.npy` | P100 + N170 + P300 VEPs, alpha desync | High-confidence visual perception (Animal / Panda) |
| `scenario_2_motor_bike_high_attention_beta_gamma.npy` | Beta (24 Hz) & Gamma (38 Hz) bursts | Dynamic visual motion & attention (Vehicle) |
| `scenario_3_food_pizza_frontal_theta_lpp.npy` | Frontal Theta (6 Hz) & LPP @ 450ms | Appetitive / reward stimulus evaluation (Food) |
| `scenario_4_resting_eyes_closed_alpha_baseline.npy` | Large 10 Hz Alpha waves (35–50 µV), no VEP | Negative baseline / resting state with eyes closed |
| `scenario_5_artifact_corrupted_eog_emg_stress_test.npy` | EOG blinks (~100 µV), EMG noise, drift | Stress-testing filter & artifact resilience |
| `scenario_6_contract_transposed_440x128.npy` | Transposed `(440, 128)` format | Verifying tensor dimension orientation handling |

---

## 🚀 Running Locally

### Prerequisites
* Python ≥ 3.10
* Node.js ≥ 20

```bash
cd mnemosplat

# 1. Set up Python environment
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Generate paired demo volumes (approx. 60s on CPU)
mnemosplat demo --scenario both --out out/demo --splats 120000 --force

# 3. Build WebXR frontend
cd web && npm install && npm run build && cd ..

# 4. Launch MnemoSplat server
mnemosplat serve --host 0.0.0.0 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## ☁️ Cloud Deployment (Render)

MnemoSplat deploys to Render using the root [`render.yaml`](render.yaml) specification with Docker:

* The Docker image builds the Vite frontend and pre-generates demo splat volumes at build time, requiring **zero GPU** and **zero API keys** for runtime browsing.
* **Cold-Start Resilience**: Built-in exponential backoff (`MNS_BRAINVIS_RETRY_BUDGET_S=120s`, `MNS_BRAINVIS_RETRIES=4`) accommodates sleeping free-tier BrainVis mirror instances seamlessly.

---

## ⚖️ Scientific Integrity & Disclaimer

* Baked comparison conditions utilize synthetic EEG, labeled as such.
* Uploaded files are labeled as recorded EEG.
* The reconstruction endpoint uses a lightweight CPU mirror of BrainVis for accessibility. Full cascaded diffusion requires the GPU native worker.
* This is a systems research prototype in computational neuroscience and spatial computing, not a diagnostic or clinical tool.
