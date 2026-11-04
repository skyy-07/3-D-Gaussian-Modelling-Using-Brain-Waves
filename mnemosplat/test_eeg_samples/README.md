# BrainVis Sample EEG Test Dataset (`.npy`)

This directory contains pre-generated sample EEG tensors in standard NumPy (`.npy`) format, conforming strictly to the **BrainVis (ICASSP 2025)** input contract (128 channels, 440 time samples @ 250 Hz sampling rate, 1.76-second stimulus observation window).

---

## 📁 Scenarios Overview

| Filename | Shape | Primary Dynamics | Intended Test Scenario |
| :--- | :--- | :--- | :--- |
| `scenario_1_visual_stimulus_panda_alpha.npy` | `(128, 440)` | P100 + N170 + P300 VEP, occipital alpha desynchronization, beta sync | High-confidence visual category perception (Animal / Panda) |
| `scenario_2_motor_bike_high_attention_beta_gamma.npy` | `(128, 440)` | High Beta (24 Hz) & Gamma (38 Hz) bursts, parietal activation | Dynamic visual motion tracking & high cognitive attention (Vehicle) |
| `scenario_3_food_pizza_frontal_theta_lpp.npy` | `(128, 440)` | Frontal midline Theta (6 Hz), Late Positive Potential (LPP) @ 450ms | Appetitive / reward stimulus evaluation (Food / Pizza) |
| `scenario_4_resting_eyes_closed_alpha_baseline.npy` | `(128, 440)` | Large synchronized Alpha rhythm (10 Hz, 35–50 µV), no VEP | Baseline control / eyes-closed resting state (no visual stimulus) |
| `scenario_5_artifact_corrupted_eog_emg_stress_test.npy` | `(128, 440)` | Frontal eye blinks (EOG ~100 µV), temporal muscle noise (EMG), drift | Robustness stress-testing, outlier rejection & filter resilience |
| `scenario_6_contract_transposed_440x128.npy` | `(440, 128)` | Transposed dimensions (`time_samples`, `channels`) | API tensor orientation handling & contract compliance |

---

## 🧪 How to Use

### 1. Via Python Script
```python
import requests
import numpy as np

# Load a test sample
eeg_data = np.load("test_eeg_samples/scenario_1_visual_stimulus_panda_alpha.npy")
print("Loaded shape:", eeg_data.shape)

# Send to BrainVis API Gateway
url = "http://localhost:8000/api/pipeline"
response = requests.post(url, json={"eeg_data": eeg_data.tolist(), "top_k": 5})

result = response.json()
print("Contract Verified:", result["analysis"]["input_contract_verified"])
print("CLIP Latent Norm:", result["analysis"]["clip_latent_norm"])
print("Top Stimulus:", result["analysis"]["predicted_stimulus"]["label"])
```

### 2. Via File Upload (`curl`)
```bash
curl -X POST "http://localhost:8000/api/upload-eeg" \
  -F "file=@test_eeg_samples/scenario_1_visual_stimulus_panda_alpha.npy"
```

### 3. Via Web Dashboard
1. Open `http://localhost:8000` (or your Render URL) in your browser.
2. Under **Upload EEG (.npy, .csv, .json)**, click "Browse..." and select any `.npy` file from this folder.
3. Click **Run BrainVis Pipeline**.
