"""
EEG Sample Generator for BrainVis Contract Testing
Generates realistic 128-channel, 440-sample (250 Hz) EEG tensors in .npy format
depicting diverse neurophysiological and experimental scenarios.
"""

import os
import numpy as np
from scipy import signal

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

CHANNELS = 128
TIME_SAMPLES = 440  # 1.76 seconds @ 250 Hz (BrainVis experimental window)
FS = 250.0          # Sampling rate (Hz)
TIME_AXIS = np.linspace(0, TIME_SAMPLES / FS, TIME_SAMPLES)

def generate_pink_noise(channels, samples):
    """Generates realistic 1/f background neural noise."""
    white = np.random.randn(channels, samples)
    # Bandpass filter between 0.5 Hz and 45 Hz
    b, a = signal.butter(2, [0.5 / (FS / 2), 45.0 / (FS / 2)], btype="band")
    return signal.filtfilt(b, a, white, axis=-1)

def build_vep(latency_p100=0.10, latency_n170=0.17, latency_p300=0.30):
    """
    Simulates Visual Evoked Potential (VEP) waveform:
    - P100: Early primary visual cortex response (~100ms)
    - N170: Structural visual recognition / semantic binding (~170ms)
    - P300: Cognitive categorization / stimulus evaluation (~300ms)
    """
    p100 = 18.0 * np.exp(-((TIME_AXIS - latency_p100) ** 2) / 0.0015)
    n170 = -26.0 * np.exp(-((TIME_AXIS - latency_n170) ** 2) / 0.003)
    p300 = 22.0 * np.exp(-((TIME_AXIS - latency_p300) ** 2) / 0.008)
    return p100 + n170 + p300

def scenario_1_visual_panda():
    """
    Scenario 1: Visual stimulus presentation (Panda / Animal)
    - Occipital-temporal visual evoked potential (P100, N170, P300)
    - Post-stimulus alpha event-related desynchronization (ERD)
    - Moderate beta (21 Hz) synchronization
    """
    noise = generate_pink_noise(CHANNELS, TIME_SAMPLES) * 3.5
    vep = build_vep(0.10, 0.17, 0.30)
    data = np.zeros((CHANNELS, TIME_SAMPLES), dtype=np.float32)

    for ch in range(CHANNELS):
        # Posterior occipital channels (80-127) exhibit strong VEP
        occipital_weight = 1.2 if ch >= 70 else 0.4
        phase = (ch / CHANNELS) * np.pi
        alpha_burst = 14.0 * np.sin(2 * np.pi * 10.5 * TIME_AXIS + phase) * np.exp(-TIME_AXIS / 0.8)
        beta_burst = 7.0 * np.sin(2 * np.pi * 21.0 * TIME_AXIS + 2 * phase) * (1.0 - np.exp(-TIME_AXIS / 0.5))
        data[ch] = (occipital_weight * vep) + alpha_burst + beta_burst + noise[ch]

    return data

def scenario_2_motor_bike():
    """
    Scenario 2: Dynamic vehicle stimulus (Mountain bike / motion perception)
    - High visual attention & motion tracking
    - Prominent Beta (24 Hz) and early Gamma (38 Hz) synchronization
    - Prominent parietal activation
    """
    noise = generate_pink_noise(CHANNELS, TIME_SAMPLES) * 4.0
    vep = build_vep(0.09, 0.16, 0.28) * 0.9
    data = np.zeros((CHANNELS, TIME_SAMPLES), dtype=np.float32)

    for ch in range(CHANNELS):
        phase = (ch / CHANNELS) * 2 * np.pi
        beta = 16.0 * np.sin(2 * np.pi * 24.0 * TIME_AXIS + phase)
        gamma = 8.5 * np.sin(2 * np.pi * 38.0 * TIME_AXIS + phase * 1.5)
        data[ch] = (0.8 * vep) + beta + gamma + noise[ch]

    return data

def scenario_3_food_pizza():
    """
    Scenario 3: Appetitive Food Stimulus (Pizza)
    - Frontal midline theta (6 Hz) enhancement
    - Prolonged Late Positive Potential (LPP) past 350ms
    - Occipital alpha suppression
    """
    noise = generate_pink_noise(CHANNELS, TIME_SAMPLES) * 3.0
    vep = build_vep(0.10, 0.18, 0.35)
    lpp = 15.0 * np.exp(-((TIME_AXIS - 0.45) ** 2) / 0.02)
    data = np.zeros((CHANNELS, TIME_SAMPLES), dtype=np.float32)

    for ch in range(CHANNELS):
        # Frontal channels (0-40) show strong theta
        frontal_weight = 1.3 if ch <= 40 else 0.5
        theta = 12.0 * np.sin(2 * np.pi * 6.0 * TIME_AXIS + (ch * 0.1)) * frontal_weight
        alpha = 6.0 * np.sin(2 * np.pi * 9.5 * TIME_AXIS + (ch * 0.05))
        data[ch] = vep + lpp + theta + alpha + noise[ch]

    return data

def scenario_4_resting_state_eyes_closed():
    """
    Scenario 4: Baseline Resting State (Eyes Closed)
    - High-amplitude synchronized Alpha rhythm (10.0 Hz, 35-50 uV)
    - No stimulus evoked potentials (no VEP)
    - Homogeneous posterior topography
    """
    noise = generate_pink_noise(CHANNELS, TIME_SAMPLES) * 2.5
    data = np.zeros((CHANNELS, TIME_SAMPLES), dtype=np.float32)

    for ch in range(CHANNELS):
        phase = np.sin(ch * 0.2)
        # Dominant, persistent 10 Hz rhythm
        alpha_continuous = 32.0 * np.sin(2 * np.pi * 10.0 * TIME_AXIS + phase)
        # Low background delta
        delta = 6.0 * np.sin(2 * np.pi * 1.8 * TIME_AXIS + phase * 0.5)
        data[ch] = alpha_continuous + delta + noise[ch]

    return data

def scenario_5_noisy_corrupted_artifacts():
    """
    Scenario 5: High-Artifact Corrupted EEG (Stress Test)
    - Frontal eye blinks (EOG: 80-120 uV slow spikes on frontal channels 0-15)
    - Temporal muscle jitter (EMG: 50-80 Hz high-frequency bursts on channels 45-65)
    - Baseline drift (0.3 Hz slow offset)
    """
    noise = generate_pink_noise(CHANNELS, TIME_SAMPLES) * 5.0
    data = np.zeros((CHANNELS, TIME_SAMPLES), dtype=np.float32)

    # Simulated blink at 0.5s and 1.2s
    blink1 = 95.0 * np.exp(-((TIME_AXIS - 0.5) ** 2) / 0.004)
    blink2 = 110.0 * np.exp(-((TIME_AXIS - 1.2) ** 2) / 0.005)
    blink = blink1 + blink2

    # Baseline drift
    drift = 25.0 * np.sin(2 * np.pi * 0.3 * TIME_AXIS)

    for ch in range(CHANNELS):
        emg = 0.0
        eog = 0.0
        if ch < 16:
            # Frontal electrodes capture eye blinks
            eog = blink * ((16 - ch) / 16.0)
        elif 45 <= ch <= 65:
            # Temporal electrodes capture muscle tension
            emg = np.random.randn(TIME_SAMPLES) * 28.0

        alpha = 8.0 * np.sin(2 * np.pi * 10.0 * TIME_AXIS)
        data[ch] = eog + emg + drift + alpha + noise[ch]

    return data

def scenario_6_transposed_format():
    """
    Scenario 6: Transposed Shape (440, 128)
    - Validates endpoint orientation handling and contract compliance.
    """
    standard = scenario_1_visual_panda()
    return standard.T  # Shape: (440, 128)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    scenarios = {
        "scenario_1_visual_stimulus_panda_alpha.npy": scenario_1_visual_panda,
        "scenario_2_motor_bike_high_attention_beta_gamma.npy": scenario_2_motor_bike,
        "scenario_3_food_pizza_frontal_theta_lpp.npy": scenario_3_food_pizza,
        "scenario_4_resting_eyes_closed_alpha_baseline.npy": scenario_4_resting_state_eyes_closed,
        "scenario_5_artifact_corrupted_eog_emg_stress_test.npy": scenario_5_noisy_corrupted_artifacts,
        "scenario_6_contract_transposed_440x128.npy": scenario_6_transposed_format
    }

    print("Generating BrainVis test .npy samples...")
    for filename, generator in scenarios.items():
        arr = generator()
        filepath = os.path.join(OUTPUT_DIR, filename)
        np.save(filepath, arr)
        print(f"  ✓ Saved: {filename} -> shape: {arr.shape}, dtype: {arr.dtype}, size: {arr.nbytes / 1024:.1f} KB")

    print(f"\nAll sample files successfully written to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
