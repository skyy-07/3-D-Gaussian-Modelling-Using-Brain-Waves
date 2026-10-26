# Science and claims

MnemoSplat is a **systems prototype** that makes one narrow, testable claim:

> Posterior EEG spectral state can gate the *density, opacity and spread* of a
> reconstructed memory volume in a way that is visible, reproducible and separable from
> the scene's semantic content.

It does **not** claim to read minds, reconstruct what a person saw, diagnose PTSD, or
improve clinical outcomes. Those are different experiments with different evidence
requirements.

This document states the hypothesis, the features, the control that makes the demo
honest, the PTSD rationale and its limits, and the study that would be required before
any stronger claim.

---

## 1. Why EEG should gate density, not pixels

EEG-to-image work often regresses RGB from an embedding. The signal-to-noise ratio of
scalp EEG is poor relative to the dimensionality of natural images, and a blurred or
hallucinated reconstruction is hard to distinguish from a model that has learned the
dataset prior. The result is a demo that looks impressive on cherry-picked trials and
collapses under a held-out protocol.

MnemoSplat targets a lower-dimensional variable with a better-established link to
posterior cortex: **how strongly a visual memory was encoded**. That variable maps
naturally onto rendering parameters — opacity, covariance, sampling density — which are
exactly the parameters a 3D Gaussian representation exposes.

The division of labour is deliberate:

| Variable | Source | Why |
| --- | --- | --- |
| What is in the room | Transcript + language model | Semantics are low-bandwidth and editable |
| Where it is | Monocular depth | Geometry is a vision problem, not an EEG problem |
| How vivid it is | Posterior EEG | Arousal/encoding is what the band ratio indexes |
| Which side is favoured | O1/O2 asymmetry | A weak, well-known hemifield prior |

EEG never multiplies an RGB value. It only changes the attention field, which changes
opacity, scale and sampling. A sceptical judge can verify this in
`mns/geometry/gaussians.py`: RGB is read from the keyframe, not from any neural tensor.

---

## 2. Features

### 2.1 Bands

Welch PSD per epoch, 0.5 s segments, 1–50 Hz, posterior montage:

| Feature | Band | Interpretation in this project |
| --- | --- | --- |
| theta | 4–8 Hz | Effortful retrieval, lower vividness |
| alpha | 8–13 Hz | Posterior idling / inhibition |
| beta | 13–30 Hz | Active processing |
| gamma | 30–45 Hz | Local encoding / feature binding |
| gamma/theta | ratio | Primary vividness axis |
| spectral entropy | 4–45 Hz | Inverse certainty; sharp spectra → concentrated memory |
| O1/O2 gamma asymmetry | 30–45 Hz | Hemifield prior, not a coordinate |

Gamma/theta is used rather than raw gamma because amplifier gain, skull thickness and
impedance vary across subjects and sessions. A ratio cancels much of the common-mode
gain; calibration handles the rest.

### 2.2 Salience

```text
log_ratio = log10(gamma / theta)
z         = (smooth(log_ratio) − center) / scale
salience  = 0.1 + 0.9 · sigmoid(1.70 · z)
```

The default `population` calibration uses `center = −0.08`, `scale = 0.20`. These are
normative constants chosen so the demo pair lands in distinguishable regions
(≈0.50 mean for vivid, ≈0.18 for diffuse). They are **not** derived from a clinical
normative sample and should be re-fit on the target montage and amplifier before any
scientific use.

`session` calibration uses median/MAD of the recording. It is better for within-subject
contrast but forces every session to look equally vivid, so it is wrong for the paired
demo. `reference` fits on rest epochs and applies the result to task epochs.

### 2.3 Certainty

```text
H        = −Σ p(f) log p(f) / log N        over 4–45 Hz
certainty = clip((1 − H) / 0.45, 0, 1)
```

A pure tone has low entropy; a flat spectrum has high entropy. The `0.45` divisor is an
empirical ceiling for posterior EEG that keeps the spread term useful instead of pinned
at “very uncertain”. Certainty widens or tightens attention kernels independently of
salience: a vivid but fuzzy memory and a faint but sharp memory are different states.

### 2.4 Lateralisation

```text
lateralization = (P_gamma(O2) − P_gamma(O1)) / (P_gamma(O2) + P_gamma(O1))
```

Positive means right occipital dominance. Because O1/O2 are roughly contralateral to the
visual hemifield, this is used as a weak horizontal prior on the attention field
(weight `0.16 · |lat| · salience`). It is not decoded into a gaze coordinate, and the
demo does not claim it is.

---

## 3. Attention field

For each entity `i`, a normalised anisotropic Gaussian sits at the box centre:

```text
σ_u = clip(0.62 · box_width  · spread, 0.028, 0.32)
σ_v = clip(0.62 · box_height · spread, 0.028, 0.32)
spread = 0.75 + 0.90 · (1 − certainty)
M_raw(u,v) = Σ_i salience_i · exp(−½[(u−u_i)²/σ_u² + (v−v_i)²/σ_v²])
```

If a SAM mask exists, the blob is multiplied by `0.18 + 0.82·mask`; memory keeps a halo
rather than a pixel-exact cut-out.

The raw field is normalised to peak 1, then:

```text
contrast = 1.05 + 1.30 · salience
M(u,v)   = 0.10 + 0.90 · M_raw(u,v)^contrast
```

The split matters:

- **Shape** is normalised, so the *where* is stable across epochs.
- **Contrast** uses absolute salience, so the *how much* survives normalisation.

If both were normalised per frame, vivid and distracted epochs would both produce a
bright blob and the neural variable would be invisible. This is the most common failure
mode in EEG-conditioned rendering demos.

Per-entity salience is:

```text
prior_i  = base_salience_i^0.6
vivid    = 0.30 + 0.70 · epoch_salience
hemi_i   = 1 + 0.22 · lateralization · (2·u_center_i − 1)
focus_i  = 1.14 if entity is in the beat's language focus list else 1.0
sal_i    = clip(prior_i · vivid · hemi_i · focus_i, 0.1, 1.0)
```

The language prior is capped at +14%: a hallucinated scene graph cannot fabricate
salience the EEG did not show.

---

## 4. The control: zero ground-truth leakage

### 4.1 What the pipeline sees

For a given episode:

- language model sees the transcript, not the target image;
- image model sees the sanitised scene graph, not the target image;
- depth model sees the synthesised keyframe, not the target image;
- EEG stage sees the recording and the scene graph, not the keyframe or target;
- lifter sees keyframe RGB, depth and attention field.

No stage sees a held-out photograph of the remembered place.

### 4.2 The paired-condition control

`run_pair()` renders focus and distracted as follows:

1. Render the first condition fully.
2. Serialise its `SceneGraph`, save its keyframe PNG and depth array.
3. Pass all three to the second condition.
4. Change **only** the EEG scenario (`focus` vs `distracted` schedule and seed).

The second manifest records:

```json
"upstream": {
  "scene_graph": "shared:paired-condition",
  "keyframe": "shared between conditions",
  "depth": "shared"
}
```

The test suite asserts byte-identical keyframes, array-identical depth and identical
object boxes. A judge can reproduce this with `tests/test_pipeline.py`.

### 4.3 The procedural-depth nuance

When the offline renderer produces the keyframe, it also returns the true depth of *that
synthesised frame*. This is an oracle for our own render, not for the memory. It removes
monocular-depth error from the offline rehearsal without leaking any target image. In
the live path, Muse Image is followed by DA-V2; the manifest labels each frame's
`depth_source`.

### 4.4 What would count as leakage

- using a target photo to prompt the image model;
- using target depth to lift a generated keyframe;
- giving one condition a better prompt, camera or depth model;
- calibrating each condition separately with `session` mode;
- selecting the best seed per condition after seeing the result.

The first three are blocked in code. The fourth is a configuration trap; the CLI default
is `population`. The fifth is a protocol issue: seeds are fixed in the scenario
definition, and the cache key includes the seed.

---

## 5. Why this might matter for PTSD (and what we are not saying)

### 5.1 The clinical observation

Intrusive memories in PTSD are often experienced as vivid, sensory, “here-and-now”
fragments. Imaginal exposure and memory rescripting ask a patient to revisit the memory
with the therapist, titrating vividness and emotional engagement. Clinicians do this
with words, tones and pacing; they have no instrument that shows how vivid the memory is
to the patient in the moment.

### 5.2 The speculative product idea

A system that parameterises vividness could, in principle:

- give the patient an external, manipulable representation of “how strong this is”;
- give the therapist a shared object to point at (“can we bring the lamp down?”);
- record a session's trajectory as a reviewable artefact;
- support habituation by showing that vividness can be reduced without erasing the memory.

MnemoSplat is a **visualisation prototype** for that idea. It is not therapy, not a
diagnostic, and not validated on clinical populations.

### 5.3 What would be required before any clinical claim

1. **Recorded EEG** from the target population, with a validated montage and amplifier.
2. **IRB approval**, informed consent, and a trauma-informed protocol; eliciting trauma
   memories is not a hackathon task.
3. **Clinician co-design** and adverse-event monitoring.
4. **Validated instruments** (e.g. PCL-5, CAPS-5) and a pre-registered analysis.
5. **A held-out ground-truth protocol** that separates memory content from EEG condition.
6. **A randomised design** comparing the visualisation against treatment as usual, with
   enough power to detect small effects.
7. **Regulatory review** if the tool is marketed for treatment.

Until then, the correct framing is: *a new way to render the vividness dimension that
trauma-focused therapies already manipulate*. That is a research question, not a claim.

---

## 6. Validation protocol for the scientific claim

The claim in §1 is testable without a clinical population.

### 6.1 Participants and design

- Within-subject, counterbalanced order of focus/distracted (or high/low arousal)
  conditions.
- Recorded posterior EEG; same montage and amplifier for all participants.
- A ground-truth image set (e.g. THINGS-EEG2) for content; a separate narrative set for
  the memory task.
- Pre-register the salience metric, calibration mode, epoch length and exclusion rules.

### 6.2 Manipulation check

Show that the EEG feature separates the intended conditions:

- gamma/theta and certainty distributions by condition;
- effect size with mixed-effects model (condition fixed, subject random);
- permutation test on condition labels;
- calibration sensitivity: population vs session vs reference.

### 6.3 Rendering check

Show that the neural variable, not a confound, changes the volume:

- density/opacity/scale statistics per epoch;
- a classifier that tries to predict condition from splat statistics **with the
  attention field ablated** (should be at chance if only EEG changed);
- a classifier on the attention field alone (should succeed);
- human ratings of vividness/depth on paired renders, blind to condition.

### 6.4 Ablations

| Ablation | Expected result |
| --- | --- |
| Shuffle EEG epochs | Condition effect disappears |
| Phase-randomise EEG (preserve spectrum) | Gamma/theta effect attenuates if phase matters; tests whether the feature is just spectral power |
| Fixed field for both conditions | No visual difference |
| Session calibration per condition | Contrast collapses toward equal means |
| Language prior only (no EEG) | Objects differ by recipe salience, not condition |
| Remove SAM masks | Field softens; condition effect should remain |

### 6.5 Metrics

- **Neural:** band ratios, entropy, lateralisation, effect sizes, test–retest.
- **Geometric:** splat count, mean opacity, scale distribution, field entropy.
- **Perceptual:** paired forced-choice “which is more vivid?”, depth ratings, motion
  sickness (SSQ) in VR.
- **Task:** free recall / recognition of scene objects, confidence, response time.
- **Clinical (only with approvals):** PCL-5/CAPS-5 trajectories, therapist usability.

### 6.6 What would falsify the claim

- The condition effect disappears when the language prior is held constant and EEG is
  shuffled.
- Human raters cannot distinguish conditions better than chance.
- The effect is explained entirely by splat count, and fixing count removes it.
- Session calibration erases the effect, indicating the population constants were doing
  the work.

A negative result is still a useful system finding: it would tell us that the mapping
from EEG to density is too weak to survive a rigorous control.

---

## 7. Limitations

1. **Synthetic EEG.** The demo uses a generator with 1/f background, posterior alpha,
   theta and gamma bursts, and a known focus schedule. It is realistic enough to exercise
   the PSD code; it is not a subject. Every manifest says `synthetic:v1`.
2. **Population constants.** `center/scale` are demo constants, not clinical norms.
3. **Single-view geometry.** The back of every object is invented. The ±16° parallax arc
   is deliberately short; a 360° orbit exposes the sheet.
4. **Scale ambiguity.** DA-V2 predicts relative inverse depth; the 0.85–6.2 m mapping is
   a plausible room, not metric truth.
5. **No headset validation.** WebXR code was exercised in desktop Chromium, not on a
   physical Quest. Latency, comfort and controller ergonomics are unmeasured.
6. **Small-N aesthetic choices.** The procedural renderer's palette, blur and grain are
   authored. They make the offline demo legible; they are not a perceptual model.
7. **Language model variability.** Different samples produce different scene graphs. The
   sanitiser and recipe fallback reduce, but do not eliminate, run-to-run variation.
8. **No clinical data.** No patient, no trauma, no diagnosis, no treatment claim.

---

## 8. Relationship to prior work

- **BrainVis** (ICASSP 2025) provides the learned EEG encoder backbone we would use for
  the salience branch. MnemoSplat deliberately uses only its embedding, not its image
  generator. See [BRAINVIS.md](BRAINVIS.md).
- **THINGS-EEG2** establishes a large-scale EEG/image benchmark and the posterior montage
  conventions used here.
- **3D Gaussian Splatting** supplies the representation; **gsplat** is the CUDA
  reference for refinement; **GaussianSplats3D** is the browser/WebXR renderer.
- **Depth-Anything-V2** supplies monocular depth for the live keyframe path.
- EEG-to-image projects such as `visual-reconstruction-eeg` and
  `EEG-Image-Reconstruction` informed the feature choices, but MnemoSplat intentionally
  stops short of pixel regression.

Links:

- <https://github.com/RomGai/BrainVis>
- <https://github.com/IsaacVanB/visual-reconstruction-eeg>
- <https://github.com/desa-lab/EEG-Image-Reconstruction>
- <https://github.com/DepthAnything/Depth-Anything-V2>
- <https://github.com/nerfstudio-project/gsplat>
- <https://github.com/mkkellogg/GaussianSplats3D>

---

## 9. One-paragraph summary for a judge

We do not claim to decode the image in your head. We claim something smaller and more
defensible: posterior gamma/theta and spectral certainty predict how vividly a memory
was encoded, and that scalar can drive the density and spread of a 3D Gaussian volume
without ever touching its colours. The demo holds the transcript, scene graph, keyframe,
depth map and camera fixed across two conditions; only the EEG changes. That is the
experiment. Everything else — Muse Spark, Muse Image, SAM 3.1, DA-V2, WebXR — exists to
make the experiment visible in a headset.
