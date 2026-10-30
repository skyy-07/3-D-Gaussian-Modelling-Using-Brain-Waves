# Implementation plan — 36-hour hackathon

**Goal:** a stage-ready demo in which a spoken memory becomes a WebXR Gaussian volume
whose density is conditioned by posterior EEG, with a paired control that a judge can
inspect in under two minutes.

**Non-goals:** decoding images from EEG in the controlled demo, clinical claims,
multi-view reconstruction, photorealism on CPU, live EEG hardware, a production
multi-tenant service.

> The optional GPU-only BrainVis static-frame branch is documented separately in
> [RECONSTRUCTION.md](RECONSTRUCTION.md). It is not part of the 36-hour controlled demo
> and does not claim video or memory reconstruction.

The plan below is ordered by demo risk, not by elegance. Every phase ends in something
that can be shown even if the next phase fails. The final section records what actually
shipped and where the plan was cut.

---

## 1. Product thesis in one sentence

Language supplies **what** is in the memory, monocular depth supplies **where**, and
posterior EEG supplies **how vividly it consolidated** — never the colours.

The demo proves the control by rendering two conditions from one transcript, one scene
graph, one keyframe and one depth map. Only the neural channel changes.

---

## 2. Demo choreography (design backwards from this)

| Time | Beat | What the audience sees |
| --- | --- | --- |
| 0:00 | Hook | Boot screen: “Same words. Different memory.” |
| 0:15 | Vivid | Dense room; mug and notebook sharp; salience 0.50 / peak 0.67 |
| 0:35 | Switch | `D`: same room, now translucent fog; 0.18 / 0.25 |
| 0:55 | Timeline | Arrow keys re-lift the same keyframe; density changes, camera does not |
| 1:15 | Evidence | “?” modal: same graph/keyframe/depth; provenance chips; synthetic EEG label |
| 1:40 | Headset | Quest: trigger advances epoch; squeeze goes back; overlay shows salience |
| 2:00 | Close | “EEG never touches a pixel. It gates how much of the memory survives.” |

Every engineering decision below exists to protect one of these beats.

---

## 3. Architecture in one screen

```text
transcript ─► Muse Spark ─► scene graph ─► Muse Image ─► keyframe ─► DA-V2 depth
                                  │                          │
EEG ─► Welch ─► calibration ─► attention field M(u,v) ───────┤
                                                             ▼
                                        single-view lift ─► .splat + manifest
                                                             ▼
                                                   WebXR / GaussianSplats3D
```

Hard constraints:

- one keyframe per episode, re-lifted per epoch;
- EEG affects opacity, covariance and density only;
- ≤ 262,144 splats, 512×512 keyframe, 120k target;
- every AI path degrades offline;
- provenance is explicit in `manifest.upstream`;
- no target image anywhere in the pipeline.

---

## 4. Workstreams

| Stream | Owner (solo or role) | Output |
| --- | --- | --- |
| A. Neural | EEG/ML | Synthetic generator, PSD, calibration, attention field |
| B. Director | LLM/app | Prompts, Meta client, recipes, layout sanitiser |
| C. Geometry | Graphics | Depth providers, lift, cap, exporters, preview |
| D. Client | Web/XR | Vite app, HUD, timeline, WebXR, frame swap |
| E. Product | Anyone | Scenarios, demo script, About modal, provenance, QA |

For a solo build, run A → C → B → D; the geometry is the riskiest unknown and the HUD
can be built against static manifests.

---

## 5. 36-hour timeline

### Hours 0–3 — skeleton and the contract

**Deliverable:** `mnemosplat selftest` produces a manifest and a `.splat`.

- Create the Python package, CLI, schemas, and a deterministic synthetic EEG generator.
- Define `manifest.json` before any UI. The client and pipeline must not drift.
- Write the smallest lift: keyframe + depth + constant field → `.splat`.
- Gate: `python -m pytest` has one geometry test and one format test.

**Cut if late:** Muse paths. The offline recipe is the demo.

### Hours 3–7 — the neural gate

**Deliverable:** focus and distracted epochs differ in salience, and the difference
survives calibration.

- Implement Welch PSD, bands, entropy, lateralisation.
- Implement `population`, `session`, `reference` calibration.
- Split field *shape* (normalised) from *contrast* (absolute salience).
- Write tests: high-focus ranks above low-focus; ranges hold; modes differ.
- Gate: focus mean > distracted mean by ≥ 0.2 on the demo scenarios.

**Cut if late:** `reference` mode; keep population + session.

### Hours 7–12 — lift quality

**Deliverable:** a preview raster that looks like the keyframe, not confetti.

- Scale = pixel footprint at depth; prune α < 0.05.
- Surfel orientation (`basis_facing`), 180° X conversion, recentre.
- Salience-weighted cap to 120k.
- Environment shell so VR is not the void.
- Software rasteriser for QA and slides.
- Gate: preview PNG is recognisably the room; no NaNs; cap respected.

**Cut if late:** environment shell (desktop demo still works, VR feels empty).

### Hours 12–16 — director and offline keyframe

**Deliverable:** a transcript becomes a sanitised graph and a plausible keyframe.

- Hand-author four recipes (study, kitchen, bedroom, street) + generic.
- Prompt + strict JSON schema for Muse Spark.
- Layout sanitiser: clamp, min side, duplicates, beats, depth contradictions.
- Procedural renderer with depth and masks.
- Gate: `mnemosplat demo --scenario focus --offline` is watchable.

**Cut if late:** three recipes; the study is the demo.

### Hours 16–20 — Meta paths and degradation

**Deliverable:** one image, one graph, SAM masks, all optional.

- `MetaModelAPI`: chat JSON, image b64, SSE segmentation, ASR.
- Retries, timeouts, clear errors.
- Every fallback writes to `manifest.upstream`.
- `doctor --probe`.
- Gate: with no key, the full pipeline still runs and labels itself offline.

**Cut if late:** SAM and ASR. The scene graph + image are the visible wins.

### Hours 20–25 — WebXR client

**Deliverable:** splats in a browser, then in a headset.

- Vite + three + GaussianSplats3D.
- COOP/COEP; `assetsDir: 'static'` to avoid `/assets` collision.
- `local` reference space; move scene by `-camera_position`.
- Remove + add frame swap; coalesce rapid scrubs.
- OrbitControls manually.
- Gate: desktop Chromium renders; `crossOriginIsolated === true`.

**Cut if late:** WebXR. A desktop orbit demo is still strong; the headset is the bonus.

### Hours 25–29 — HUD and story

**Deliverable:** the audience can see the evidence without narration.

- Salience dial + sparkline, band bars, telemetry, ranked entities.
- Keyframe / salience-field thumbnails; lightbox.
- Timeline with beat ticks; play.
- Provenance chips; About modal with the control and the synthetic-EEG disclaimer.
- Gate: a stranger can explain the demo after 30 seconds of looking.

**Cut if late:** sparkline, telemetry details, lightbox.

### Hours 29–32 — paired control and QA

**Deliverable:** `run_pair` shares graph/keyframe/depth and proves it.

- Second condition receives serialised graph, PNG and depth array.
- Tests assert byte-identical keyframes and array-identical depth.
- Manifest records `shared:paired-condition`.
- Regenerate demo assets; verify no console/HTTP errors.
- Gate: manifests diff only in EEG fields and derived salience/density.

### Hours 32–35 — polish and rehearsal

**Deliverable:** a repeatable 2-minute run.

- Pre-generate the pair; never depend on live network.
- Quest over `adb reverse`; test controller mapping.
- Rehearse the script three times; time it.
- Capture fallback screenshots in case the projector/browser misbehaves.
- Gate: a full run with Wi-Fi off.

### Hour 35–36 — freeze

**Deliverable:** tag, README, env checklist, charged headset.

- No new features. Only fixes that protect the script.
- Record a 60-second screen capture as a backup.
- Print the runbook: commands, URLs, key env vars.

---

## 6. Meta mapping

| Need | Model | Call | Fallback |
| --- | --- | --- | --- |
| Scene graph | `muse-spark-1.3` | `/chat/completions` JSON schema | Recipes |
| Keyframe | `muse-image-1.0` | `/images/generations` b64 | Procedural render |
| Masks | `sam-3.1` | `/responses` SSE | Box masks |
| Voice | `muse-voice-transcribe-1.0` | `/asr/transcribe` | Typed transcript |
| Harder text | `muse-wonder-1.0` | `MNS_TEXT_MODEL` | same recipes |

Cost per pair: one image (`$0.01` list at build), one text call, ≤5 segment calls.
Base URL `https://api.meta.ai/v1`, key `MODEL_API_KEY`. Details:
[META_INTEGRATION.md](META_INTEGRATION.md).

---

## 7. Risk register and cut ladder

| Rank | Risk | Early warning | Mitigation | Cut |
| --- | --- | --- | --- | --- |
| 1 | Splats look like confetti | Preview is noisy | Footprint scale, prune, median-filter lift input, cap | Reduce to 60k and tighten FOV |
| 2 | EEG difference invisible | Both conditions look bright | Absolute contrast, population calibration, HUD numbers | Add salience dial; narrate numbers |
| 3 | Headset unavailable / WebXR fails | No device, browser flags | Desktop orbit + video capture | Cut live headset beat |
| 4 | No API key / network dies | `doctor` shows false | Offline pair pre-generated | `--offline` |
| 5 | Muse Image looks wrong | Weird room, safety filter | Prompt + procedural fallback | Procedural keyframe |
| 6 | Quest performance | Low FPS, heat | One frame resident, 120k cap, 16-bit sort | Drop to 80k |
| 7 | Timeline tears geometry | Visible popping | One keyframe re-lifted | Disable play, manual epochs |
| 8 | Judge asks “is this real EEG?” | — | Explicit synthetic chip + manifest | Say it first |

Cut order if time runs out: ASR → SAM → per-beat keyframes → environment shell → WebXR →
sparkline → telemetry → procedural polish. Never cut the paired control or the
provenance chips.

---

## 8. Acceptance criteria

A build is demo-ready when all of these are true:

- [ ] `mnemosplat demo --scenario both --offline --force` completes on a laptop CPU.
- [ ] Both manifests have 4 epochs, ~120k splats, and clear salience separation.
- [ ] Keyframe bytes and depth arrays are identical across conditions.
- [ ] `mnemosplat serve` hosts client, API and assets on one origin.
- [ ] Desktop Chromium loads, switches conditions and epochs, opens About and lightbox.
- [ ] No console errors, no failed requests, `crossOriginIsolated === true`.
- [ ] Quest (or emulator) enters VR; trigger/squeeze change epochs.
- [ ] About modal states synthetic EEG, one shared keyframe, and the no-leakage control.
- [ ] With the network off, the pre-generated demo still runs.
- [ ] `pytest` and `ruff` pass.
- [ ] The 2-minute script has been rehearsed three times.

---

## 9. What actually shipped

| Planned | Status |
| --- | --- |
| Python package, CLI, schemas, tests | Shipped |
| Synthetic EEG, PSD, calibration, attention field | Shipped |
| Four recipes + generic, sanitiser, prompts | Shipped |
| Procedural keyframe + depth + masks | Shipped |
| Gaussian lift, cap, shell, `.ply`/`.splat`, preview, turntable | Shipped |
| FastAPI jobs + static hosting | Shipped |
| Vite + three + GaussianSplats3D HUD | Shipped |
| WebXR, controller mapping, DOM overlay | Implemented; not headset-verified |
| Meta client (text/image/SAM/ASR) + degradation | Implemented; not key-verified |
| BrainVis bridge | Interface + artifact check; checkpoints absent |
| Paired control | Shipped and tested |
| Docs | Shipped |

Deviations from the original plan, and why:

- **Turntable is ±16°, not 360°.** A single-view lift is a sheet; a full orbit advertises
  the failure. A short parallax arc proves depth without breaking the illusion.
- **Lift input is median-filtered.** Procedural grain became per-splat colour noise.
  The saved keyframe stays sharp; only the lifter sees the smoothed copy.
- **`population` calibration is the default.** Session calibration would force both
  conditions to look equally vivid and erase the demo's result.
- **The scene is moved, not the camera, in WebXR.** It keeps desktop orbit and headset
  origin consistent and centres the environment shell on the viewer.
- **Frames are removed before adding.** Quest memory and the CPU sort are the scarce
  resources.

---

## 10. Post-hackathon roadmap

1. **Recorded EEG.** Swap the synthetic generator for a real recording; validate montage,
   preprocessing and calibration on the target amplifier.
2. **BrainVis salience head.** Follow [BRAINVIS.md](BRAINVIS.md); leave-one-subject-out
   evaluation against the spectral baseline.
3. **GPU refinement.** Optional `gsplat` pass on the lifted cloud for research exports;
   the WebXR path stays single-view.
4. **Perceptual study.** Blind paired comparisons of vividness, depth and comfort.
5. **Clinical co-design.** Only with IRB, clinicians and a trauma-informed protocol.
6. **HTTPS deployment.** Caddy or a tunnel for Wi-Fi Quest testing; lock down CORS.
7. **Streaming assets.** Progressive `.splat` loading if the frame count grows beyond
   the current four.
8. **Accessibility.** Reduced-motion mode, contrast controls, seated/standing reference
   spaces, captions for the narrative.

---

## 11. Runbook (paste this on the day)

```bash
cd mnemosplat
. .venv/bin/activate

# Pre-flight
mnemosplat doctor
python -m pytest -q
ruff check src tests

# Demo assets (already generated; only if needed)
mnemosplat demo --scenario both --out out/demo --splats 120000 --force

# Client
cd web && npm run build && cd ..

# Serve
mnemosplat serve --host 0.0.0.0 --port 8000

# Quest over USB
adb reverse tcp:8000 tcp:8000
# open http://127.0.0.1:8000
```

If the network is unavailable, use the pre-generated `out/demo` pair and do not pass
`--muse`. If the headset is unavailable, use the desktop orbit and the recorded backup
video. If a judge asks whether the EEG is real, say “synthetic, labelled in the manifest”
before they finish the question.
