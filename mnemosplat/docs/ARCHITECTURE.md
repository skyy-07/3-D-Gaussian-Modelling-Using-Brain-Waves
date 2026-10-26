# Architecture

MnemoSplat is a five-stage CPU pipeline plus a thin WebXR client. The pipeline is
deliberately boring: deterministic numpy/Pillow code, one JSON contract, and explicit
fallbacks. The interesting decisions are about **what is allowed to influence what**.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. DIRECTOR                                                                 │
│    transcript ──► Muse Spark (or keyword recipe)                            │
│    output: SceneGraph {objects[], beats[], palette, style}                  │
│    gate:   sanitize_graph() clamps boxes, fixes depth contradictions,       │
│            normalises beats to [0,1], drops duplicate ids                   │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────────┐
│ 2. KEYFRAME + DEPTH                                                         │
│    SceneGraph ──► Muse Image (or procedural renderer) ──► keyframe.png      │
│    keyframe   ──► DA-V2 (or oracle of the procedural frame) ──► depth.npy   │
│    keyframe   ──► SAM 3.1 (or recipe masks) ──► masks                       │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │  keyframe, depth, masks are CONDITION-INVARIANT
┌───────────────────────────────▼─────────────────────────────────────────────┐
│ 3. NEURAL CONDITION                                                         │
│    EEG recording ──► Welch PSD ──► band powers ──► robust calibration       │
│    output per epoch: salience, certainty, gamma/theta, lateralisation       │
│    SceneGraph + epoch ──► per-object salience ──► attention field M(u,v)    │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────────┐
│ 4. LIFT + EXPORT                                                            │
│    RGB × depth × M(u,v) ──► camera-space points ──► three.js frame          │
│    scale = z / f × stride; opacity = f(M, global salience); prune α<0.05    │
│    salience-weighted cap ──► environment shell ──► .ply + .splat            │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────────┐
│ 5. DELIVERY                                                                 │
│    manifest.json + frames/*.splat ──► FastAPI /assets ──► WebXR client      │
│    one keyframe re-lifted per epoch; timeline changes density, not camera   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Director

**Input:** free-form first-person transcript.
**Output:** `SceneGraph` (`mns/schemas.py`).

Muse Spark is asked for 4–8 concrete objects with normalised boxes, `depth_hint`,
`base_salience`, and 3–6 beats. The prompt forbids people, text, brands and diagnostic
content (`mns/director/prompts.py`). If the model returns fewer than three usable
objects, the offline recipe wins.

The sanitiser (`mns/director/layout.py`) runs on *every* graph, including offline ones:

| Repair | Why |
| --- | --- |
| Clamp boxes into `[0,1]`, enforce 0.045 minimum side | A box leaving frame becomes a floating Gaussian wall |
| Rename/drop duplicate ids | Salience is keyed by id |
| Normalise beats to a contiguous 0–1 timeline | Client maps beats to epoch count |
| Filter beat `focus` to surviving ids | A hallucinated id would be silently ignored later |
| If `v1 > 0.9` and `depth_hint > 0.62`, pull depth to 0.55 | Bottom-of-frame clutter cannot be the far wall |
| Default palette of 5 | The procedural renderer needs light colours |

Overlap is **reported, not removed**: a mug sits on a desk, a curtain hangs in front of
a window. Occlusion is resolved by depth.

Offline recipes (`mns/director/recipes.py`) cover study, kitchen, bedroom, street and a
generic room. Each is authored as a plausible 512×512 photograph at eye height. Keyword
matching chooses one; the user's transcript is preserved in `SceneGraph.narrative`.

---

## 2. Keyframe, depth, masks

One keyframe per episode is a product decision, not a shortcut. It guarantees that the
timeline changes *only* cognitive density.

### Providers

| Asset | Production | Offline | Manifest field |
| --- | --- | --- | --- |
| Keyframe | `muse-image-1.0` | `procedural/v1` scene-graph render | `upstream.keyframe` |
| Depth | DA-V2 Small via `transformers` | oracle depth of the procedural frame | `upstream.depth`, `frame.depth_source` |
| Masks | SAM 3.1 | exact per-object masks from the procedural renderer | `upstream.segmentation` |

DA-V2 predicts relative inverse depth; the adapter maps it monotonically into
0.85–6.2 m. Absolute scale changes the apparent size of the volume, not splat *shape*
(scale is proportional to depth). The manifest never calls it metric truth.

The procedural renderer returns a metric depth map and per-object alpha masks alongside
the RGB. That depth belongs to the synthesised keyframe, not to a held-out target
image, so it does not violate the no-leakage control.

### Lift input conditioning

The lifter receives a 3×3 median-filtered, 0.55 px Gaussian-blurred copy of the
keyframe. Procedural grain looks good in the photo and terrible as per-splat colour
noise. Edges and window bars survive; speckle does not. The saved keyframe and the
manifest still reference the unblurred image.

### SAM refinement

`refine_with_sam` sorts objects by box area, asks SAM 3.1 for up to five objects, and
matches the best segment to the existing box by IoU (minimum 0.08). Boxes are blended
(`0.45` toward the segment), never replaced: a bad segmentation cannot throw an object
out of frame. Mask payloads are decoded only if the declared raster length matches;
otherwise the object falls back to a feathered box mask.

---

## 3. Neural condition

### Feature extraction

`mns/eeg/psd.py`:

1. Pick posterior channels (`O1, O2, Oz, P3, P4, Pz`) when present.
2. Welch PSD per epoch, 0.5 s segments, 1–50 Hz, constant detrend.
3. Integrate theta (4–8), alpha (8–13), beta (13–30), gamma (30–45).
4. `gamma_theta = gamma / theta`.
5. Spectral entropy over 4–45 Hz → `certainty = clip((1 − H/Hmax) / 0.45, 0, 1)`.
6. O1/O2 gamma asymmetry → `lateralization ∈ [−1, 1]`.
7. Smooth `log10(gamma/theta)` with a 3-tap `[0.25, 0.5, 0.25]` kernel.
8. Calibrate to salience with a logistic; default `population` mode.

The three calibration modes are load-bearing:

| Mode | Centre/scale | Use |
| --- | --- | --- |
| `population` | fixed normative `(−0.08, 0.20)` | **Demo default**; keeps focus and distracted absolutely comparable |
| `session` | median/MAD of this recording | Best within-subject contrast; destroys between-condition comparison |
| `reference` | fitted on the first N rest epochs | Task vs baseline studies |

Using session calibration for the paired demo would force both conditions to look
equally vivid. That is the single easiest way to accidentally erase the result.

### Attention field

`mns/fusion/attention.py` composes anisotropic Gaussians, one per entity, plus an
optional hemifield blob. Each blob is weighted by entity salience and widened by
uncertainty (`0.75×` at full certainty, `1.65×` at zero). SAM masks multiply the blob
(`0.18 + 0.82·mask`) so memory keeps a small halo rather than a pixel-exact cut-out.

Two things are deliberately separated:

- **Shape** comes from the normalised field (where attention sits).
- **Contrast** comes from the absolute salience scalar (`1.05 + 1.30·salience`).

A naive implementation normalises every frame to peak 1.0. Then vivid and distracted
epochs both produce a bright blob, and the neural signal is gone. Here a low-salience
field has a higher ambient mean and a flatter exponent; a vivid field is dark around a
tight peak.

Per-entity salience starts from the recipe/LLM prior, is scaled by epoch vividness,
biased by the hemifield prior, and boosted at most 14% by the beat's language `focus`
list. A hallucinated scene graph cannot fabricate salience the EEG did not show.

---

## 4. Lift and export

### Camera model

`CameraModel.from_fov(512, 512, 60°)` gives square pixels:

```text
fx = fy = (w/2) / tan(hfov/2)
cx = (w−1)/2,  cy = (h−1)/2
x = (u − cx)·z / fx
y = (v − cy)·z / fy   # camera Y points DOWN
z = depth(u,v)        # metres, forward
```

### Why scale is `z/f × stride`

A Gaussian must cover its own pixel footprint at its own depth. Constant scale makes
near objects mush and far objects holes. `surfel_scale = 1.02`, thickness `0.32`, cap
`0.09 m`.

### Cognitive density

```text
spread  = 1 + 0.95·(1−M) + 0.45·(1−global_salience)
s_xy    = clip(1.02 · (z/fx) · stride · spread, 1e-4, 0.09)
s_z     = 0.32 · s_xy
base    = clip(0.48 + 0.48·global_salience, 0.45, 0.97)
α       = base · (0.06 + 0.94·M^contrast)
```

Splats with `α < 0.05` are pruned. If pruning leaves fewer than 16, the code keeps all
rather than emit an empty frame.

### Budget

The sampling grid uses an integer stride chosen so `h·w/stride² ≈ 1.35 × hard_cap`.
After pruning, a deterministic salience-weighted selection keeps focused geometry dense
and thins the haze:

```text
p = 0.35 + 0.65·M
P(keep) = min(1, p · target / Σp)
```

Hard cap `262,144` (512×512). Demo target `120,000`. A faint environment shell
(72 × 28 directions at 7.5 m, α 0.14) is added afterwards so turning the head in VR is
not the void.

### Coordinate frames

| Stage | Frame | Axes |
| --- | --- | --- |
| Pixels | image | u right, v down, origin top-left |
| Unprojection | camera | X right, **Y down**, Z forward |
| Export / client | three.js / WebXR | X right, Y up, **−Z forward**, metres |

Conversion is a 180° rotation about X:

```text
[1  0  0]
[0 −1  0]
[0  0 −1]
```

Each splat is oriented as a surfel facing the camera (`basis_facing`). Quaternions are
`(w,x,y,z)`, unit length.

The cloud is then recentred on the origin. The camera position is `−centroid`, which
lands at roughly `(0, 0, +median depth)` in three.js space. The manifest records both
`camera.position` and `camera.look_at`.

### Formats

- **`.ply`**: INRIA 3DGS binary little-endian; 17 float32 fields; `f_dc` colour,
  logit opacity, log scale, `(w,x,y,z)` rotation. Frame 0 only by default.
- **`.splat`**: 32 bytes/splat: position 3f, scale 3f, RGBA 4u8, quaternion 4u8
  (`q·128+128`, rounded to nearest).

Both are written from the same arrays, so researchers can open the `.ply` in
gsplat/Blender while the headset loads the `.splat`.

The 8-bit quaternion quantisation is the only rotation precision the headset gets;
round-to-nearest avoids the downward bias of truncation.

---

## 5. Delivery

### Manifest contract

`manifest.json` is the only wire contract. `mns/schemas.py` is its source of truth.

```jsonc
{
  "id": "episode-cache-key",
  "title": "The desk by the window",
  "scenario": "focus",
  "scene_graph": { "objects": [], "beats": [], "palette": [], "..." : "..." },
  "frames": [
    {
      "index": 0,
      "t": 0.0,
      "label": "enter the room · 0.00s",
      "splat": "frames/000.splat",
      "ply": "frames/000.ply",              // frame 0 only
      "keyframe": "keyframe.png",
      "attention": "frames/000.attention.png",
      "camera": { "position": [x,y,z], "look_at": [0,0,0], "fx": 443.4, "..." : 0 },
      "epoch": { "salience": 0.43, "certainty": 0.62, "lateralization": -0.2, "..." : 0 },
      "object_salience": { "mug": 0.81, "window": 0.64 },
      "objects": [ { "id": "mug", "bbox": {}, "depth_hint": 0.26 } ],
      "n_splats": 121747,
      "depth_source": "oracle-keyframe",
      "bounds": { "min": [], "max": [] }
    }
  ],
  "upstream": {
    "scene_graph": "heuristic:recipe",
    "eeg": "synthetic:v1",
    "eeg_calibration": "population:center=-0.080",
    "keyframe": "procedural(v1)",
    "depth": "oracle-keyframe",
    "segmentation": "scene-graph masks (exact)"
  },
  "notes": [],
  "stats": {},
  "objects": [],
  "world_scale_m": 1.0
}
```

`upstream` is intentionally plain strings, not an enum: a new provider should be
visible to the UI immediately.

### API

| Route | Purpose |
| --- | --- |
| `GET /api/health` | key/model/depth/BrainVis/web build capability |
| `GET /api/scenarios` | built-in narratives |
| `GET /api/scenes` | `out/index.json`, or manifests discovered on disk |
| `POST /api/episode` | queue a generation job, returns `{job}` |
| `GET /api/jobs/{id}` | poll state, warnings, elapsed |
| `POST /api/segment` | SAM on an uploaded frame |
| `POST /api/transcribe` | Muse Voice on an uploaded clip |
| `POST /api/selftest` | tiny CPU episode |
| `GET /assets/*` | manifests, splats, PNGs |
| `GET /` | built client from `web/dist` |

Generation runs in a single-worker thread pool: Muse Image can take 30–60 s, and the UI
must not block. Jobs live in memory; this is a demo server, not a queue service.

FastAPI sets `Cross-Origin-Opener-Policy: same-origin`,
`Cross-Origin-Embedder-Policy: require-corp`, and
`Cross-Origin-Resource-Policy: same-origin` on every response. Vite sets the same in
dev/preview. Without them `SharedArrayBuffer` is unavailable and the splat sorter falls
back to copying.

### Caching

`PipelineConfig.cache_key()` hashes narrative, scenario, dimensions, target splats,
Muse/ per-beat flags, seed and a version integer. An episode is reused when
`manifest.json` and `.episode.json` exist with the same key and `force=False`. Changing
any capacity knob changes the hash.

`run_pair` always forces a fresh pair because it needs the first condition's keyframe
and depth to seed the second.

### Client

`web/src/main.js` owns state, data loading, HUD rendering, keyboard/XR handlers.
`web/src/stage.js` wraps GaussianSplats3D.

Important choices:

- **One render loop.** The viewer owns `renderer.setAnimationLoop`, required for
  immersive sessions. No second `requestAnimationFrame`.
- **`local` reference space.** The origin is the headset at session start; the manifest
  camera maps to `(0,0,0)`. `local-floor` would drop the room to the user's feet.
- **Move the scene, not the eye.** `setFraming` stores `-camera_position`; every load
  passes it as `position`. Desktop OrbitControls targets the same offset.
- **Remove + add on swap.** Four resident volumes would exhaust Quest memory and the
  CPU splat sort. Rapid calls are coalesced; the latest URL wins, and callers share one
  promise so the busy indicator is accurate.
- **Manual desktop controls.** Built-in controls are disabled when `webXRMode` is set,
  so the client instantiates OrbitControls itself.
- **Condition switch preserves the viewpoint.** Only the first scenario calls
  `setFraming`; switching conditions must not steal the user's camera.
- **DOM overlay is optional.** Controllers work without it: trigger next, squeeze
  previous.

### Static hosting

Vite emits hashed bundles to `web/dist/static/` (`assetsDir: 'static'`) because `/assets`
is already the demo data mount. FastAPI mounts `/assets` first, then `/` with
`html=True`.

---

## Performance budgets

| Resource | Budget | Current offline build |
| --- | --- | --- |
| Keyframe | 512×512 | 512×512 |
| Splats per frame | ≤ 262,144 | ~121,700 |
| `.splat` per frame | ≲ 8 MB | ~3.9 MB |
| Pair download | ≲ 30 MB | ~29.7 MB + previews |
| Frames resident | 1 | 1 |
| JS bundle | — | 736 KB / 190 KB gzip |
| Cold pair generation | < 2 min CPU | ~60 s on build laptop |

The shell uses a 16-bit sort distance map and integer sorting to keep the Quest sort
affordable.

---

## Failure matrix

| Failure | Behaviour | Visible where |
| --- | --- | --- |
| No `MODEL_API_KEY` | Recipes + procedural keyframe | `manifest.upstream`, About modal |
| Muse Spark returns <3 objects | Offline recipe, warning | `result.warnings`, upstream |
| Muse Image safety filter / timeout | Procedural keyframe | `upstream.keyframe` includes `[muse fallback]` |
| SAM parse/decode fails | Box masks; objects kept | `upstream.segmentation = box-fallback` |
| DA-V2 missing / OOM | Heuristic depth | `frame.depth_source = heuristic` |
| Depth has <16 usable pixels | `ValueError` from lift | Job state `error` |
| Pruning leaves <16 splats | Keep all | — |
| Splat load fails | Toast; boot error on first frame | UI |
| WebGL unavailable | Boot screen message | UI |
| WebXR unsupported | Button relabelled “headset required” | UI |
| No index.json | Client rebuilds from manifests | UI |

---

## Module map

| Module | Responsibility |
| --- | --- |
| `mns.schemas` | Wire contracts, bbox/camera maths |
| `mns.scenarios` | Demo narratives + EEG schedules |
| `mns.eeg.synthetic` | 1/f background, alpha/theta/gamma bursts |
| `mns.eeg.psd` | Welch, bands, entropy, calibration |
| `mns.eeg.brainvis` | Loader, encoder protocol, artifact check |
| `mns.director.recipes` | Offline scene graphs |
| `mns.director.prompts` | System/user prompts + JSON schema |
| `mns.director.meta_client` | HTTP, retries, SAM parsing, ASR |
| `mns.director.scene_graph` | Director orchestration, SAM refinement |
| `mns.director.layout` | Deterministic graph sanitiser |
| `mns.render` | Keyframe providers |
| `mns.render.procedural` | Offline RGB + depth + masks |
| `mns.depth` | DA-V2 / heuristic / oracle facade |
| `mns.fusion.attention` | Field, entity salience, overlays |
| `mns.geometry.camera` | Unprojection, bases, quaternions |
| `mns.geometry.gaussians` | Lift, cap, shell, recentre |
| `mns.geometry.preview` | Software rasteriser + parallax arc |
| `mns.export.ply` | `.ply` / `.splat` writers + readers |
| `mns.pipeline` | Orchestration, pair, cache, selftest |
| `mns.api` | FastAPI jobs + static |
| `mns.cli` | `demo`, `serve`, `selftest`, `doctor`, `export` |
| `web/src/stage.js` | GaussianSplats3D, WebXR, scene offset, swap queue |
| `web/src/main.js` | State, HUD, keyboard, XR overlay |
