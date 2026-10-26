# Meta integration

MnemoSplat uses the **Meta Model API** at `https://api.meta.ai/v1` with bearer auth from
`MODEL_API_KEY`. The client is intentionally small: `httpx`, JSON, one SSE parser, and
explicit timeouts. Every call has an offline fallback, and every fallback is recorded in
the manifest.

> **Verification status.** The code paths below were implemented against the documented
> wire formats, but this checkout had no `MODEL_API_KEY`, so no live request was made.
> The offline demo, parsers and degradation paths are covered by tests. Treat the model
> IDs and request fields as integration targets to re-verify against the current API
> console before a live demo.

---

## Why the Meta Model API

The retired `api.llama.com` preview is not used. `https://api.meta.ai/v1` is the single
base URL for the models this demo needs, and the surface is close enough to the
OpenAI conventions that a thin client is sufficient:

- chat-completions style structured JSON for the scene graph;
- image generation returning `b64_json`;
- a streaming Responses endpoint for SAM 3.1;
- multipart audio transcription for spoken memories.

Keeping one base URL means one key, one timeout policy, one retry policy and one
provenance story.

---

## Model map

| Model ID (default) | Role | Code path | Offline fallback |
| --- | --- | --- | --- |
| `muse-spark-1.3` | Transcript → `SceneGraph` JSON | `MetaModelAPI.chat_json` → `build_scene_graph` | `director/recipes.py` |
| `muse-image-1.0` | Photographic keyframe | `MetaModelAPI.generate_image` → `render.muse_keyframe` | `render.procedural.render_scene` |
| `sam-3.1` | Boxes + masks on the keyframe | `MetaModelAPI.segment` → `refine_with_sam` | Recipe masks / feathered boxes |
| `muse-voice-transcribe-1.0` | Spoken memory → text | `transcript_from_audio` | Typed transcript |
| `muse-wonder-1.0` | Optional harder narratives | set `MNS_TEXT_MODEL=muse-wonder-1.0` | same recipes |

The demo path calls **one** image model per episode, shared by both conditions.

---

## Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `MODEL_API_KEY` | unset | Presence is the only switch that enables live calls |
| `MNS_BASE_URL` | `https://api.meta.ai/v1` | Override for a proxy or staging |
| `MNS_TEXT_MODEL` | `muse-spark-1.3` | Director |
| `MNS_IMAGE_MODEL` | `muse-image-1.0` | Keyframe |
| `MNS_SEG_MODEL` | `sam-3.1` | Segmentation |
| `MNS_ASR_MODEL` | `muse-voice-transcribe-1.0` | Voice |
| `MNS_LOG` | `INFO` | `DEBUG` shows retry warnings |

`mnemosplat doctor` prints the effective values. `--probe` calls `GET /models` and lists
up to 20 IDs. It tolerates both `{data:[...]}` and `{models:[...]}` response shapes.

---

## 1. Scene graph — `muse-spark-1.3`

### Request

`POST /chat/completions`

```json
{
  "model": "muse-spark-1.3",
  "messages": [
    {"role": "system", "content": "You are the spatial memory director ..."},
    {"role": "user", "content": "Memory transcript: ... Produce the scene graph as JSON ..."}
  ],
  "temperature": 0.6,
  "max_tokens": 3000,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "mnemosplat_scene_graph",
      "strict": true,
      "schema": { "...": "..." }
    }
  }
}
```

The schema (`mns/director/prompts.py`) requires:

- title, style, lighting, palette;
- 3–8 objects, each with id, label, normalised bbox, depth hint, base salience, prompt;
- 3–6 beats, each with label and focus ids.

If the hosted model rejects `json_schema`, the client retries once with
`{"type":"json_object"}`. JSON is extracted defensively from fences or surrounding prose.

### Validation

Nothing from the model reaches the renderer unvalidated:

1. `_coerce_graph` drops malformed objects, coerces numerics, slugs ids, de-duplicates.
2. Fewer than three usable objects → warning + offline recipe.
3. `sanitize_graph` clamps boxes, repairs depth contradictions, normalises beats, drops
   unknown focus ids.

### Provenance

`manifest.upstream.scene_graph` is either `meta:<model-id>` or `heuristic:recipe`.
The paired second condition records `shared:paired-condition` because it receives the
first condition's serialised graph.

---

## 2. Keyframe — `muse-image-1.0`

### Request

`POST /images/generations`

```json
{
  "model": "muse-image-1.0",
  "prompt": "First-person photograph of a remembered room. ...",
  "n": 1,
  "size": "512x512",
  "response_format": "b64_json",
  "output_format": "png",
  "reasoning_strength": "high",
  "tool_enablement": {"enable_web_search": false, "enable_shell": false}
}
```

The prompt is camera-first and object-complete: style, lighting, back-to-front object
list with normalised centres, no people/text/watermark, target aspect. It is built by
`keyframe_prompt()` from the sanitised graph, so a model box and the image prompt cannot
disagree about what is in the room.

The response is centre-cropped to the target aspect (never letterboxed) and resized with
Lanczos. Safety-filtered or empty responses raise `MetaAPIError`; the pipeline falls back
to the procedural renderer and records `[muse fallback]`.

### Why one image

One image per episode is the core control:

- focus and distracted share the same photons, geometry prompt and camera;
- cost stays `$0.01` per memory instead of `$0.04–0.06`;
- the timeline changes only density.

`PipelineConfig(per_beat_keyframes=True)` is reserved for an experimental mode; it is
not the demo.

---

## 3. Segmentation — `sam-3.1`

### Request

`POST /responses`, streaming SSE:

```json
{
  "model": "sam-3.1",
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": [
        {"type": "input_text", "text": "ceramic mug include_confidence=true"},
        {"type": "input_image", "image_url": "data:image/png;base64,..."}
      ]
    }
  ],
  "stream": true
}
```

The client concatenates `response.output_text.delta` events and parses the SAM wire
grammar:

```text
0<|box;x1=10;y1=20;x2=110;y2=120;w=512;h=512;c=0.91|>
```

Masks, when present, follow the box as a compressed raster token. `decode_mask` validates
the declared `h × w` length after A85 + zlib decode; a format change degrades to boxes
instead of corrupting the attention field.

### Matching

Objects are processed largest-box first, up to five. The best segment is the one with
the highest IoU against the language model's box; below `0.08` it is rejected. Matching
boxes are blended 45% toward the segment, so a bad mask cannot eject an object.

Provenance: `sam-3.1`, `box-fallback`, or `scene-graph masks (exact)` for the offline
renderer.

---

## 4. Voice — `muse-voice-transcribe-1.0`

`POST /asr/transcribe` (fallback `/audio/transcriptions`), multipart:

```text
model=muse-voice-transcribe-1.0
file=@memory.webm
```

The API endpoint `POST /api/transcribe` writes the upload to a temporary file, tries the
known endpoints in order, deletes the file in `finally`, and returns `{text, model}`.
Without a key it returns 503/502 rather than silently returning an empty transcript.

A spoken memory is compelling on stage, but the demo script never depends on it.

---

## 5. Error handling and retries

`MetaModelAPI._request`:

- connection timeout 10 s, read timeout 120 s (ASR 300 s);
- retries on 429 and 5xx, three attempts, exponential backoff 0.6/1.2/2.4 s;
- other 4xx → `MetaAPIError` with the first 300 bytes of the body;
- every attempt is appended to `api.calls` with path, status, milliseconds and attempt
  number, available for a debug panel or post-mortem;
- `segment` checks `available` before opening the streaming client, so a missing key
  raises a clean `MetaAPIError` rather than an illegal `Bearer ` header.

The pipeline catches provider failures at stage boundaries, never around the whole
episode: a failed image call must not lose the EEG or the graph.

---

## 6. Degradation matrix

| Condition | Scene graph | Keyframe | Depth | Masks | EEG |
| --- | --- | --- | --- | --- | --- |
| No key | recipe | procedural | oracle of procedural frame | exact recipe | synthetic |
| Key, no `--muse` | Meta | procedural | oracle | exact recipe | synthetic |
| Key + `--muse` | Meta | Meta | DA-V2 if installed, else heuristic | SAM if `use_sam` | synthetic |
| Muse image fails | Meta | procedural | oracle | exact recipe | synthetic |
| SAM fails | Meta | Meta | DA-V2/heuristic | boxes | synthetic |
| Recorded EEG supplied | any | any | any | any | file |

The About modal chips are generated from `manifest.upstream`, so this table is visible to
the audience rather than buried in logs.

---

## 7. Cost model

| Item | Calls per pair | Notes |
| --- | --- | --- |
| Muse Spark | 1 | Cached with the episode |
| Muse Image | **1** | Shared between focus/distracted |
| SAM 3.1 | ≤ 5 | One per largest object, only with Muse keyframe |
| Muse Voice | 0–1 | Optional |
| Image spend | **$0.01** | List price at build time |

Text/segmentation/ASR are metered separately; check the current console for exact rates.
A live pair is therefore one image call plus a handful of small requests. The expensive
failure mode is accidentally calling Muse Image once per timeline frame — the code
prevents that by construction.

---

## 8. Security and data handling

- The key is read from the environment; it is never written to the manifest, logs or
  client bundle.
- CORS is `*` for the local demo. Lock this down before any public deployment.
- The server binds `127.0.0.1` by default. Quest testing over USB uses `adb reverse`;
  Wi-Fi testing needs HTTPS for WebXR.
- Transcripts are sent to the model provider only when `use_meta` is true and a key is
  present. The offline path sends nothing.
- The prompt forbids identifiable people, private locations, medical/diagnostic content
  and text. This is a product guardrail, not a compliance programme.
- Generated jobs live in memory and under `out/demo/live/`; there is no auth or cleanup
  service. This is a hackathon server.

---

## 9. Verification checklist (run before a live demo)

```bash
export MODEL_API_KEY=...
mnemosplat doctor --probe

# 1. Structured scene graph
python - <<'PY'
from mns.director.meta_client import MetaModelAPI
from mns.director.prompts import SYSTEM, USER_TEMPLATE, scene_graph_schema
api = MetaModelAPI()
raw, model = api.chat_json(SYSTEM, USER_TEMPLATE.format(
    narrative="A desk by a window with a warm mug.",
    viewpoint="standing eye height", mood="quiet", width=512, height=512, aspect=1.0,
), scene_graph_schema())
print(model, len(raw.get("objects", [])), len(raw.get("beats", [])))
PY

# 2. One image
python - <<'PY'
from mns.director.meta_client import MetaModelAPI
api = MetaModelAPI()
blobs = api.generate_image("A quiet study with an oak desk and a white mug, 35mm photo")
print(len(blobs), len(blobs[0]))
PY

# 3. SAM grammar (inspect the raw stream if parsing returns nothing)
#    The code logs retries; set MNS_LOG=DEBUG.

# 4. Full live pair (one image, both conditions)
mnemosplat demo --scenario both --out out/live --muse --force

# 5. Confirm provenance and the shared inputs
python - <<'PY'
import json
for k in ("focus", "distracted"):
    m = json.load(open(f"out/live/{k}/manifest.json"))
    print(k, m["upstream"])
PY
```

Acceptance:

- `upstream.scene_graph` starts with `meta:` for the first condition;
- `upstream.keyframe == "muse-image-1.0"`;
- the second condition says `shared between conditions` and `shared` for depth;
- both keyframe PNGs are byte-identical;
- `doctor --probe` lists the four model IDs;
- no secrets appear in `manifest.json` or `web/dist`.

If any step fails, leave the offline episode in place. A demo that tells the truth about
a fallback is stronger than one that stalls on stage.

---

## 10. Known integration risks

| Risk | Mitigation |
| --- | --- |
| Model IDs differ from the current console | `MNS_*_MODEL` env overrides; `doctor --probe` |
| `json_schema` unsupported | One retry with `json_object`; offline recipe if still bad |
| Image safety filter rejects a memory | Procedural fallback + warning chip |
| SAM mask encoding changes | Length-validated decode; boxes still work |
| Responses API event names change | Parser ignores unknown events; empty output → boxes |
| ASR endpoint path differs | Tries `/asr/transcribe` then `/audio/transcriptions` |
| Rate limits during demo | Retries with backoff; episodes cache; pre-generate before stage |
| Network dies mid-demo | Offline pair already on disk; `--offline` always works |

The last row is the important one: **pre-generate the demo pair**. The stage should
never depend on a live round trip.
