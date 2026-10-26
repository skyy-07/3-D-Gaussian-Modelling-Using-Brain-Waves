# Deploying MnemoSplat on Render

MnemoSplat runs as one Docker web service: FastAPI serves the API, the built Vite client,
and the generated `.splat` assets on the same HTTPS origin. The image builds the client
and generates the offline focus/distracted demo **at image-build time**, so the running
service needs no GPU, no API key, and no persistent disk.

Use Render's **Docker** runtime. Do not switch this service to the native Python runtime:
`out/` is intentionally not committed, and the baked assets are produced by the image
build.

> The demo is synthetic EEG, not a clinical tool. See [SCIENCE.md](SCIENCE.md).

---

## 1. Put the code in a Git repository

Render deploys from GitHub/GitLab/Bitbucket. Choose one layout:

**Monorepo (repository root contains `mnemosplat/`)** — keep the parent
`render.yaml` and push the parent folder. It points at
`mnemosplat/Dockerfile` with `dockerContext: ./mnemosplat`.

**Project-only (repository root is `mnemosplat/`)** — use
`mnemosplat/render.yaml` and delete/ignore the parent file. It points at
`./Dockerfile` with context `.`.

Only the `render.yaml` at the repository root is read; the other file is simply unused.
The Docker build itself only needs `Dockerfile`, `.dockerignore`, `pyproject.toml`,
`README.md`, `src/`, and `web/`.

---

## 2. Create the Render service

### Blueprint (recommended)

1. Push the repository to GitHub.
2. In Render, choose **New → Blueprint** and select the repository.
3. Render reads `render.yaml`, creates the `mnemosplat` Docker web service, and starts
   the first build.
4. When prompted, leave `MODEL_API_KEY` and `MNS_ADMIN_TOKEN` empty for the read-only
   offline demo.

### Manual web service

If you prefer the dashboard:

| Setting | Value |
| --- | --- |
| Type | Web Service |
| Runtime | Docker |
| Root directory (monorepo) | `mnemosplat` |
| Dockerfile path | `./Dockerfile` (relative to the root directory) |
| Docker context | `.` |
| Health check path | `/api/health` |
| Plan | Free is sufficient for the demo |

There is no build command or start command to enter; Render uses the Dockerfile `CMD`.
The image listens on `$PORT`, which Render injects.

---

## 3. Environment variables

The image already sets sensible defaults. The blueprint also sets them explicitly:

| Variable | Value | Purpose |
| --- | --- | --- |
| `MNS_PUBLIC_DEMO` | `1` | Read-only demo; block paid generation unless a valid admin token is supplied |
| `MNS_OUT` | `/app/out/demo` | Asset root served at `/assets` |
| `MNS_WEB_DIST` | `/app/web/dist` | Built client served at `/` |
| `MNS_LOG` | `INFO` | Log verbosity |
| `MODEL_API_KEY` | unset | Optional Meta Model API key |
| `MNS_ADMIN_TOKEN` | unset | Optional bearer-like token for write endpoints |

`PORT` is set by Render; do not override it. `MNS_OUT` and `MNS_WEB_DIST` matter because
`pip install .` puts `mns` in `site-packages`; the Dockerfile's absolute paths make the
asset lookup independent of the package location.

### Public demo guard

`MNS_PUBLIC_DEMO=1` makes these endpoints return **403** unless a valid
`X-MNS-Token` is supplied and `MNS_ADMIN_TOKEN` is configured:

- `POST /api/episode`
- `POST /api/selftest`
- `POST /api/segment`
- `POST /api/transcribe`

The read-only UI never calls them. With no admin token, `GET /api/health` reports
`"generation_enabled": false`. This prevents a public URL from spending your
`MODEL_API_KEY` or queuing CPU-heavy jobs.

To enable authenticated live generation, set **both** `MODEL_API_KEY` and a long random
`MNS_ADMIN_TOKEN`, then send the token in `X-MNS-Token`:

```bash
curl -X POST "https://YOUR-SERVICE.onrender.com/api/episode" \
  -H "X-MNS-Token: $MNS_ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"scenario":"focus","use_muse_image":true,"target_splats":120000}'
```

There is no rate limiter. Keep live generation disabled on a public hackathon URL unless
you intend to pay for it.

---

## 4. What the build does

```text
Node 20 stage
  npm ci ──► vite build ──► /web/dist

Python 3.11 stage
  pip install .                       install FastAPI + numpy/scipy/Pillow
  COPY /web/dist                      serve the client from FastAPI
  mnemosplat demo --offline ...       generate focus + distracted at build time
  chown app:app /app                  run as a non-root user
```

The asset generation is deterministic and offline:

```bash
mnemosplat demo --scenario both --out /app/out/demo \
  --splats 120000 --force --offline --no-preview
```

`--offline` guarantees the build never spends API credit even if a key happens to be in
the build environment. `--no-preview` skips QA turntables that the web client does not
use. The two conditions share one transcript, scene graph, keyframe, and depth map.

Typical first build: a few minutes on Render. Subsequent source-only changes rebuild
the relevant layers. The generated assets are roughly 50 MB and are baked into the
image, so cold starts only start uvicorn.

If the build runs out of memory on a small plan, lower the splat budget by editing the
default:

```dockerfile
ARG MNS_SPLATS=80000
```

or build locally with `--build-arg MNS_SPLATS=80000`. The pipeline's hard cap is
262,144 splats.

---

## 5. Verify the deployment

```bash
URL=https://YOUR-SERVICE.onrender.com

curl -fsS "$URL/api/health" | python -m json.tool
curl -fsSI "$URL/" | grep -i -E 'cross-origin|content-type'
curl -fsS "$URL/assets/index.json" | python -m json.tool | head -40
curl -sS -o /dev/null -w '%{http_code}\n' \
  -X POST "$URL/api/episode" -H 'Content-Type: application/json' -d '{}'
```

Expected:

- `/api/health` returns `200`, `"ok": true`, `"web_built": true`,
  `"public_demo": true`, `"generation_enabled": false`.
- `/` returns the client HTML with `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp`.
- `/assets/index.json` lists `focus` and `distracted`.
- The unauthenticated `POST /api/episode` returns `403`.

The client needs `crossOriginIsolated === true` for the splat sorter's
`SharedArrayBuffer`. The FastAPI middleware sets COOP/COEP on every response, including
static assets.

### WebXR on Quest

Render provides HTTPS, so the page is a secure context. Open the Render URL in the
Quest browser and press **Enter VR**. The manifest camera is mapped to the headset
origin. Trigger advances the epoch, squeeze goes back.

If the headset cannot reach the public URL, run locally over USB instead:

```bash
adb reverse tcp:8000 tcp:8000
# open http://127.0.0.1:8000 in the Quest browser
```

---

## 6. Free-plan behaviour

- The free web instance has 0.1 CPU / 512 MB. It is enough to serve the baked demo.
- Free services spin down after inactivity. The first request after a spin-down can take
  roughly 30–60 seconds while the container starts; the assets themselves are already in
  the image, so there is no generation step at runtime.
- The image build is separate from the runtime instance and may take longer than a
  request. Watch the deploy log if the health check initially fails.
- Render terminates TLS and forwards `PORT`; the Dockerfile binds `0.0.0.0:${PORT}`.

---

## 7. Troubleshooting

| Symptom | Check |
| --- | --- |
| `502` / health check failing | Deploy log; confirm the process binds `${PORT}`, not a hard-coded port |
| `/` is 404 but `/api/health` works | `MNS_WEB_DIST` must point at the built client (`/app/web/dist`); rebuild the image |
| `/assets/...` is 404 | `MNS_OUT` must point at `/app/out/demo`; confirm the demo layer ran |
| `SharedArrayBuffer` unavailable | Response must include COOP/COEP; do not strip them in a proxy or CDN |
| Build OOM or times out | Lower `MNS_SPLATS`, or move to a larger plan |
| Generation returns 403 | Expected with `MNS_PUBLIC_DEMO=1` and no token; see the guard section |
| Model calls fail | Unrelated to the offline demo; `MODEL_API_KEY` is optional |
| Slow first request | Free-plan cold start; subsequent requests are fast |

If you use Cloudflare or another proxy in front of Render, make sure it preserves the
COOP/COEP response headers and does not cache `/` or `/api/health` incorrectly.

---

## 8. Local Docker check (optional)

Docker is not required to deploy through Render, but if you have it:

```bash
cd mnemosplat
docker build -t mnemosplat .
docker run --rm -p 8000:8000 -e PORT=8000 mnemosplat
curl -fsS http://127.0.0.1:8000/api/health | python -m json.tool
```

Then open <http://127.0.0.1:8000>. To test the public guard with a key present:

```bash
docker run --rm -p 8000:8000 \
  -e PORT=8000 \
  -e MNS_PUBLIC_DEMO=1 \
  -e MODEL_API_KEY=dummy \
  mnemosplat
curl -sS -o /dev/null -w '%{http_code}\n' \
  -X POST http://127.0.0.1:8000/api/episode \
  -H 'Content-Type: application/json' -d '{}'   # 403
```

The local `out/demo` ignored by Git is not needed in the image; the build creates it.
