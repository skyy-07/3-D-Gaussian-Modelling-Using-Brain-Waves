# MnemoSplat

**Same words. Different memory.** EEG-salience-conditioned 3D Gaussian memory volumes
for WebXR.

**Live demo:** <https://mnemosplat.onrender.com>

**Source and full documentation:** [`mnemosplat/`](mnemosplat/)

> GitHub hosts the source code, not the running FastAPI/Docker service. This project is
> not deployed with GitHub Pages because it needs a Python server, generated assets and
> a WebXR client on the same origin. The public instance runs on Render; the free plan
> spins down when idle, so the first load can take 30–60 seconds.

The baked comparison conditions use synthetic EEG, labelled as such. Uploaded
recordings are labelled recorded EEG. This is a systems prototype, not a diagnostic or
a clinical claim.

## Run it locally

```bash
cd mnemosplat
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
mnemosplat demo --scenario both --out out/demo --splats 120000 --force
cd web && npm install && npm run build && cd ..
mnemosplat serve --host 0.0.0.0 --port 8000
```

Open <http://127.0.0.1:8000>. The generated `out/` directory and `web/dist/` are
intentionally ignored by Git; the Docker image creates them at build time.
