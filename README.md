# MnemoSplat

**Same words. Different memory.** EEG-salience-conditioned 3D Gaussian memory volumes
for WebXR.

The project lives in [`mnemosplat/`](mnemosplat/). Start there:

- [Project README](mnemosplat/README.md) — architecture, quickstart, demo script,
  honest capability status.
- [Deploy on Render](mnemosplat/docs/DEPLOY_RENDER.md) — Docker image, blueprint,
  environment variables, verification, troubleshooting.
- [Architecture](mnemosplat/docs/ARCHITECTURE.md),
  [science notes](mnemosplat/docs/SCIENCE.md),
  [Meta integration](mnemosplat/docs/META_INTEGRATION.md).

## Render

This repository root includes [`render.yaml`](render.yaml) for a monorepo deploy. It
builds `mnemosplat/Dockerfile` with `mnemosplat/` as the Docker context. The image
builds the Vite client and bakes the offline focus/distracted assets; no GPU or API key
is required at runtime. The baked demo is public; paid generation is blocked, while
recorded-EEG uploads remain available and write only temporary assets.

If you push `mnemosplat/` itself as the repository root instead, use
[`mnemosplat/render.yaml`](mnemosplat/render.yaml).

The baked comparison conditions use synthetic EEG, labelled as such. Uploaded
recordings are labelled recorded EEG. This is a systems prototype, not a diagnostic or
a clinical claim.
