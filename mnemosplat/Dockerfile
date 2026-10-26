# syntax=docker/dockerfile:1
#
# MnemoSplat production image.
#
# Stage 1 builds the Vite client. Stage 2 installs the Python package and bakes the
# offline demo assets into the image, so a Render cold start serves static files with
# no model keys, network calls, or GPU. Regenerate assets with a new image build after
# changing the pipeline.
#
# Build:
#   docker build -t mnemosplat .
#   docker run --rm -p 8000:8000 -e PORT=8000 mnemosplat
#
# Lower the build-time splat budget for small instances:
#   docker build --build-arg MNS_SPLATS=80000 -t mnemosplat .

FROM node:20-bookworm-slim AS web

WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/index.html web/vite.config.js ./
COPY web/src ./src
RUN npm run build

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MNS_OUT=/app/out/demo \
    MNS_WEB_DIST=/app/web/dist \
    MNS_LOG=INFO \
    MNS_PUBLIC_DEMO=1

ARG MNS_SPLATS=120000

WORKDIR /app

# Copy only what the package build needs so source edits do not invalidate the npm layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

# Create the unprivileged runtime user before generating assets. Running the demo as
# `app` and copying the client with the same owner avoids a final recursive chown layer
# that would duplicate the ~50 MB asset tree in the image.
RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app --no-create-home app \
    && mkdir -p /app/out \
    && chown -R app:app /app/out
USER app

# Bake the paired demo (focus + distracted) into the image. --offline guarantees the
# build never spends API credit even if a key is present in the build environment.
# --no-preview skips QA turntables that the web client does not use.
RUN mnemosplat demo \
        --scenario both \
        --out /app/out/demo \
        --splats "${MNS_SPLATS}" \
        --force \
        --offline \
        --no-preview

# Copy the client after asset generation so a web-only change does not rebuild the
# demo. /app/out is writable by `app` for admin-only generation and public recorded-EEG
# re-lifts; Render treats this filesystem as ephemeral.
COPY --from=web --chown=app:app /web/dist ./web/dist

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/api/health', timeout=3)"

# Render injects PORT. Bind all interfaces so the platform proxy can reach uvicorn.
CMD ["sh", "-c", "exec mnemosplat serve --host 0.0.0.0 --port ${PORT:-8000}"]
