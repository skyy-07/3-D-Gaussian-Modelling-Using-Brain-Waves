# CUDA worker for the experimental BrainVis branch. Build on a GPU host; the Render
# CPU web image is a separate Dockerfile and must not include this.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BRAINVIS_ROOT=/opt/BrainVis

RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3.10 python3-pip python3.10-venv libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install -r requirements.txt

# Pin the public checkout; the authors' checkpoint is mounted separately.
RUN git clone https://github.com/RomGai/BrainVis.git /opt/BrainVis && \
    cd /opt/BrainVis && git checkout 129b8c23d69b56998f08e3c463e24facd7010f1c

COPY app.py engine.py ./
EXPOSE 8000
CMD ["python3.10", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
