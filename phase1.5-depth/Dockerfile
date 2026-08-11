# Phase 1.5 — Stella keyframes → DA3METRIC face depths → frames.json
#
# IMPORTANT: torch + torchvision must be installed together from the same
# pytorch cu124 channel. Later deps (open3d/evo/etc.) can pull a mismatched
# torchvision from PyPI and break with:
#   RuntimeError: operator torchvision::nms does not exist
# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        libgl1 libglib2.0-0 ffmpeg curl ca-certificates \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && pip3 install --no-cache-dir -U pip setuptools wheel

# Non-torch deps first (may try to pull a CPU/wrong torchvision — we fix below)
RUN pip3 install --no-cache-dir \
    "numpy<2" \
    "opencv-python-headless" \
    scipy \
    scikit-learn \
    tqdm \
    pyyaml \
    einops \
    addict \
    matplotlib \
    huggingface_hub \
    imageio \
    imageio-ffmpeg \
    pillow \
    pillow_heif \
    omegaconf \
    safetensors \
    typer \
    requests \
    trimesh \
    plyfile \
    e3nn \
    "moviepy==1.0.3" \
    decorator \
    proglog \
    fastapi \
    uvicorn \
    pydantic \
    evo \
    pycolmap \
 && (pip3 install --no-cache-dir open3d || echo "WARN: open3d unavailable — continuing")

# Matching CUDA 12.4 torch stack LAST (wins over any PyPI torchvision pulled above).
# Do NOT install torchaudio if unused; keep torchvision from the same index.
RUN pip3 install --no-cache-dir --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch torchvision \
 && (pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 xformers \
     || echo "WARN: xformers unavailable — continuing")

# Sanity: fail the build if torchvision/torch are mismatched
RUN python3 -c "import torch, torchvision; print('torch', torch.__version__); print('tv', torchvision.__version__); import torchvision.transforms as T; print('torchvision OK')"

WORKDIR /app
COPY Depth-Anything-3 /app/Depth-Anything-3
COPY stella_to_da3_depth.py /app/stella_to_da3_depth.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONPATH=/app/Depth-Anything-3/src:/app/Depth-Anything-3/colmap_depth_pipeline/src:${PYTHONPATH}

ENTRYPOINT ["/app/entrypoint.sh"]
