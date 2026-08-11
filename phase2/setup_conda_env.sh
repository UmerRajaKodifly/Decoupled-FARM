#!/usr/bin/env bash
# Create a host conda env for Phase 2 (Detect-Segment-Embed) without the FARM Docker image.
#
# Env name: farm-phase2
# What it includes: Python 3.10 + CUDA PyTorch + FARM scene_graph + YOLOE + MobileCLIP + DINOv3 deps
# What it SKIPS (not needed for Phase 2): ROS 2, vLLM, caption/embedding servers
#
# Usage:
#   bash /home/kodifly/Desktop/farm-git/pipeline/phase2-detect-segment-embed/setup_conda_env.sh
#   conda activate farm-phase2
#   python phase2_runner.py --frames-json ... --data-root ... --device cuda
#
set -euo pipefail

ENV_NAME="${ENV_NAME:-farm-phase2}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
# Driver 580 + system CUDA 12.6: cu124 wheels are the widely available / stable choice.
# Override if you want cu128: TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu128
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FARM_ROOT="$(cd -- "${SCRIPT_DIR}/../../FARM-Project" && pwd)"
YOLOE_DIR="${FARM_ROOT}/third_party/yoloe"
MOBILECLIP_DIR="${YOLOE_DIR}/third_party/ml-mobileclip"

die() { echo "ERROR: $*" >&2; exit 1; }

command -v conda >/dev/null 2>&1 || die "conda not found. Install Miniconda/Anaconda first."
[[ -d "${FARM_ROOT}/src/scene_graph" ]] || die "FARM src not found at ${FARM_ROOT}/src/scene_graph"
[[ -d "${YOLOE_DIR}/ultralytics" ]] || die "YOLOE submodule missing. Run: cd ${FARM_ROOT} && git submodule update --init --recursive"
[[ -d "${MOBILECLIP_DIR}/mobileclip" ]] || die "MobileCLIP not found under third_party/yoloe/third_party/ml-mobileclip"
[[ -f "${FARM_ROOT}/models/yoloe/yoloe-v8l-seg.pt" ]] || echo "WARN: YOLOE weights missing — run ${FARM_ROOT}/bootstrap_models.sh"
[[ -f "${FARM_ROOT}/models/mobileclip/mobileclip_blt.pt" ]] || echo "WARN: MobileCLIP weights missing — run ${FARM_ROOT}/bootstrap_models.sh"
[[ -d "${FARM_ROOT}/models/dinov3-vits16" ]] || die "DINOv3 weights missing at ${FARM_ROOT}/models/dinov3-vits16"

# Ensure conda is hookable from this shell
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"

echo "==> Creating / updating conda env: ${ENV_NAME} (Python ${PYTHON_VERSION})"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "    env exists — will update packages in-place"
else
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

conda activate "${ENV_NAME}"
python -m pip install -U pip setuptools wheel

# System libs OpenCV sometimes needs (best-effort via conda)
conda install -y -c conda-forge libgl libglib ffmpeg || true

echo "==> Installing CUDA-aware PyTorch from ${TORCH_CUDA_INDEX}"
python -m pip install \
  --index-url "${TORCH_CUDA_INDEX}" \
  torch torchvision torchaudio

echo "==> Installing Phase-2 Python deps"
python -m pip install \
  "transformers>=4.57.0" \
  huggingface_hub safetensors accelerate \
  opencv-python-headless \
  h5py loguru "networkx>=2.8" supervision tqdm timm \
  scikit-learn scipy transforms3d "imageio[ffmpeg]" matplotlib \
  pydantic Pillow PyYAML requests pandas seaborn psutil py-cpuinfo \
  open-clip-torch ultralytics-thop

echo "==> Installing YOLOE (editable) + MobileCLIP (editable)"
# Prefer "setuptools<70" for older package metadata on some YOLOE-adjacent packages.
python -m pip install "setuptools<70" || true
python -m pip install -e "${YOLOE_DIR}"
python -m pip install -e "${MOBILECLIP_DIR}"

echo "==> Installing farm-scene-graph (editable, no vLLM extras)"
python -m pip install -e "${FARM_ROOT}" --no-deps
# Explicitly ensure FARM runtime deps (no torch re-pull from PyPI CPU)
python -m pip install \
  "transformers>=4.57.0" ultralytics opencv-python-headless \
  h5py huggingface_hub loguru numpy Pillow PyYAML requests \
  "networkx>=2.8" supervision tqdm timm scikit-learn scipy \
  transforms3d matplotlib "imageio[ffmpeg]" viser

echo "==> Smoke checks"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
from ultralytics import YOLOE
print("YOLOE import OK:", YOLOE)
import mobileclip  # noqa: F401
print("mobileclip import OK")
import scene_graph  # noqa: F401
print("scene_graph import OK")
from scene_graph.segmentation.yoloe import YOLOESegmenter  # noqa: F401
from scene_graph.segmentation.dino import DINOFeaturesExtractor  # noqa: F401
print("segmenter + dino classes OK")
PY

cat <<EOF

============================================================
Env ready: ${ENV_NAME}

Activate:
  conda activate ${ENV_NAME}

Run Phase 2:
  cd ${SCRIPT_DIR}
  python phase2_runner.py \\
    --frames-json /home/kodifly/Desktop/depth-reconstr/da3_scan_depth/frames_json/frames.json \\
    --data-root   /home/kodifly/Desktop/depth-reconstr/da3_scan_depth \\
    --output-dir  ./output \\
    --device cuda --conf 0.35

Optional smoke (1/5 keyframes):
  python phase2_runner.py ... --stride 5
============================================================
EOF
