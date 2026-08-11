#!/usr/bin/env bash
# Create / refresh the Depth Anything 3 conda environment.
set -euo pipefail

ENV_NAME="${DA3_ENV_NAME:-da3}"
PYTHON_VERSION="${DA3_PYTHON:-3.10}"
# Match common CUDA 12.x wheels; driver 595 + toolkit 12.8 work with cu121/cu124.
TORCH_CUDA="${DA3_TORCH_CUDA:-cu121}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> Creating conda env '${ENV_NAME}' (python=${PYTHON_VERSION})"
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip ffmpeg

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "==> Installing PyTorch + xformers (${TORCH_CUDA})"
# Avoid mixing packages from ~/.local into this env.
export PYTHONNOUSERSITE=1
pip install --upgrade pip
pip install "numpy<2"
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
pip install xformers --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

echo "==> Installing Depth Anything 3 (editable) + pipeline deps"
cd "${REPO_ROOT}"
pip install -e .
pip install -r colmap_depth_pipeline/requirements.txt
pip install "numpy<2" tqdm pyyaml scipy

echo "==> Smoke check"
python - <<'PY'
import torch
import depth_anything_3
print("python ok")
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
print("depth_anything_3", depth_anything_3.__file__)
PY

echo
echo "Done. Activate with:  conda activate ${ENV_NAME}"
