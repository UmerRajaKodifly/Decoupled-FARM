#!/usr/bin/env bash
# Source this before running the pipeline:
#   source /home/kodifly/Desktop/farm-rnd/farm-object-map/env.sh
#
# Puts COLMAP 4.1.0+Caspar ahead of the system COLMAP 3.7, and activates
# the dedicated conda env so we do not touch other project environments.

export COLMAP_ROOT="${COLMAP_ROOT:-/home/kodifly/tools/colmap-4.1.0}"
export PATH="${COLMAP_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${COLMAP_ROOT}/lib:${LD_LIBRARY_PATH:-}"

if [[ -f /home/kodifly/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /home/kodifly/miniconda3/etc/profile.d/conda.sh
  conda activate farm-map
fi
# Prefer the conda env binary over pyenv shims.
export PATH="/home/kodifly/miniconda3/envs/farm-map/bin:${PATH}"

export PYTHONNOUSERSITE=1
export SS3DGS_ROOT="${SS3DGS_ROOT:-/home/kodifly/Desktop/farm-rnd/ss-3dgs}"
export FARM_OBJECT_MAP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FARM_PROJECT_SRC="${FARM_PROJECT_SRC:-/home/kodifly/Desktop/farm-rnd/FARM-Project/src}"
export SCENE_GRAPH_MODEL_DIR="${SCENE_GRAPH_MODEL_DIR:-/home/kodifly/Desktop/farm-rnd/FARM-Project/models}"
export PYTHONPATH="${FARM_OBJECT_MAP_ROOT}/src:${FARM_PROJECT_SRC}:${SS3DGS_ROOT}:${PYTHONPATH:-}"

echo "colmap: $(command -v colmap)"
colmap version 2>/dev/null | head -n 1 || true
echo "python: $(command -v python)"
python -c "import sys; print(sys.version.split()[0], sys.prefix)" 2>/dev/null || true
