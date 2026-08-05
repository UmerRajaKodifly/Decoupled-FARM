#!/usr/bin/env bash
# Source from anywhere:
#   source /path/to/Decoupled-FARM/env.sh
#
# Puts an optional COLMAP 4.1+Caspar install on PATH and activates conda env
# farm-map when available. Does not assume a sibling farm-rnd checkout.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FARM_OBJECT_MAP_ROOT="${ROOT}"

if [[ -z "${COLMAP_ROOT:-}" && -x "${HOME}/tools/colmap-4.1.0/bin/colmap" ]]; then
  export COLMAP_ROOT="${HOME}/tools/colmap-4.1.0"
fi
if [[ -n "${COLMAP_ROOT:-}" ]]; then
  export PATH="${COLMAP_ROOT}/bin:${PATH}"
  export LD_LIBRARY_PATH="${COLMAP_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

_activate_farm_map() {
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate farm-map
    return 0
  fi
  for conda_sh in \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "${HOME}/mambaforge/etc/profile.d/conda.sh"
  do
    if [[ -f "${conda_sh}" ]]; then
      # shellcheck disable=SC1090
      source "${conda_sh}"
      conda activate farm-map
      return 0
    fi
  done
  return 1
}

if [[ "${CONDA_DEFAULT_ENV:-}" != "farm-map" ]]; then
  _activate_farm_map || true
fi

export PYTHONNOUSERSITE=1
export FARM_PROJECT_ROOT="${FARM_PROJECT_ROOT:-${ROOT}/third_party/FARM-Project}"
export SS3DGS_ROOT="${SS3DGS_ROOT:-${ROOT}/third_party/ss-3dgs}"
export SCENE_GRAPH_MODEL_DIR="${SCENE_GRAPH_MODEL_DIR:-${FARM_PROJECT_ROOT}/models}"
export PYTHONPATH="${ROOT}/src:${FARM_PROJECT_ROOT}/src:${SS3DGS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "colmap: $(command -v colmap || echo missing)"
colmap version 2>/dev/null | head -n 1 || true
echo "python: $(command -v python || echo missing)"
python -c "import sys; print(sys.version.split()[0], sys.prefix)" 2>/dev/null || true
echo "FARM_PROJECT_ROOT=${FARM_PROJECT_ROOT}"
echo "SS3DGS_ROOT=${SS3DGS_ROOT}"
