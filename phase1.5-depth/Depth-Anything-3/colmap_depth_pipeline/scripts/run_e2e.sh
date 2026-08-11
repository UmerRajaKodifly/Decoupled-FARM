#!/usr/bin/env bash
# End-to-end: equirect video OR frames dir → COLMAP SfM → DA3METRIC depth + PLY
#
# Usage:
#   ./colmap_depth_pipeline/scripts/run_e2e.sh <video.mp4|frames_dir> <output_dir> [options]
#
# Examples:
#   ./colmap_depth_pipeline/scripts/run_e2e.sh /data/clip.mp4 /data/runs/clip01
#   ./colmap_depth_pipeline/scripts/run_e2e.sh /data/frames /data/runs/clip01 --max-frames 100
#   ./colmap_depth_pipeline/scripts/run_e2e.sh clip.mp4 out --fps 2 --window-size 4 --skip-sfm
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PIPELINE_ROOT}/.." && pwd)"

usage() {
  cat <<EOF
Usage: $(basename "$0") <video.mp4|frames_dir> <output_dir> [options]

Required:
  input          Equirect video file OR directory of equirect frames
  output_dir     Root output directory (creates frames/, sfm/, depth/)

Options:
  --fps N              Extract FPS for video (default: config / 2)
  --max-frames N       Cap extracted / used frames
  --frames-dir DIR     Override frames location (default: <output>/frames)
  --overwrite-frames   Re-extract frames even if present
  --window-size N      DA3 window size in frames (default: 4)
  --overlap N          DA3 window overlap (default: 1)
  --config PATH        Pipeline YAML (default: configs/default.yaml)
  --conda-env NAME     Activate conda env before depth (default: da3)
  --no-conda           Do not activate conda (use current python)
  --skip-sfm           Skip COLMAP; reuse <output>/sfm
  --skip-depth         Skip DA3METRIC depth (SfM only)
  --no-ply             Do not export pointcloud.ply
  --device DEV         Torch device (default: auto)
  -h, --help           Show this help

Outputs:
  <output>/frames/                 equirect frames (from video or copied layout)
  <output>/sfm/                    COLMAP panorama project (sparse/, images/, ...)
  <output>/depth/                  face depths + manifest
  <output>/depth/pointcloud.ply   COLMAP K + poses
EOF
}

if [[ $# -ge 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

INPUT="$1"
OUTPUT="$2"
shift 2

FPS=""
MAX_FRAMES=""
FRAMES_DIR=""
OVERWRITE_FRAMES=0
WINDOW_SIZE=4
OVERLAP=1
CONFIG=""
CONDA_ENV="da3"
USE_CONDA=1
SKIP_SFM=0
SKIP_DEPTH=0
EXPORT_PLY=1
DEVICE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fps) FPS="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --frames-dir) FRAMES_DIR="$2"; shift 2 ;;
    --overwrite-frames) OVERWRITE_FRAMES=1; shift ;;
    --window-size) WINDOW_SIZE="$2"; shift 2 ;;
    --overlap) OVERLAP="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --no-conda) USE_CONDA=0; shift ;;
    --skip-sfm) SKIP_SFM=1; shift ;;
    --skip-depth) SKIP_DEPTH=1; shift ;;
    --no-ply) EXPORT_PLY=0; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

INPUT="$(realpath -m "${INPUT}")"
OUTPUT="$(realpath -m "${OUTPUT}")"
mkdir -p "${OUTPUT}"

SFM_DIR="${OUTPUT}/sfm"
DEPTH_DIR="${OUTPUT}/depth"
if [[ -z "${FRAMES_DIR}" ]]; then
  FRAMES_DIR="${OUTPUT}/frames"
else
  FRAMES_DIR="$(realpath -m "${FRAMES_DIR}")"
fi

if [[ -f "${INPUT}" ]]; then
  INPUT_KIND="video"
elif [[ -d "${INPUT}" ]]; then
  INPUT_KIND="frames"
else
  echo "ERROR: input not found (file or directory): ${INPUT}" >&2
  exit 1
fi

# --- python / conda ---
activate_conda() {
  if [[ "${USE_CONDA}" -eq 0 ]]; then
    return 0
  fi
  if ! command -v conda >/dev/null 2>&1; then
    # common install locations
    for cand in \
      "${HOME}/miniconda/etc/profile.d/conda.sh" \
      "${HOME}/miniconda3/etc/profile.d/conda.sh" \
      "${HOME}/anaconda3/etc/profile.d/conda.sh" \
      "/opt/conda/etc/profile.d/conda.sh"; do
      if [[ -f "${cand}" ]]; then
        # shellcheck source=/dev/null
        source "${cand}"
        break
      fi
    done
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "WARNING: conda not found; using current python ($(command -v python3 || true))" >&2
    return 0
  fi
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
  echo "Using conda env: ${CONDA_ENV} ($(command -v python))"
}

activate_conda
PYTHON="$(command -v python3 || command -v python)"
if [[ -z "${PYTHON}" ]]; then
  echo "ERROR: python not found" >&2
  exit 1
fi

SFM_PY="${SCRIPT_DIR}/run_panorama_sfm.py"
DEPTH_PY="${SCRIPT_DIR}/run_pipeline.py"

common_src_args=()
if [[ "${INPUT_KIND}" == "video" ]]; then
  common_src_args+=(--video "${INPUT}" --frames_dir "${FRAMES_DIR}")
  if [[ -n "${FPS}" ]]; then
    common_src_args+=(--fps "${FPS}")
  fi
  if [[ -n "${MAX_FRAMES}" ]]; then
    common_src_args+=(--max_frames "${MAX_FRAMES}")
  fi
  if [[ "${OVERWRITE_FRAMES}" -eq 1 ]]; then
    common_src_args+=(--overwrite_frames)
  fi
else
  # Existing frames: use as pano_dir. Optionally mirror into OUTPUT/frames via symlink
  # when user kept default frames dir empty of content.
  if [[ "$(realpath -m "${INPUT}")" != "$(realpath -m "${FRAMES_DIR}")" ]]; then
    mkdir -p "${FRAMES_DIR}"
    # Prefer symlink of directory contents reference: point pipeline at INPUT directly.
    :
  fi
  common_src_args+=(--pano_dir "${INPUT}")
fi

if [[ -n "${CONFIG}" ]]; then
  common_src_args+=(--config "${CONFIG}")
fi

echo "============================================================"
echo "E2E pipeline"
echo "  input:   ${INPUT} (${INPUT_KIND})"
echo "  output:  ${OUTPUT}"
echo "  frames:  ${FRAMES_DIR}"
echo "  sfm:     ${SFM_DIR}"
echo "  depth:   ${DEPTH_DIR}"
echo "============================================================"

# --- 1) SfM ---
if [[ "${SKIP_SFM}" -eq 0 ]]; then
  echo "[1/2] Running panorama SfM ..."
  mkdir -p "${SFM_DIR}"
  "${PYTHON}" "${SFM_PY}" \
    "${common_src_args[@]}" \
    --out_dir "${SFM_DIR}"
else
  echo "[1/2] Skipping SfM (--skip-sfm)"
  if [[ ! -d "${SFM_DIR}/sparse" ]]; then
    echo "ERROR: --skip-sfm but missing ${SFM_DIR}/sparse" >&2
    exit 1
  fi
fi

# After video extract, frames live in FRAMES_DIR; depth step should use that dir
# so we don't re-extract with a different default path.
depth_src_args=()
if [[ "${INPUT_KIND}" == "video" ]]; then
  has_frames=0
  if [[ -d "${FRAMES_DIR}" ]]; then
    if find "${FRAMES_DIR}" -maxdepth 1 -type f \
      \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
      -print -quit | grep -q .; then
      has_frames=1
    fi
  fi
  if [[ "${has_frames}" -eq 1 ]]; then
    depth_src_args+=(--pano_dir "${FRAMES_DIR}")
  else
    depth_src_args+=(--video "${INPUT}" --frames_dir "${FRAMES_DIR}")
    if [[ -n "${FPS}" ]]; then
      depth_src_args+=(--fps "${FPS}")
    fi
    if [[ -n "${MAX_FRAMES}" ]]; then
      depth_src_args+=(--max_frames "${MAX_FRAMES}")
    fi
  fi
else
  depth_src_args+=(--pano_dir "${INPUT}")
fi
if [[ -n "${CONFIG}" ]]; then
  depth_src_args+=(--config "${CONFIG}")
fi

# --- 2) DA3METRIC depth ---
if [[ "${SKIP_DEPTH}" -eq 0 ]]; then
  echo "[2/2] Running DA3METRIC dense depth ..."
  mkdir -p "${DEPTH_DIR}"
  depth_cmd=(
    "${PYTHON}" "${DEPTH_PY}"
    --colmap_dir "${SFM_DIR}"
    "${depth_src_args[@]}"
    --out_dir "${DEPTH_DIR}"
    --window_size "${WINDOW_SIZE}"
    --overlap "${OVERLAP}"
  )
  if [[ "${EXPORT_PLY}" -eq 1 ]]; then
    depth_cmd+=(--export_ply)
  fi
  if [[ -n "${DEVICE}" ]]; then
    depth_cmd+=(--device "${DEVICE}")
  fi
  "${depth_cmd[@]}"
else
  echo "[2/2] Skipping depth (--skip-depth)"
fi

echo "============================================================"
echo "Done."
echo "  SfM:    ${SFM_DIR}"
echo "  Depth:  ${DEPTH_DIR}"
if [[ -f "${DEPTH_DIR}/pointcloud.ply" ]]; then
  echo "  PLY:    ${DEPTH_DIR}/pointcloud.ply"
elif [[ -f "${DEPTH_DIR}/pointcloud_faces.ply" ]]; then
  echo "  PLY:    ${DEPTH_DIR}/pointcloud_faces.ply"
fi
if [[ -f "${DEPTH_DIR}/manifest.json" ]]; then
  echo "  Manifest: ${DEPTH_DIR}/manifest.json"
fi
echo "============================================================"
