#!/usr/bin/env bash
# Phase 1 entrypoint — Stella VSLAM Dense (headless)
set -euo pipefail

STAGE="phase1-slam"
RUN_ID="${PIPELINE_RUN_ID:-run_local}"
LOG_DIR="${PIPELINE_LOG_DIR:-/outputs/logs}/${RUN_ID}"
mkdir -p "${LOG_DIR}"
STAGE_LOG="${LOG_DIR}/${STAGE}.log"
MERGED_LOG="${LOG_DIR}/pipeline.log"

ts() { date -Iseconds; }
log() {
  local msg="[$(ts)] [${STAGE}] [INFO] $*"
  echo "$msg" | tee -a "${STAGE_LOG}" | tee -a "${MERGED_LOG}" >/dev/null
  # also print to console (tee to both files already swallowed stdout of echo via redirect)
  echo "$msg"
}

VIDEO_FILE="${VIDEO_FILE:-}"
if [[ -z "${VIDEO_FILE}" ]]; then
  # Auto-pick first mp4/MP4 under /inputs
  VIDEO_FILE="$(find /inputs -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mov' \) 2>/dev/null | sort | head -n1 || true)"
fi
# Bare filename → /inputs/<name>  (must happen before -f check)
if [[ -n "${VIDEO_FILE}" && "${VIDEO_FILE}" != /* ]]; then
  VIDEO_FILE="/inputs/${VIDEO_FILE}"
fi
if [[ -z "${VIDEO_FILE}" || ! -f "${VIDEO_FILE}" ]]; then
  log "ERROR: no video found. Set VIDEO_FILE or place an .mp4 under /inputs"
  log "  ls /inputs: $(ls -la /inputs 2>&1 || true)"
  log "  VIDEO_FILE was: ${VIDEO_FILE:-<empty>}"
  exit 1
fi
RESIZE="${RESIZE:-1920x960}"
FRAME_STEP="${FRAME_STEP:-2}"
VOCAB="${VOCAB_PATH:-/data/orb_vocab.fbow}"
CONFIG="${SLAM_CONFIG:-/config/slam_config.yaml}"
OUT_DIR="${OUT_DIR:-/outputs}"

mkdir -p "${OUT_DIR}/keyframes" "${OUT_DIR}/traj"

log "================================================"
log "START Stella VSLAM Dense"
log "  video=${VIDEO_FILE}"
log "  resize=${RESIZE} frame_step=${FRAME_STEP}"
log "  config=${CONFIG}"
log "  vocab=${VOCAB}"
log "  out=${OUT_DIR}"
log "  run_id=${RUN_ID}"
log "================================================"

if [[ ! -f "${VOCAB}" ]]; then
  log "ERROR: ORB vocab missing at ${VOCAB} — run bootstrap_models.sh"
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  log "ERROR: slam config missing at ${CONFIG}"
  exit 1
fi

T0=$(date +%s)

# PYTHONPATH may include /usr/local/lib/python* site packages; stella pybind when installed there
export PYTHONPATH="${PYTHONPATH:-}:/usr/local/lib"

set +e
python3 -u /stella/tools/run_video_slam.py \
  -v "${VOCAB}" \
  -c "${CONFIG}" \
  -m "${VIDEO_FILE}" \
  --resize "${RESIZE}" \
  --frame-step "${FRAME_STEP}" \
  -o "${OUT_DIR}/out.db" \
  -p "${OUT_DIR}/out.ply" \
  -k "${OUT_DIR}/keyframes/" \
  --eval-log-dir "${OUT_DIR}/traj" \
  --auto-term \
  --disable-viewer \
  2>&1 | tee -a "${STAGE_LOG}" | tee -a "${MERGED_LOG}"
EXIT_CODE=${PIPESTATUS[0]}
set -e

T1=$(date +%s)
ELAPSED=$((T1 - T0))
log "END exit=${EXIT_CODE} elapsed_s=${ELAPSED}"
echo "${ELAPSED}" > "${LOG_DIR}/${STAGE}.elapsed"
exit ${EXIT_CODE}
