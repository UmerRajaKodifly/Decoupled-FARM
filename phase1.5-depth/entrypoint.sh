#!/usr/bin/env bash
# Phase 1.5 entrypoint — DA3 metric depth from Stella keyframes
set -euo pipefail

STAGE="phase1.5-depth"
RUN_ID="${PIPELINE_RUN_ID:-run_local}"
LOG_DIR="${PIPELINE_LOG_DIR:-/outputs/logs}/${RUN_ID}"
mkdir -p "${LOG_DIR}"
STAGE_LOG="${LOG_DIR}/${STAGE}.log"
MERGED_LOG="${LOG_DIR}/pipeline.log"

ts() { date -Iseconds; }
log() {
  local msg="[$(ts)] [${STAGE}] [INFO] $*"
  echo "$msg" | tee -a "${STAGE_LOG}" | tee -a "${MERGED_LOG}"
}

PHASE1="${PHASE1_DIR:-/phase1}"
OUT_DIR="${OUT_DIR:-/phase1.5}"
WINDOW_SIZE="${WINDOW_SIZE:-4}"
MAX_KFS="${MAX_KFS:-}"
MODEL="${DA3_MODEL:-depth-anything/DA3METRIC-LARGE}"

# Prefer local cached weights if bootstrap put them under /models
if [[ -d /models/da3metric-large ]]; then
  MODEL="/models/da3metric-large"
fi

DB="${PHASE1}/out.db"
KF_DIR="${PHASE1}/keyframes"
KF_TRAJ="${PHASE1}/traj/keyframe_trajectory.txt"

log "================================================"
log "START DA3 metric depth"
log "  db=${DB}"
log "  kf_dir=${KF_DIR}"
log "  traj=${KF_TRAJ}"
log "  out=${OUT_DIR}"
log "  model=${MODEL}"
log "  window=${WINDOW_SIZE} max_kfs=${MAX_KFS:-all}"
log "  run_id=${RUN_ID}"
log "================================================"

for f in "$DB" "$KF_DIR" "$KF_TRAJ"; do
  if [[ ! -e "$f" ]]; then
    log "ERROR: missing Stella artifact: $f (Phase 1 must succeed first)"
    exit 1
  fi
done

mkdir -p "${OUT_DIR}"
export PYTHONPATH="/app/Depth-Anything-3/src:/app/Depth-Anything-3/colmap_depth_pipeline/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/models/hf_cache}"
mkdir -p "${HF_HOME}"

EXTRA_ARGS=()
if [[ -n "${MAX_KFS}" ]]; then
  EXTRA_ARGS+=(--max-kfs "${MAX_KFS}")
fi

T0=$(date +%s)
set +e
python3 -u /app/stella_to_da3_depth.py \
  --db "${DB}" \
  --kf-image-dir "${KF_DIR}" \
  --kf-traj "${KF_TRAJ}" \
  --out-dir "${OUT_DIR}" \
  --window-size "${WINDOW_SIZE}" \
  --model "${MODEL}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "${STAGE_LOG}" | tee -a "${MERGED_LOG}"
EXIT_CODE=${PIPESTATUS[0]}
set -e

T1=$(date +%s)
ELAPSED=$((T1 - T0))
log "END exit=${EXIT_CODE} elapsed_s=${ELAPSED}"
echo "${ELAPSED}" > "${LOG_DIR}/${STAGE}.elapsed"
exit ${EXIT_CODE}
