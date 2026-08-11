#!/usr/bin/env bash
# Farm service entrypoint — Phase 2 then Phase 3
set -euo pipefail

RUN_ID="${PIPELINE_RUN_ID:-run_local}"
LOG_DIR="${PIPELINE_LOG_DIR:-/outputs/logs}/${RUN_ID}"
mkdir -p "${LOG_DIR}"
export PIPELINE_RUN_ID="${RUN_ID}"
export PIPELINE_LOG_DIR="${PIPELINE_LOG_DIR:-/outputs/logs}"
export FARM_ROOT="${FARM_ROOT:-/farm_src}"
export SCENE_GRAPH_MODEL_DIR="${SCENE_GRAPH_MODEL_DIR:-/models}"
export MOBILECLIP_CHECKPOINT="${MOBILECLIP_CHECKPOINT:-/models/mobileclip/mobileclip_blt.pt}"
export MOBILECLIP_BLT_CKPT="${MOBILECLIP_BLT_CKPT:-$MOBILECLIP_CHECKPOINT}"
export MOBILECLIP_WEIGHTS_DIR="${MOBILECLIP_WEIGHTS_DIR:-/models/mobileclip}"
export CONSTRUCTION_VOCAB="${CONSTRUCTION_VOCAB:-/vocab/construction_vocab.txt}"

PHASE15="${PHASE15_DIR:-/phase1.5}"
PHASE2_OUT="${PHASE2_DIR:-/phase2}"
PHASE3_OUT="${PHASE3_DIR:-/phase3}"
DEVICE="${DEVICE:-cuda}"
CONF="${YOLOE_CONF:-0.35}"
STRIDE="${KF_STRIDE:-1}"

ts() { date -Iseconds; }

run_stage() {
  local stage="$1"; shift
  local stage_log="${LOG_DIR}/${stage}.log"
  local merged="${LOG_DIR}/pipeline.log"
  echo "[$(ts)] [${stage}] [INFO] =================================================" | tee -a "${stage_log}" | tee -a "${merged}"
  echo "[$(ts)] [${stage}] [INFO] START $*" | tee -a "${stage_log}" | tee -a "${merged}"
  local t0 t1 elapsed
  t0=$(date +%s)
  set +e
  "$@" 2>&1 | tee -a "${stage_log}" | tee -a "${merged}"
  local code=${PIPESTATUS[0]}
  set -e
  t1=$(date +%s)
  elapsed=$((t1 - t0))
  echo "[$(ts)] [${stage}] [INFO] END exit=${code} elapsed_s=${elapsed}" | tee -a "${stage_log}" | tee -a "${merged}"
  echo "${elapsed}" > "${LOG_DIR}/${stage}.elapsed"
  return ${code}
}

FRAMES_JSON="${PHASE15}/frames_json/frames.json"
if [[ ! -f "${FRAMES_JSON}" ]]; then
  echo "[$(ts)] [farm] [ERROR] missing ${FRAMES_JSON} — run Phase 1.5 first" >&2
  exit 1
fi

mkdir -p "${PHASE2_OUT}" "${PHASE3_OUT}"

export PYTHONPATH="/workspace/common:/workspace/phase2:/workspace/phase3:/farm_src/src:${PYTHONPATH:-}"

# Phase 2
run_stage phase2 \
  python -u /workspace/phase2/phase2_runner.py \
    --frames-json "${FRAMES_JSON}" \
    --data-root "${PHASE15}" \
    --output-dir "${PHASE2_OUT}" \
    --device "${DEVICE}" \
    --conf "${CONF}" \
    --stride "${STRIDE}" \
    --vocab "${CONSTRUCTION_VOCAB}"

# Phase 3
run_stage phase3 \
  python -u /workspace/phase3/run_phase3.py \
    --det-dir "${PHASE2_OUT}" \
    --output-dir "${PHASE3_OUT}" \
    --device "${DEVICE}"

echo "[$(ts)] [farm] [INFO] Phase 2 + 3 complete → ${PHASE3_OUT}/scene_state.pt"
