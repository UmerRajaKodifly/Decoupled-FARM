#!/usr/bin/env bash
# Farm service entrypoint — Phase 2 → 3 → 3.5 → 4a (with per-phase validation)
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

PHASE1="${PHASE1_DIR:-/phase1}"
PHASE15="${PHASE15_DIR:-/phase1.5}"
PHASE2_OUT="${PHASE2_DIR:-/phase2}"
PHASE3_OUT="${PHASE3_DIR:-/phase3}"
PHASE35_OUT="${PHASE35_DIR:-/phase3.5}"
PHASE4_OUT="${PHASE4_DIR:-/phase4}"
VAL_OUT="${VALIDATION_DIR:-/validation}"
DEVICE="${DEVICE:-cuda}"
CONF="${YOLOE_CONF:-0.35}"
STRIDE="${KF_STRIDE:-1}"
DETECTOR="${DETECTOR:-yoloe}"
STRICT="${STRICT_VALIDATE:-0}"
LABEL_MIN="${LABEL_MIN_SCORE:-0.25}"
LABEL_MARG="${LABEL_MARGIN:-1.15}"

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

# Validation stages: non-blocking unless STRICT_VALIDATE=1
run_validate() {
  local stage="$1"; shift
  local val_log="${LOG_DIR}/validate_${stage}.log"
  local merged="${LOG_DIR}/pipeline.log"
  echo "[$(ts)] [validate:${stage}] [INFO] START" | tee -a "${val_log}" | tee -a "${merged}"
  set +e
  "$@" 2>&1 | tee -a "${val_log}" | tee -a "${merged}"
  local code=${PIPESTATUS[0]}
  set -e
  if [[ "${code}" -ne 0 ]]; then
    echo "[$(ts)] [validate:${stage}] [WARN] Validation exit=${code}" | tee -a "${val_log}" | tee -a "${merged}"
    if [[ "${STRICT}" == "1" ]]; then
      echo "[$(ts)] [validate:${stage}] [ERROR] STRICT_VALIDATE=1 — aborting" >&2
      exit "${code}"
    fi
  else
    echo "[$(ts)] [validate:${stage}] [INFO] PASS" | tee -a "${val_log}" | tee -a "${merged}"
  fi
}

# ------------------------------------------------------------------
# Pre-flight checks
# ------------------------------------------------------------------
FRAMES_JSON="${PHASE15}/frames_json/frames.json"
if [[ ! -f "${FRAMES_JSON}" ]]; then
  echo "[$(ts)] [farm] [ERROR] missing ${FRAMES_JSON} — run Phase 1.5 first" >&2
  exit 1
fi

mkdir -p "${PHASE2_OUT}" "${PHASE3_OUT}" "${PHASE35_OUT}" "${PHASE4_OUT}" \
         "${VAL_OUT}/phase2" "${VAL_OUT}/phase3" \
         "${VAL_OUT}/phase3.5" "${VAL_OUT}/phase4"

export PYTHONPATH="/workspace/common:/workspace/phase2:/workspace/phase3:/workspace/phase3.5-stella-geometry:/workspace/phase4-caption-best-view:/farm_src/src:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# Phase 2 — Detect + Segment + Embed
# ------------------------------------------------------------------
run_stage phase2 \
  python -u /workspace/phase2/phase2_runner.py \
    --frames-json "${FRAMES_JSON}" \
    --data-root "${PHASE15}" \
    --output-dir "${PHASE2_OUT}" \
    --device "${DEVICE}" \
    --conf "${CONF}" \
    --stride "${STRIDE}" \
    --vocab "${CONSTRUCTION_VOCAB}" \
    --detector "${DETECTOR}"

run_validate phase2 \
  python -u /workspace/phase2/validate_phase2.py \
    --det-dir "${PHASE2_OUT}" \
    --out-dir "${VAL_OUT}/phase2" \
    --vocab-file "${CONSTRUCTION_VOCAB}"

# ------------------------------------------------------------------
# Phase 3 — Associate + Fuse + Map
# ------------------------------------------------------------------
run_stage phase3 \
  python -u /workspace/phase3/run_phase3.py \
    --det-dir "${PHASE2_OUT}" \
    --output-dir "${PHASE3_OUT}" \
    --device "${DEVICE}" \
    --label-min-score "${LABEL_MIN}" \
    --label-margin "${LABEL_MARG}"

run_validate phase3 \
  python -u /workspace/phase3/validate_phase3.py \
    --output-dir "${PHASE3_OUT}" \
    --out-dir "${VAL_OUT}/phase3" \
    --vocab-file "${CONSTRUCTION_VOCAB}"

# ------------------------------------------------------------------
# Phase 3.5 — Stella PCD geometry refinement
# ------------------------------------------------------------------
if [[ -f "${PHASE1}/out.db" ]]; then
  run_stage phase3.5 \
    python -u /workspace/phase3.5-stella-geometry/run_phase35.py \
      --phase1-dir  "${PHASE1}" \
      --phase15-dir "${PHASE15}" \
      --det-dir     "${PHASE2_OUT}" \
      --scene-state "${PHASE3_OUT}/scene_state.pt" \
      --output-dir  "${PHASE35_OUT}"

  run_validate phase3.5 \
    python -u /workspace/phase3.5-stella-geometry/validate_phase35.py \
      --phase3-state "${PHASE3_OUT}/scene_state.pt" \
      --stella-state "${PHASE35_OUT}/scene_state_stella.pt" \
      --summary-json "${PHASE35_OUT}/phase35_summary.json" \
      --out-dir      "${VAL_OUT}/phase3.5"

  BEST_SCENE="${PHASE35_OUT}/scene_state_stella.pt"
else
  echo "[$(ts)] [phase3.5] [WARN] out.db not found at ${PHASE1}/out.db — skipping Stella geometry"
  BEST_SCENE="${PHASE3_OUT}/scene_state.pt"
fi

# ------------------------------------------------------------------
# Phase 4a — Best-view crops
# ------------------------------------------------------------------
run_stage phase4a \
  python -u /workspace/phase4-caption-best-view/run_phase4_crops.py \
    --scene-state "${BEST_SCENE}" \
    --det-dir     "${PHASE2_OUT}" \
    --output-dir  "${PHASE4_OUT}"

run_validate phase4 \
  python -u /workspace/phase4-caption-best-view/validate_phase4.py \
    --phase4-dir "${PHASE4_OUT}" \
    --out-dir    "${VAL_OUT}/phase4" \
    --vocab-file "${CONSTRUCTION_VOCAB}"

# ------------------------------------------------------------------
# 3D Viewer data build (non-fatal — viewer is a convenience tool)
# ------------------------------------------------------------------
echo "[$(ts)] [3d-viewer] [INFO] Building WebGL viewer data …"
python -u /workspace/3d-viewer/build_viewer_data.py \
  --stella-state "${BEST_SCENE}" \
  --db-path      "${PHASE1}/out.db" \
  --crops-dir    "${PHASE4_OUT}/crops" \
  --vocab-file   "${CONSTRUCTION_VOCAB}" \
  --output-dir   "${VAL_OUT}/3d-viewer" \
  --voxel-size   0.05 || \
  echo "[$(ts)] [3d-viewer] [WARN] Viewer data build failed (non-fatal)"

echo "[$(ts)] [farm] [INFO] ================================================="
echo "[$(ts)] [farm] [INFO] ALL STAGES COMPLETE"
echo "[$(ts)] [farm] [INFO]   Phase 3  → ${PHASE3_OUT}/scene_state.pt"
echo "[$(ts)] [farm] [INFO]   Phase 3.5→ ${PHASE35_OUT}/scene_state_stella.pt"
echo "[$(ts)] [farm] [INFO]   Phase 4a → ${PHASE4_OUT}/scene_state_with_crops.pt"
echo "[$(ts)] [farm] [INFO]   Crops    → ${PHASE4_OUT}/crops/"
echo "[$(ts)] [farm] [INFO]   Validate → ${VAL_OUT}/"
echo "[$(ts)] [farm] [INFO]   3D Viewer→ ${VAL_OUT}/3d-viewer/"
echo "[$(ts)] [farm] [INFO] ================================================="
