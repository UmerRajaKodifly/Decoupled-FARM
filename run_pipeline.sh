#!/usr/bin/env bash
# End-to-end pipeline: Stella → DA3 → Phase 2 → 3 → 3.5 → 4a
#
# Per-run isolation
# -----------------
# Every run is saved under outputs/runs/<RUN_ID>/.  Existing flat outputs
# (outputs/phase1, outputs/phase2, …) from the original layout are archived
# automatically into outputs/runs/legacy_<timestamp>/ on the first new run.
# A outputs/latest symlink always points to the current run.
#
# Usage
# -----
#   cd /home/kodifly/Desktop/farm-git/repo
#   bash run_pipeline.sh                        # full run
#   VIDEO_FILE=other.mp4 bash run_pipeline.sh   # different video
#   KF_STRIDE=2 bash run_pipeline.sh            # every 2nd keyframe
#
# Skip stages already completed:
#   SKIP_STELLA=1 SKIP_DA3=1 bash run_pipeline.sh   # only re-run farm stages

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

# ------------------------------------------------------------------
# Full-video default — ignore inherited MAX_KFS from the shell
# ------------------------------------------------------------------
# Accidental smoke caps (e.g. export MAX_KFS=20) must not shrink runs.
# Opt-in smoke test only: SMOKE_MAX_KFS=20 bash run_pipeline.sh
if [[ -n "${SMOKE_MAX_KFS:-}" ]]; then
  export MAX_KFS="${SMOKE_MAX_KFS}"
else
  unset MAX_KFS
fi

# ------------------------------------------------------------------
# Video detection
# ------------------------------------------------------------------
VIDEO_FILE="${VIDEO_FILE:-}"
if [[ -z "${VIDEO_FILE}" ]]; then
  VIDEO_FILE="$(find "${ROOT}/inputs" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mov' \) | sort | head -n1 || true)"
  [[ -n "${VIDEO_FILE}" ]] && VIDEO_FILE="$(basename "${VIDEO_FILE}")"
fi
if [[ -z "${VIDEO_FILE}" ]]; then
  echo "ERROR: place a .mp4 in ${ROOT}/inputs/ or set VIDEO_FILE=myvideo.mp4"
  exit 1
fi

# ------------------------------------------------------------------
# Run ID + per-run directory
# ------------------------------------------------------------------
# Fresh run directory every invocation. To continue a prior run:
#   RESUME_RUN_ID=run_20260812_130739 SKIP_STELLA=1 bash run_pipeline.sh
# Reuse Stella + DA3 from a baseline for experiments:
#   EXPERIMENT_BASELINE_RUN_ID=run_20260812_130739 SKIP_STELLA=1 SKIP_DA3=1 bash run_pipeline.sh
export PIPELINE_RUN_ID="${RESUME_RUN_ID:-}"
if [[ -z "${PIPELINE_RUN_ID}" ]]; then
  suffix=""
  [[ "${DETECTOR:-yoloe}" == "sam3" ]] && suffix="_sam3"
  export PIPELINE_RUN_ID="run_$(date +%Y%m%d_%H%M%S)${suffix}"
fi
RUN_DIR="${ROOT}/outputs/runs/${PIPELINE_RUN_ID}"

# Experiment mode: symlink phase1 + phase1.5 from baseline (saves SLAM + DA3 time)
if [[ -n "${EXPERIMENT_BASELINE_RUN_ID:-}" ]]; then
  BASE_RUN="${ROOT}/outputs/runs/${EXPERIMENT_BASELINE_RUN_ID}"
  if [[ ! -d "${BASE_RUN}/phase1.5" ]]; then
    echo "ERROR: EXPERIMENT_BASELINE_RUN_ID=${EXPERIMENT_BASELINE_RUN_ID} missing phase1.5"
    exit 1
  fi
  export SKIP_STELLA="${SKIP_STELLA:-1}"
  export SKIP_DA3="${SKIP_DA3:-1}"
fi

# Archive any existing flat-layout outputs (legacy from before per-run isolation)
_LEGACY_DIRS=()
for d in phase1 phase1.5 phase2 phase3 phase3.5 phase4 validation; do
  [[ -d "${ROOT}/outputs/${d}" ]] && _LEGACY_DIRS+=("${d}")
done
if [[ ${#_LEGACY_DIRS[@]} -gt 0 ]]; then
  LEGACY_ID="legacy_$(date +%Y%m%d_%H%M%S)"
  LEGACY_DIR="${ROOT}/outputs/runs/${LEGACY_ID}"
  echo "======================================================================"
  echo "  Archiving existing flat outputs → outputs/runs/${LEGACY_ID}/"
  echo "  (original layout detected: ${_LEGACY_DIRS[*]})"
  echo "======================================================================"
  mkdir -p "${LEGACY_DIR}"
  for d in "${_LEGACY_DIRS[@]}"; do
    mv "${ROOT}/outputs/${d}" "${LEGACY_DIR}/${d}"
    echo "  moved outputs/${d} → outputs/runs/${LEGACY_ID}/${d}"
  done
  # Update latest symlink to legacy (so old viewers still work via outputs/latest)
  ln -sfn "runs/${LEGACY_ID}" "${ROOT}/outputs/latest_legacy"
  echo "  Legacy run accessible via outputs/runs/${LEGACY_ID}/"
  echo "======================================================================"
fi

# Create run dir
mkdir -p \
  "${RUN_DIR}/phase1" \
  "${RUN_DIR}/phase1.5" \
  "${RUN_DIR}/phase2" \
  "${RUN_DIR}/phase3" \
  "${RUN_DIR}/phase3.5" \
  "${RUN_DIR}/phase4" \
  "${RUN_DIR}/validation/phase1" \
  "${RUN_DIR}/validation/phase1.5" \
  "${RUN_DIR}/validation/phase2" \
  "${RUN_DIR}/validation/phase3" \
  "${RUN_DIR}/validation/phase3.5" \
  "${RUN_DIR}/validation/phase4" \
  "${ROOT}/outputs/logs/${PIPELINE_RUN_ID}"

# Link phase1 / phase1.5 from baseline after empty dirs were created
if [[ -n "${EXPERIMENT_BASELINE_RUN_ID:-}" ]]; then
  BASE_RUN="${ROOT}/outputs/runs/${EXPERIMENT_BASELINE_RUN_ID}"
  for phase in phase1 phase1.5; do
    rm -rf "${RUN_DIR}/${phase}"
    ln -sfn "${BASE_RUN}/${phase}" "${RUN_DIR}/${phase}"
    echo "[$(date -Iseconds)] Linked ${phase} ← ${EXPERIMENT_BASELINE_RUN_ID}" >> "${ROOT}/outputs/logs/${PIPELINE_RUN_ID}/pipeline.log" 2>/dev/null || true
  done
fi

# Export per-phase host paths for docker-compose volume substitution
export RUN_PHASE1_HOST="${RUN_DIR}/phase1"
export RUN_PHASE15_HOST="${RUN_DIR}/phase1.5"
export RUN_PHASE2_HOST="${RUN_DIR}/phase2"
export RUN_PHASE3_HOST="${RUN_DIR}/phase3"
export RUN_PHASE35_HOST="${RUN_DIR}/phase3.5"
export RUN_PHASE4_HOST="${RUN_DIR}/phase4"
export RUN_VAL_HOST="${RUN_DIR}/validation"
export PIPELINE_LOG_DIR="${ROOT}/outputs/logs"
export VIDEO_FILE
export CONSTRUCTION_VOCAB="${CONSTRUCTION_VOCAB:-${ROOT}/vocab/construction_vocab.txt}"
# Docker farm service sees vocab under /vocab/
if [[ "${CONSTRUCTION_VOCAB}" == "${ROOT}/vocab/"* ]]; then
  export CONSTRUCTION_VOCAB_DOCKER="/vocab/${CONSTRUCTION_VOCAB#${ROOT}/vocab/}"
else
  export CONSTRUCTION_VOCAB_DOCKER="${CONSTRUCTION_VOCAB}"
fi
export YOLOE_CONF="${YOLOE_CONF:-}"
export LABEL_MIN_SCORE="${LABEL_MIN_SCORE:-}"
export LABEL_MARGIN="${LABEL_MARGIN:-}"
export DETECTOR="${DETECTOR:-yoloe}"
export SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-}"

# Update latest symlink
ln -sfn "runs/${PIPELINE_RUN_ID}" "${ROOT}/outputs/latest"

MERGED="${ROOT}/outputs/logs/${PIPELINE_RUN_ID}/pipeline.log"
ts() { date -Iseconds; }

banner() {
  {
    echo "=================================================================="
    echo "[$(ts)] $*"
    echo "  run_id  = ${PIPELINE_RUN_ID}"
    echo "  run_dir = outputs/runs/${PIPELINE_RUN_ID}/"
    echo "  video   = ${VIDEO_FILE}"
    echo "  max_kfs = ${MAX_KFS:-all Stella keyframes}"
    echo "  vocab   = ${CONSTRUCTION_VOCAB}"
    echo "  detector= ${DETECTOR:-yoloe}"
    [[ -n "${EXPERIMENT_BASELINE_RUN_ID:-}" ]] && echo "  baseline= ${EXPERIMENT_BASELINE_RUN_ID} (shared phase1/1.5)"
    echo "=================================================================="
  } | tee -a "${MERGED}"
}

fmt_s() { printf '%dm %02ds' $(($1 / 60)) $(($1 % 60)); }

elapsed_file() {
  local f="${ROOT}/outputs/logs/${PIPELINE_RUN_ID}/$1.elapsed"
  [[ -f "$f" ]] && cat "$f" || echo 0
}

# Host-side validation runner (after Stella/DA3 Docker stages return)
run_validate_host() {
  local stage="$1"; shift
  local val_log="${ROOT}/outputs/logs/${PIPELINE_RUN_ID}/validate_${stage}_host.log"
  echo "[$(ts)] [validate:${stage}] host start" | tee -a "${MERGED}"
  set +e
  conda run -n farm-phase2 python "$@" 2>&1 | tee -a "${val_log}" | tee -a "${MERGED}"
  local code=${PIPESTATUS[0]}
  set -e
  if [[ "${code}" -ne 0 && "${STRICT_VALIDATE:-0}" == "1" ]]; then
    echo "[$(ts)] [validate:${stage}] STRICT abort" >&2; exit "${code}"
  fi
  [[ "${code}" -eq 0 ]] && echo "[$(ts)] [validate:${stage}] PASS" | tee -a "${MERGED}" \
                         || echo "[$(ts)] [validate:${stage}] WARN exit=${code}" | tee -a "${MERGED}"
}

# Model check
check_models() {
  local missing=0
  [[ -f "${ROOT}/models/orb_vocab.fbow" ]]              || { echo "missing orb_vocab.fbow"; missing=1; }
  [[ -d "${ROOT}/models/dinov3-vits16" ]]                || { echo "missing dinov3-vits16/"; missing=1; }
  if [[ "${DETECTOR:-yoloe}" == "sam3" ]]; then
    # Runtime uses the baked image path /opt/sam3/sam3.pt. A host copy is only
    # required when building/rebuilding the farm image.
    if [[ ! -f "${ROOT}/models/sam3/sam3.pt" ]]; then
      if docker image inspect farm-e2e-farm:latest >/dev/null 2>&1 \
        && docker run --rm --entrypoint python3 farm-e2e-farm:latest -c \
             "from pathlib import Path; raise SystemExit(0 if Path('/opt/sam3/sam3.pt').is_file() else 1)" \
             >/dev/null 2>&1; then
        echo "[$(ts)] SAM3 checkpoint present in farm image (/opt/sam3/sam3.pt)"
      elif [[ -n "${HF_TOKEN:-}" ]]; then
        echo "[$(ts)] models/sam3/sam3.pt missing — fetching via bootstrap_models.sh (needed to bake into image)"
        bash "${ROOT}/bootstrap_models.sh" --skip-da3 || true
        [[ -f "${ROOT}/models/sam3/sam3.pt" ]] || { echo "missing models/sam3/sam3.pt after bootstrap"; missing=1; }
      else
        echo "missing models/sam3/sam3.pt (needed to build farm image; or use an image that already bakes /opt/sam3/sam3.pt)"
        missing=1
      fi
    fi
  else
    [[ -f "${ROOT}/models/yoloe/yoloe-v8l-seg.pt" ]]      || { echo "missing yoloe-v8l-seg.pt"; missing=1; }
    [[ -f "${ROOT}/models/yoloe/yoloe-v8l-seg-pf.pt" ]]   || { echo "missing yoloe-v8l-seg-pf.pt"; missing=1; }
    [[ -f "${ROOT}/models/mobileclip/mobileclip_blt.pt" ]] || { echo "missing mobileclip_blt.pt"; missing=1; }
  fi
  if [[ "${missing}" -ne 0 ]]; then
    echo "Run: bash ${ROOT}/bootstrap_models.sh"; exit 1
  fi
}

run_service() {
  local name="$1"
  banner "STAGE ${name}"
  local t0 t1
  t0=$(date +%s)
  docker compose run --rm \
    -e PIPELINE_RUN_ID="${PIPELINE_RUN_ID}" \
    -e VIDEO_FILE="${VIDEO_FILE}" \
    -e MAX_KFS="${MAX_KFS:-}" \
    -e CONSTRUCTION_VOCAB="${CONSTRUCTION_VOCAB_DOCKER:-${CONSTRUCTION_VOCAB}}" \
    -e YOLOE_CONF="${YOLOE_CONF:-}" \
    -e LABEL_MIN_SCORE="${LABEL_MIN_SCORE:-}" \
    -e LABEL_MARGIN="${LABEL_MARGIN:-}" \
    -e DETECTOR="${DETECTOR:-yoloe}" \
    -e SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-}" \
    -e RUN_PHASE1_HOST="${RUN_PHASE1_HOST}" \
    -e RUN_PHASE15_HOST="${RUN_PHASE15_HOST}" \
    -e RUN_PHASE2_HOST="${RUN_PHASE2_HOST}" \
    -e RUN_PHASE3_HOST="${RUN_PHASE3_HOST}" \
    -e RUN_PHASE35_HOST="${RUN_PHASE35_HOST}" \
    -e RUN_PHASE4_HOST="${RUN_PHASE4_HOST}" \
    -e RUN_VAL_HOST="${RUN_VAL_HOST}" \
    "${name}"
  t1=$(date +%s)
  echo "[$(ts)] STAGE ${name} wall_elapsed_s=$((t1-t0))" | tee -a "${MERGED}"
}

# ------------------------------------------------------------------
check_models
banner "PIPELINE START"

if [[ "${FORCE_BUILD:-0}" == "1" ]] || ! docker image inspect farm-e2e-stella:latest >/dev/null 2>&1; then
  echo "[$(ts)] Building Docker images…" | tee -a "${MERGED}"
  docker compose build
elif [[ "${DETECTOR:-yoloe}" == "sam3" ]]; then
  # Ensure farm image has SAM3 package + current phase2 (cheap no-op if cache warm)
  echo "[$(ts)] Ensuring farm image includes SAM3…" | tee -a "${MERGED}"
  docker compose build farm
fi

# ------------------------------------------------------------------
# Phase 1 — Stella VSLAM
# ------------------------------------------------------------------
if [[ "${SKIP_STELLA:-0}" != "1" ]]; then
  run_service stella
else
  echo "[$(ts)] [phase1] SKIPPED (SKIP_STELLA=1)" | tee -a "${MERGED}"
fi

run_validate_host phase1 \
  "${ROOT}/common/validate_phase1.py" \
  --phase1-dir "${RUN_DIR}/phase1" \
  --out-dir    "${RUN_DIR}/validation/phase1"

# ------------------------------------------------------------------
# Phase 1.5 — DA3 depth
# ------------------------------------------------------------------
if [[ "${SKIP_DA3:-0}" != "1" ]]; then
  run_service da3
else
  echo "[$(ts)] [phase1.5] SKIPPED (SKIP_DA3=1)" | tee -a "${MERGED}"
fi

run_validate_host phase1.5 \
  "${ROOT}/common/validate_phase15.py" \
  --phase15-dir "${RUN_DIR}/phase1.5" \
  --out-dir     "${RUN_DIR}/validation/phase1.5"

# ------------------------------------------------------------------
# Farm: Phase 2 → 3 → 3.5 → 4a  (all inside one container run)
# Phase-level validation runs inside the container and writes to
# ${RUN_DIR}/validation/<phase>/
# ------------------------------------------------------------------
if [[ "${SKIP_FARM:-0}" != "1" ]]; then
  run_service farm
else
  echo "[$(ts)] [farm] SKIPPED (SKIP_FARM=1)" | tee -a "${MERGED}"
fi

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
P1=$(elapsed_file phase1-slam)
P15=$(elapsed_file phase1.5-depth)
P2=$(elapsed_file phase2)
P3=$(elapsed_file phase3)
P35=$(elapsed_file phase3.5)
P4=$(elapsed_file phase4a)
TOTAL=$((P1 + P15 + P2 + P3 + P35 + P4))

BEST_SCENE=""
for candidate in \
  "${RUN_DIR}/phase4/scene_state_with_crops.pt" \
  "${RUN_DIR}/phase3.5/scene_state_stella.pt" \
  "${RUN_DIR}/phase3/scene_state.pt"; do
  [[ -f "${candidate}" ]] && { BEST_SCENE="${candidate}"; break; }
done
CROPS_DIR="${RUN_DIR}/phase4/crops"
VOCAB="${ROOT}/vocab/construction_vocab.txt"

{
  echo "=================================================================="
  echo "RUN COMPLETE: ${PIPELINE_RUN_ID}"
  echo ""
  echo "  Phase 1   (Stella):         $(fmt_s "$P1")"
  echo "  Phase 1.5 (DA3 depth):      $(fmt_s "$P15")"
  echo "  Phase 2   (Detect+Embed):   $(fmt_s "$P2")"
  echo "  Phase 3   (Fuse+Map):       $(fmt_s "$P3")"
  echo "  Phase 3.5 (Stella geom):    $(fmt_s "$P35")"
  echo "  Phase 4a  (Best crops):     $(fmt_s "$P4")"
  echo "  Total wall time:            $(fmt_s "$TOTAL")"
  echo ""
  echo "  Run directory:"
  echo "    ${RUN_DIR}/"
  echo "    outputs/latest  →  runs/${PIPELINE_RUN_ID}/"
  echo ""
  echo "  Key outputs:"
  echo "    ${RUN_DIR}/phase3/scene_state.pt"
  echo "    ${RUN_DIR}/phase3.5/scene_state_stella.pt"
  echo "    ${RUN_DIR}/phase4/scene_state_with_crops.pt"
  echo "    ${RUN_DIR}/phase4/crops/"
  echo ""
  echo "  Validation summaries:"
  echo "    cat ${RUN_DIR}/validation/phase3.5/summary.txt"
  echo "    cat ${RUN_DIR}/validation/phase4/summary.txt"
  echo "    xdg-open ${RUN_DIR}/validation/phase3/overlays_3d.html"
  echo "    xdg-open ${RUN_DIR}/validation/phase4/crop_grid.html"
  echo ""
  echo "  3D WebGL Viewer (recommended — handles large clouds):"
  echo "    conda activate farm-phase2"
  echo "    python ${ROOT}/3d-viewer/serve.py --data-dir ${RUN_DIR}/validation/3d-viewer"
  echo "    # opens http://127.0.0.1:8090 automatically"
  echo ""
  echo "  Track B — caption + visual query (host, needs GOOGLE_API_KEY):"
  echo "    export GOOGLE_API_KEY=..."
    echo "    bash ${ROOT}/scripts/run_track_b.sh"
  echo ""
  echo "  Or rebuild viewer data manually:"
  echo "    python ${ROOT}/3d-viewer/build_viewer_data.py --output-dir ${RUN_DIR}/validation/3d-viewer"
  echo ""
  if [[ -n "${BEST_SCENE}" ]]; then
    echo "  Viser viewer (legacy — for Phase 2/3 only):"
    echo "    conda activate farm-phase2"
    echo "    python ${ROOT}/viser-viewer/run_viewer.py \\"
    echo "      --scene-state ${BEST_SCENE} \\"
    [[ -d "${CROPS_DIR}" ]] && echo "      --crops-dir   ${CROPS_DIR} \\"
    echo "      --vocab       ${VOCAB} \\"
    echo "      --cube-opacity 0.12"
    echo "    # then open http://127.0.0.1:8080"
  fi
  echo ""
  echo "  Compare runs:"
  echo "    ls ${ROOT}/outputs/runs/"
  echo "=================================================================="
} | tee -a "${MERGED}"
