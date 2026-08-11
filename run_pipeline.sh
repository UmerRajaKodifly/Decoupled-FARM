#!/usr/bin/env bash
# Run the full MP4 → Phase 1 → 1.5 → 2 → 3 pipeline with run-scoped logs.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

VIDEO_FILE="${VIDEO_FILE:-}"
if [[ -z "${VIDEO_FILE}" ]]; then
  # Auto-pick first video under inputs/
  VIDEO_FILE="$(find "${ROOT}/inputs" -maxdepth 1 -type f \( -iname '*.mp4' -o -iname '*.mov' \) | sort | head -n1 || true)"
  if [[ -n "${VIDEO_FILE}" ]]; then
    VIDEO_FILE="$(basename "${VIDEO_FILE}")"
  fi
fi
if [[ -z "${VIDEO_FILE}" ]]; then
  echo "ERROR: place a .mp4 in ${ROOT}/inputs/ or set VIDEO_FILE=myvideo.mp4"
  exit 1
fi

export VIDEO_FILE
export PIPELINE_RUN_ID="${PIPELINE_RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"
export PIPELINE_LOG_DIR="${ROOT}/outputs/logs"
mkdir -p \
  "${ROOT}/outputs/phase1" \
  "${ROOT}/outputs/phase1.5" \
  "${ROOT}/outputs/phase2" \
  "${ROOT}/outputs/phase3" \
  "${ROOT}/outputs/logs/${PIPELINE_RUN_ID}"

MERGED="${ROOT}/outputs/logs/${PIPELINE_RUN_ID}/pipeline.log"
ts() { date -Iseconds; }
banner() {
  {
    echo "================================================="
    echo "[$(ts)] $* "
    echo "  run_id=${PIPELINE_RUN_ID}"
    echo "  video=${VIDEO_FILE}"
    echo "================================================="
  } | tee -a "${MERGED}"
}

check_models() {
  local missing=0
  [[ -f "${ROOT}/models/orb_vocab.fbow" ]] || { echo "missing models/orb_vocab.fbow"; missing=1; }
  [[ -f "${ROOT}/models/yoloe/yoloe-v8l-seg.pt" ]] || { echo "missing models/yoloe/yoloe-v8l-seg.pt"; missing=1; }
  [[ -f "${ROOT}/models/yoloe/yoloe-v8l-seg-pf.pt" ]] || { echo "missing models/yoloe/yoloe-v8l-seg-pf.pt"; missing=1; }
  [[ -f "${ROOT}/models/mobileclip/mobileclip_blt.pt" ]] || { echo "missing models/mobileclip/mobileclip_blt.pt"; missing=1; }
  [[ -d "${ROOT}/models/dinov3-vits16" ]] || { echo "missing models/dinov3-vits16/"; missing=1; }
  if [[ "${missing}" -ne 0 ]]; then
    echo "Run: bash ${ROOT}/bootstrap_models.sh"
    exit 1
  fi
}

fmt_s() {
  local s="${1:-0}"
  printf '%dm %02ds' $((s / 60)) $((s % 60))
}

elapsed_file() {
  local f="${ROOT}/outputs/logs/${PIPELINE_RUN_ID}/$1.elapsed"
  if [[ -f "$f" ]]; then cat "$f"; else echo 0; fi
}

check_models
banner "PIPELINE START"

# Build images if missing / FORCE_BUILD=1
if [[ "${FORCE_BUILD:-0}" == "1" ]] || ! docker image inspect farm-e2e-stella:latest >/dev/null 2>&1; then
  echo "[$(ts)] Building images (first run may take a long time)…" | tee -a "${MERGED}"
  docker compose build
fi

run_service() {
  local name="$1"
  banner "STAGE ${name}"
  local t0 t1
  t0=$(date +%s)
  docker compose run --rm \
    -e PIPELINE_RUN_ID="${PIPELINE_RUN_ID}" \
    -e VIDEO_FILE="${VIDEO_FILE}" \
    "${name}"
  t1=$(date +%s)
  echo "[$(ts)] STAGE ${name} wall_elapsed_s=$((t1 - t0))" | tee -a "${MERGED}"
}

run_service stella
run_service da3
run_service farm

P1=$(elapsed_file phase1-slam)
P15=$(elapsed_file phase1.5-depth)
P2=$(elapsed_file phase2)
P3=$(elapsed_file phase3)
TOTAL=$((P1 + P15 + P2 + P3))

{
  echo "================================================="
  echo "RUN COMPLETE: ${PIPELINE_RUN_ID}"
  echo "  Phase 1  (Stella SLAM):   $(fmt_s "$P1")  ✓"
  echo "  Phase 1.5 (DA3 depth):    $(fmt_s "$P15")  ✓"
  echo "  Phase 2  (Detect+Embed):  $(fmt_s "$P2")  ✓"
  echo "  Phase 3  (Fuse+Map):      $(fmt_s "$P3")  ✓"
  echo "  Total (stage timers):     $(fmt_s "$TOTAL")"
  echo "  Outputs: ${ROOT}/outputs/"
  echo "  Logs:    ${ROOT}/outputs/logs/${PIPELINE_RUN_ID}/"
  echo "  Final:   ${ROOT}/outputs/phase3/scene_state.pt"
  echo "================================================="
} | tee -a "${MERGED}"
