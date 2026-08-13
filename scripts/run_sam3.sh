#!/usr/bin/env bash
# SAM3 Phase 2–4 experiment against a pinned Stella + DA3 baseline.
# Cuboid faces from Phase 1.5 are reused; only the detector changes.
#
# Results land in a new run directory (run_*_sam3) so YOLOE outputs stay intact.
#
# Prerequisite:
#   HF_TOKEN with access to https://huggingface.co/facebook/sam3
#   bash bootstrap_models.sh          # downloads models/sam3/sam3.pt
#   docker compose build farm         # installs the sam3 package in the farm image
#
# Usage:
#   bash scripts/run_sam3.sh
#   YOLOE_CONF=0.40 bash scripts/run_sam3.sh

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

BASELINE_RUN_ID="${BASELINE_RUN_ID:-}"
if [[ -z "${BASELINE_RUN_ID}" && -f outputs/baselines/manifest.json ]]; then
  BASELINE_RUN_ID="$(python3 -c "import json; print(json.load(open('outputs/baselines/manifest.json'))['baseline_run_id'])")"
fi
if [[ -z "${BASELINE_RUN_ID}" || ! -d "outputs/runs/${BASELINE_RUN_ID}/phase1.5" ]]; then
  echo "ERROR: set BASELINE_RUN_ID or run scripts/snapshot_baseline.sh first"
  exit 1
fi
if [[ ! -f "${ROOT}/models/sam3/sam3.pt" ]]; then
  echo "ERROR: missing models/sam3/sam3.pt"
  echo "  Accept the license at https://huggingface.co/facebook/sam3"
  echo "  then:  HF_TOKEN=... bash bootstrap_models.sh"
  exit 1
fi

export EXPERIMENT_BASELINE_RUN_ID="${BASELINE_RUN_ID}"
export SKIP_STELLA=1
export SKIP_DA3=1
export DETECTOR=sam3
export SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-/models/sam3/sam3.pt}"
export CONSTRUCTION_VOCAB="${CONSTRUCTION_VOCAB:-${ROOT}/vocab/construction_vocab.txt}"
export YOLOE_CONF="${YOLOE_CONF:-0.35}"
export LABEL_MIN_SCORE="${LABEL_MIN_SCORE:-0.25}"
export LABEL_MARGIN="${LABEL_MARGIN:-1.15}"
export VIDEO_FILE="${VIDEO_FILE:-export_video_2.mp4}"
export FORCE_BUILD="${FORCE_BUILD:-0}"

echo "======================================================================"
echo "SAM3 farm experiment (shared Stella + DA3)"
echo "  baseline geometry = ${BASELINE_RUN_ID}"
echo "  vocab             = ${CONSTRUCTION_VOCAB}"
echo "  conf              = ${YOLOE_CONF}"
echo "  checkpoint        = models/sam3/sam3.pt"
echo "======================================================================"

echo "Building farm image (SAM3 package)…"
docker compose build farm

bash "${ROOT}/run_pipeline.sh"

RUN_ID="$(readlink -f outputs/latest | xargs basename)"
echo ""
echo "SAM3 run complete: ${RUN_ID}"
echo "  detections → outputs/runs/${RUN_ID}/phase2/"
echo "  map        → outputs/runs/${RUN_ID}/phase3/"
echo "  viewer     → python 3d-viewer/serve.py --data-dir outputs/runs/${RUN_ID}/validation/3d-viewer"
echo "  or pick this run in the compare viewer dropdown after registering it."
