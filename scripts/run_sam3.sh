#!/usr/bin/env bash
# SAM3 Phase 2–4 experiment against a pinned Stella + DA3 baseline.
# Cuboid faces from Phase 1.5 are reused; only the detector changes.
#
# Packaging
# ---------
# • SAM3 *code* is baked into the farm Docker image (/opt/sam3) at build time.
# • SAM3 *checkpoint* is gated on Hugging Face and lives on the host at
#   models/sam3/sam3.pt (mounted into the container as /models/sam3/sam3.pt).
#
# Fresh machine / after git pull:
#   1. Accept the license: https://huggingface.co/facebook/sam3
#   2. HF_TOKEN=hf_... bash bootstrap_models.sh   # downloads sam3.pt (+ other weights)
#   3. docker compose build farm                  # this script does this for you
#   4. bash scripts/run_sam3.sh
#
# Full pipeline without a baseline (runs Stella + DA3 + SAM3):
#   HF_TOKEN=hf_... bash bootstrap_models.sh
#   docker compose build
#   DETECTOR=sam3 bash run_pipeline.sh
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

# Auto-fetch gated checkpoint when missing and HF_TOKEN is available
if [[ ! -f "${ROOT}/models/sam3/sam3.pt" ]]; then
  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "models/sam3/sam3.pt missing — downloading via bootstrap_models.sh…"
    bash "${ROOT}/bootstrap_models.sh" --skip-da3
  fi
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
# Always rebuild farm so /opt/sam3 + phase2/sam3_segmenter.py match the repo
export FORCE_BUILD="${FORCE_BUILD:-1}"

echo "======================================================================"
echo "SAM3 farm experiment (shared Stella + DA3)"
echo "  baseline geometry = ${BASELINE_RUN_ID}"
echo "  vocab             = ${CONSTRUCTION_VOCAB}"
echo "  conf              = ${YOLOE_CONF}"
echo "  checkpoint        = models/sam3/sam3.pt"
echo "======================================================================"

echo "Building farm image (SAM3 package + Phase 2 code)…"
docker compose build farm

# Sanity: code import + checkpoint mount path
docker compose run --rm --entrypoint python3 farm -c "
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import importlib.util, pathlib
assert pathlib.Path('/workspace/phase2/sam3_segmenter.py').is_file(), 'sam3_segmenter.py missing from image — rebuild farm'
assert pathlib.Path('/models/sam3/sam3.pt').is_file(), 'checkpoint not mounted at /models/sam3/sam3.pt'
print('SAM3 image OK: package + segmenter + checkpoint')
"

bash "${ROOT}/run_pipeline.sh"

RUN_ID="$(readlink -f outputs/latest | xargs basename)"
echo ""
echo "SAM3 run complete: ${RUN_ID}"
echo "  detections → outputs/runs/${RUN_ID}/phase2/"
echo "  map        → outputs/runs/${RUN_ID}/phase3/"
echo "  viewer     → python 3d-viewer/serve.py --data-dir outputs/runs/${RUN_ID}/validation/3d-viewer"
echo "  or pick this run in the compare viewer dropdown after registering it."
