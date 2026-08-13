#!/usr/bin/env bash
# Re-run farm stages (Phase 2–4a) against a pinned baseline Stella+DA3 run.
# Defaults match the original full-vocab pipeline; override via env or the compare UI.
#
# Prerequisite:
#   bash scripts/snapshot_baseline.sh <baseline_run_id>
#
# Usage:
#   bash scripts/run_experiment.sh
#   YOLOE_CONF=0.38 LABEL_MIN_SCORE=0.30 bash scripts/run_experiment.sh

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

export EXPERIMENT_BASELINE_RUN_ID="${BASELINE_RUN_ID}"
export SKIP_STELLA=1
export SKIP_DA3=1
export CONSTRUCTION_VOCAB="${CONSTRUCTION_VOCAB:-${ROOT}/vocab/construction_vocab.txt}"
export YOLOE_CONF="${YOLOE_CONF:-0.35}"
export LABEL_MIN_SCORE="${LABEL_MIN_SCORE:-0.25}"
export LABEL_MARGIN="${LABEL_MARGIN:-1.15}"
export VIDEO_FILE="${VIDEO_FILE:-export_video_2.mp4}"

VOCAB_BASENAME="$(basename "${CONSTRUCTION_VOCAB}")"
EXPERIMENT_LABEL="${EXPERIMENT_LABEL:-Experiment (${VOCAB_BASENAME%.txt}, conf=${YOLOE_CONF})}"

echo "======================================================================"
echo "Farm experiment re-run"
echo "  baseline Stella/DA3 = ${BASELINE_RUN_ID}"
echo "  vocab               = ${CONSTRUCTION_VOCAB}"
echo "  YOLOE_CONF          = ${YOLOE_CONF}"
echo "  LABEL_MIN_SCORE     = ${LABEL_MIN_SCORE}"
echo "  LABEL_MARGIN        = ${LABEL_MARGIN}"
echo "======================================================================"

bash "${ROOT}/run_pipeline.sh"

RUN_ID="$(readlink -f outputs/latest | xargs basename)"
AID="$(python3 -c "import sys; sys.path.insert(0,'${ROOT}/experiments/compare-viewer'); from ablation_registry import ablation_id; print(ablation_id(float('${YOLOE_CONF}'), float('${LABEL_MIN_SCORE}'), float('${LABEL_MARGIN}')))")"
mkdir -p "${ROOT}/outputs/ablation/manifests"
ABLATION_MF="${ROOT}/outputs/ablation/manifests/${AID}.json"

python3 "${ROOT}/experiments/compare-viewer/build_manifest.py" \
  --baseline-dir "outputs/runs/${BASELINE_RUN_ID}" \
  --experiment-dir "outputs/runs/${RUN_ID}" \
  --output "${ABLATION_MF}" \
  --experiment-label "${EXPERIMENT_LABEL}"

python3 "${ROOT}/experiments/compare-viewer/register_ablation.py" \
  --baseline-run-id "${BASELINE_RUN_ID}" \
  --run-id "${RUN_ID}" \
  --conf "${YOLOE_CONF}" --vote "${LABEL_MIN_SCORE}" --margin "${LABEL_MARGIN}" \
  --vocab "$(basename "${CONSTRUCTION_VOCAB}")" \
  --note "manual rerun"

mkdir -p "${ROOT}/outputs/compare/latest"
ln -sfn "../../ablation/manifests/${AID}.json" "${ROOT}/outputs/compare/latest/manifest.json"

python3 - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone
meta = {
  "baseline_run_id": "${BASELINE_RUN_ID}",
  "experiment_run_id": "${RUN_ID}",
  "manifest": "outputs/ablation/manifests/${AID}.json",
  "ablation_id": "${AID}",
  "params": {
    "vocab": "${CONSTRUCTION_VOCAB}",
    "yoloe_conf": float("${YOLOE_CONF}"),
    "label_min_score": float("${LABEL_MIN_SCORE}"),
    "label_margin": float("${LABEL_MARGIN}"),
  },
  "finished_at": datetime.now(timezone.utc).isoformat(),
}
Path("outputs/compare/last_run.json").write_text(json.dumps(meta, indent=2))
PY

echo ""
echo "Ablation manifest → outputs/ablation/manifests/${AID}.json"
