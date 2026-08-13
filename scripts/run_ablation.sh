#!/usr/bin/env bash
# One-at-a-time ablation on construction_vocab.txt (48 classes).
# Holds two params at baseline while sweeping the third.
#
# Usage:
#   bash scripts/run_ablation.sh              # full grid (~8 runs × ~5 min)
#   bash scripts/run_ablation.sh --register-baseline-only

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

REGISTER_ONLY=0
[[ "${1:-}" == "--register-baseline-only" ]] && REGISTER_ONLY=1

BASELINE_RUN_ID="${BASELINE_RUN_ID:-}"
if [[ -z "${BASELINE_RUN_ID}" && -f outputs/baselines/manifest.json ]]; then
  BASELINE_RUN_ID="$(python3 -c "import json; print(json.load(open('outputs/baselines/manifest.json'))['baseline_run_id'])")"
fi
if [[ -z "${BASELINE_RUN_ID}" ]]; then
  echo "ERROR: pin baseline first: bash scripts/snapshot_baseline.sh <run_id>"
  exit 1
fi

VOCAB="${ROOT}/vocab/construction_vocab.txt"
ABLATION_DIR="${ROOT}/outputs/ablation/manifests"
mkdir -p "${ABLATION_DIR}"

register_run() {
  local conf="$1" vote="$2" margin="$3" run_id="$4" note="${5:-}"
  python3 "${ROOT}/experiments/compare-viewer/register_ablation.py" \
    --baseline-run-id "${BASELINE_RUN_ID}" \
    --run-id "${run_id}" \
    --conf "${conf}" --vote "${vote}" --margin "${margin}" \
    --vocab construction_vocab.txt \
    --note "${note}"
}

build_manifest() {
  local conf="$1" vote="$2" margin="$3" exp_run="$4"
  local aid
  aid="$(python3 -c "import sys; sys.path.insert(0,'${ROOT}/experiments/compare-viewer'); from ablation_registry import ablation_id; print(ablation_id(${conf}, ${vote}, ${margin}))")"
  local out="${ABLATION_DIR}/${aid}.json"
  python3 "${ROOT}/experiments/compare-viewer/build_manifest.py" \
    --baseline-dir "outputs/runs/${BASELINE_RUN_ID}" \
    --experiment-dir "outputs/runs/${exp_run}" \
    --output "${out}" \
    --experiment-label "conf=${conf} vote=${vote} margin=${margin}"
  echo "${out}"
}

# Baseline params match the pinned full-vocab run
register_run 0.35 0.25 1.15 "${BASELINE_RUN_ID}" "pinned baseline (full pipeline)"
build_manifest 0.35 0.25 1.15 "${BASELINE_RUN_ID}" >/dev/null

if [[ "${REGISTER_ONLY}" -eq 1 ]]; then
  echo "Registered baseline only."
  exit 0
fi

# Ablation grid: one parameter varied at a time
GRID=(
  "0.30 0.25 1.15 conf_sweep"
  "0.40 0.25 1.15 conf_sweep"
  "0.35 0.20 1.15 vote_sweep"
  "0.35 0.30 1.15 vote_sweep"
  "0.35 0.40 1.15 vote_sweep"
  "0.35 0.25 1.10 margin_sweep"
  "0.35 0.25 1.25 margin_sweep"
  "0.35 0.25 1.50 margin_sweep"
)

echo "======================================================================"
echo "Ablation study (${#GRID[@]} runs + baseline registered)"
echo "  baseline = ${BASELINE_RUN_ID}"
echo "  vocab    = construction_vocab.txt"
echo "======================================================================"

for spec in "${GRID[@]}"; do
  read -r CONF VOTE MARGIN NOTE <<< "${spec}"
  AID="$(python3 -c "import sys; sys.path.insert(0,'${ROOT}/experiments/compare-viewer'); from ablation_registry import ablation_id; print(ablation_id(${CONF}, ${VOTE}, ${MARGIN}))")"

  if [[ -f "outputs/ablation/manifests/${AID}.json" ]]; then
    EXISTING="$(python3 -c "
import json
for e in json.load(open('outputs/ablation/index.json')).get('experiments',[]):
    if e['id']=='${AID}':
        print(e['run_id']); break
")"
    if [[ -n "${EXISTING}" && -d "outputs/runs/${EXISTING}/phase3" ]]; then
      echo "[skip] ${AID} already complete (run ${EXISTING})"
      continue
    fi
  fi

  echo ""
  echo "---- ${AID} (${NOTE}) conf=${CONF} vote=${VOTE} margin=${MARGIN} ----"
  export EXPERIMENT_BASELINE_RUN_ID="${BASELINE_RUN_ID}"
  export SKIP_STELLA=1 SKIP_DA3=1
  export CONSTRUCTION_VOCAB="${VOCAB}"
  export YOLOE_CONF="${CONF}"
  export LABEL_MIN_SCORE="${VOTE}"
  export LABEL_MARGIN="${MARGIN}"
  export VIDEO_FILE="${VIDEO_FILE:-export_video_2.mp4}"
  export EXPERIMENT_LABEL="conf=${CONF} vote=${VOTE} margin=${MARGIN}"

  bash "${ROOT}/run_pipeline.sh"

  RUN_ID="$(readlink -f outputs/latest | xargs basename)"
  MF="$(build_manifest "${CONF}" "${VOTE}" "${MARGIN}" "${RUN_ID}")"
  register_run "${CONF}" "${VOTE}" "${MARGIN}" "${RUN_ID}" "${NOTE}"
  echo "  → run ${RUN_ID} manifest ${MF}"
done

ln -sfn "../../ablation/manifests/c035_v025_m115.json" outputs/compare/latest/manifest.json 2>/dev/null || true

echo ""
echo "Ablation complete. Index: outputs/ablation/index.json"
python3 -c "import json; d=json.load(open('outputs/ablation/index.json')); print(f\"  {len(d['experiments'])} experiments registered\")"
