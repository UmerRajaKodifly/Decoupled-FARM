#!/usr/bin/env bash
# Pin a completed run as the Phase A baseline for A/B comparison.
#
# Usage:
#   bash scripts/snapshot_baseline.sh run_20260812_130739
#   bash scripts/snapshot_baseline.sh   # uses outputs/latest

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

RUN_ID="${1:-}"
if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(readlink -f outputs/latest 2>/dev/null | xargs basename || true)"
fi
if [[ -z "${RUN_ID}" || ! -d "outputs/runs/${RUN_ID}" ]]; then
  echo "ERROR: run not found. Usage: bash scripts/snapshot_baseline.sh <run_id>"
  exit 1
fi

SRC="outputs/runs/${RUN_ID}"
DEST="outputs/baselines/${RUN_ID}"
mkdir -p outputs/baselines
rm -f outputs/baseline
ln -sfn "${RUN_ID}" outputs/baseline

if [[ -e "${DEST}" ]]; then
  echo "Baseline already exists: ${DEST}"
else
  ln -sfn "../runs/${RUN_ID}" "${DEST}"
fi

cat > outputs/baselines/manifest.json <<EOF
{
  "baseline_run_id": "${RUN_ID}",
  "baseline_dir": "${SRC}",
  "viewer_data": "${SRC}/validation/3d-viewer",
  "snapshot_at": "$(date -Iseconds)",
  "label": "baseline (48-class vocab, conf=0.35)"
}
EOF

echo "Baseline pinned:"
echo "  run_id  = ${RUN_ID}"
echo "  path    = ${SRC}"
echo "  symlink = outputs/baseline -> ${RUN_ID}"
echo "  manifest= outputs/baselines/manifest.json"
