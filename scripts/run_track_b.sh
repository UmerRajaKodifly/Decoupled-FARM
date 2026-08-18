#!/usr/bin/env bash
# Track B — Gemini captioning → embeddings → query index → viewer rebuild
#
# Runs on the HOST (needs GOOGLE_API_KEY or use MOCK=1 for offline smoke test).
# Expects Phase 4a complete: scene_state_with_crops.pt + phase1.5/faces/ + crops/ fallback
#
# Usage:
#   export GOOGLE_API_KEY=...
#   bash scripts/run_track_b.sh
#
#   MOCK=1 MAX_OBJECTS=50 bash scripts/run_track_b.sh   # offline dev
#   RUN_DIR=outputs/runs/run_XXXX bash scripts/run_track_b.sh

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

P4Q="${ROOT}/phase4-visual-query"
RUN_DIR="${RUN_DIR:-${ROOT}/outputs/latest}"
PHASE4="${RUN_DIR}/phase4"
VIEWER_OUT="${VIEWER_OUT:-${RUN_DIR}/validation/3d-viewer-trackb}"
VOCAB="${ROOT}/vocab/construction_vocab.txt"

SCENE_IN="${SCENE_IN:-${PHASE4}/scene_state_with_crops.pt}"
MOCK="${MOCK:-0}"
MAX_OBJECTS="${MAX_OBJECTS:-0}"
CAPTION_MODEL="${CAPTION_MODEL:-gemini-3.0-flash}"
EMBED_MODEL="${EMBED_MODEL:-text-embedding-004}"

ts() { date -Iseconds; }

if [[ ! -f "${SCENE_IN}" ]]; then
  echo "[track-b] ERROR: missing ${SCENE_IN} — run Phase 4a first"
  exit 2
fi

mkdir -p "${PHASE4}"

echo "[$(ts)] [track-b] Installing phase4-visual-query deps …"
pip install -q -r "${P4Q}/requirements.txt"

MOCK_FLAG=()
[[ "${MOCK}" == "1" ]] && MOCK_FLAG=(--mock)

MAX_FLAG=()
[[ "${MAX_OBJECTS}" != "0" ]] && MAX_FLAG=(--max-objects "${MAX_OBJECTS}")

FACES_FLAG=()
if [[ -d "${RUN_DIR}/phase1.5/faces" ]]; then
  FACES_FLAG=(--faces-dir "${RUN_DIR}/phase1.5/faces")
fi

echo "[$(ts)] [track-b] Phase 4b — captioning (full face + bbox) …"
python -u "${P4Q}/run_phase4b_caption.py" \
  --scene-state "${SCENE_IN}" \
  --output-dir "${PHASE4}" \
  --vocab-file "${VOCAB}" \
  --cache-dir "${PHASE4}/gemini_cache" \
  --caption-model "${CAPTION_MODEL}" \
  "${FACES_FLAG[@]}" \
  "${MOCK_FLAG[@]}" \
  "${MAX_FLAG[@]}"

echo "[$(ts)] [track-b] Phase 4c — embeddings …"
python -u "${P4Q}/run_phase4c_embed.py" \
  --scene-state "${PHASE4}/scene_state_captioned.pt" \
  --output-dir "${PHASE4}" \
  --cache-dir "${PHASE4}/gemini_cache" \
  --embed-model "${EMBED_MODEL}" \
  "${MOCK_FLAG[@]}"

echo "[$(ts)] [track-b] Phase 4d — post-caption merge …"
python -u "${P4Q}/run_phase4d_merge.py" \
  --scene-state "${PHASE4}/scene_state_enriched.pt" \
  --output-dir "${PHASE4}"

SCENE_MERGED="${PHASE4}/scene_state_merged.pt"
SCENE_QUERY="${SCENE_MERGED}"
if [[ ! -f "${SCENE_MERGED}" ]]; then
  SCENE_QUERY="${PHASE4}/scene_state_enriched.pt"
fi

echo "[$(ts)] [track-b] Building query index …"
python -u "${P4Q}/run_build_query_index.py" \
  --scene-state "${SCENE_QUERY}" \
  --output "${PHASE4}/query_index.json" \
  --vocab-file "${VOCAB}"

echo "[$(ts)] [track-b] Rebuilding 3D viewer data …"
if python -u "${ROOT}/3d-viewer/build_viewer_data.py" \
  --stella-state "${SCENE_QUERY}" \
  --crops-dir "${PHASE4}/crops" \
  --vocab-file "${VOCAB}" \
  --output-dir "${VIEWER_OUT}"; then
  echo "[$(ts)] [track-b] Viewer data → ${VIEWER_OUT}"
else
  echo "[$(ts)] [track-b] WARN: viewer rebuild failed (often root-owned outputs/latest/validation/3d-viewer from Docker)"
  mkdir -p "${VIEWER_OUT}"
  if [[ -f "${PHASE4}/query_index.json" ]]; then
    cp -f "${PHASE4}/query_index.json" "${VIEWER_OUT}/query_index.json"
    echo "[$(ts)] [track-b] Copied query_index.json to ${VIEWER_OUT}/ (serve with --data-dir; needs objects.json from a prior build or rebuild after chown)"
  fi
fi

echo "[$(ts)] [track-b] DONE"
echo "  Scene (enriched): ${PHASE4}/scene_state_enriched.pt"
echo "  Scene (merged):   ${SCENE_MERGED}"
echo "  Query index:      ${PHASE4}/query_index.json"
echo "  Viewer data:      ${VIEWER_OUT}/"
echo ""
echo "  CLI query:"
echo "    python ${P4Q}/run_query.py \"shipping container\" --scene-state ${SCENE_QUERY} ${MOCK_FLAG[*]}"
echo ""
echo "  Viewer:"
echo "    python ${ROOT}/3d-viewer/serve.py --data-dir ${VIEWER_OUT} --port 8090"
