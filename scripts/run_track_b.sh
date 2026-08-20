#!/usr/bin/env bash
# Track B — HK vLLM captioning → embeddings → query index → viewer rebuild
#
# Runs on the HOST. Requires VLLM_API_KEY (and NetBird to HK).
# Expects Phase 4a complete: scene_state_with_crops.pt + phase1.5/faces/ + crops/ fallback
#
# Usage:
#   bash scripts/run_track_b.sh
#
#   MAX_OBJECTS=5 bash scripts/run_track_b.sh   # probe before full 7k run
#   RUN_DIR=outputs/runs/run_XXXX bash scripts/run_track_b.sh

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# Load secrets from .env
for _env_file in "${ROOT}/.env" "${ROOT}/../repo/.env"; do
  if [[ -f "${_env_file}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${_env_file}"
    set +a
    break
  fi
done

P4Q="${ROOT}/phase4-visual-query"
RUN_DIR="${RUN_DIR:-${ROOT}/outputs/latest}"
PHASE4="${RUN_DIR}/phase4"
VIEWER_OUT="${VIEWER_OUT:-${RUN_DIR}/validation/3d-viewer-trackb}"
VOCAB="${ROOT}/vocab/construction_vocab.txt"

SCENE_IN="${SCENE_IN:-${PHASE4}/scene_state_with_crops.pt}"
MAX_OBJECTS="${MAX_OBJECTS:-0}"
CAPTION_MODEL="${CAPTION_MODEL:-${VLLM_VL_MODEL:-qwen3-vl-8b}}"
EMBED_MODEL="${EMBED_MODEL:-${VLLM_EMBED_MODEL:-qwen3-emb-0.6b}}"

ts() { date -Iseconds; }

if [[ ! -f "${SCENE_IN}" ]]; then
  echo "[track-b] ERROR: missing ${SCENE_IN} — run Phase 4a first"
  exit 2
fi

if [[ -z "${VLLM_API_KEY:-}" ]]; then
  echo "[track-b] ERROR: set VLLM_API_KEY in ${ROOT}/.env (same value as HK spatial-gpt/.env)"
  exit 2
fi

mkdir -p "${PHASE4}"

echo "[$(ts)] [track-b] Installing phase4-visual-query deps …"
pip install -q -r "${P4Q}/requirements.txt"

MAX_FLAG=()
[[ "${MAX_OBJECTS}" != "0" ]] && MAX_FLAG=(--max-objects "${MAX_OBJECTS}")

FACES_FLAG=()
if [[ -d "${RUN_DIR}/phase1.5/faces" ]]; then
  FACES_FLAG=(--faces-dir "${RUN_DIR}/phase1.5/faces")
fi

echo "[$(ts)] [track-b] Phase 4b — captioning via HK vLLM (padded bbox crop) …"
python -u "${P4Q}/run_phase4b_caption.py" \
  --scene-state "${SCENE_IN}" \
  --output-dir "${PHASE4}" \
  --vocab-file "${VOCAB}" \
  --cache-dir "${PHASE4}/vlm_cache" \
  --caption-model "${CAPTION_MODEL}" \
  "${FACES_FLAG[@]}" \
  "${MAX_FLAG[@]}"

if [[ ! -f "${PHASE4}/scene_state_captioned.pt" ]]; then
  echo "[track-b] ERROR: Phase 4b did not write scene_state_captioned.pt"
  exit 2
fi

echo "[$(ts)] [track-b] Phase 4c — embeddings …"
python -u "${P4Q}/run_phase4c_embed.py" \
  --scene-state "${PHASE4}/scene_state_captioned.pt" \
  --output-dir "${PHASE4}" \
  --cache-dir "${PHASE4}/vlm_cache" \
  --embed-model "${EMBED_MODEL}"

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

echo "[$(ts)] [track-b] Writing caption review HTML …"
python -u "${P4Q}/build_caption_review.py" \
  --scene-state "${SCENE_QUERY}" \
  --vocab-file "${VOCAB}" \
  --output "${RUN_DIR}/validation/caption_review.html"

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
echo "  Caption review:   ${RUN_DIR}/validation/caption_review.html"
echo "  Viewer data:      ${VIEWER_OUT}/"
echo ""
echo "  CLI query:"
echo "    python ${P4Q}/run_query.py \"shipping container\" --scene-state ${SCENE_QUERY}"
echo ""
echo "  Caption review:"
echo "    xdg-open ${RUN_DIR}/validation/caption_review.html"
echo ""
echo "  Viewer:"
echo "    python ${ROOT}/3d-viewer/serve.py --data-dir ${VIEWER_OUT} --port 8090"
