#!/usr/bin/env bash
# Serve Qwen3-VL-8B-Instruct-FP8 on the NetBird overlay IP only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
set -a
source "${ROOT}/.env"
set +a

: "${VLLM_API_KEY:?VLLM_API_KEY missing in .env}"
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

HOST="${VLLM_BIND_HOST:-0.0.0.0}"
PORT="${VLLM_VLM_PORT:-8100}"
MODEL="${VLLM_VLM_HF_CKPT:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
SERVED="${VLLM_VLM_SERVED_NAME:-qwen3-vl-8b}"
PYTHON="${ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "missing ${PYTHON} — create the 3.12 venv first" >&2
  exit 1
fi

exec "${PYTHON}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name "${SERVED}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype auto \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.70 \
  --limit-mm-per-prompt '{"image": 1}' \
  --api-key "${VLLM_API_KEY}" \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --trust-remote-code
