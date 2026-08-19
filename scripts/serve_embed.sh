#!/usr/bin/env bash
# Serve Qwen3-Embedding-0.6B (pooling) on the NetBird overlay IP only.
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
PORT="${VLLM_EMBED_PORT:-8102}"
MODEL="${VLLM_EMBED_HF_CKPT:-Qwen/Qwen3-Embedding-0.6B}"
SERVED="${VLLM_EMBED_SERVED_NAME:-qwen3-emb-0.6b}"
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
  --runner pooling \
  --max-model-len 512 \
  --gpu-memory-utilization 0.15 \
  --api-key "${VLLM_API_KEY}" \
  --trust-remote-code
