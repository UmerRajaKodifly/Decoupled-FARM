#!/usr/bin/env bash
# One-time download/copy of model weights into ./models (gitignored).
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${ROOT}/models"
SKIP_DA3=0

usage() {
  cat <<'EOF'
Usage: ./bootstrap_models.sh [--skip-da3]

Fetches into models/:
  orb_vocab.fbow
  yoloe/yoloe-v8l-seg.pt + yoloe-v8l-seg-pf.pt
  mobileclip/mobileclip_blt.pt
  dinov3-vits16/   (copied from sibling FARM-Project when available)
  da3metric-large/ (HF snapshot unless --skip-da3)
  sam3/sam3.pt     (gated facebook/sam3; skipped unless HF_TOKEN is set)

Re-runs skip files already present. Does not run any git hooks/CI.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-da3) SKIP_DA3=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown: $1" >&2; usage >&2; exit 1 ;;
  esac
done

fetch() {
  local url="$1" dest="$2" min_bytes="$3"
  if [[ -f "$dest" ]]; then
    local sz
    sz=$(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest")
    if [[ "$sz" -ge "$min_bytes" ]]; then
      echo "  [skip] $(basename "$dest") already present"
      return 0
    fi
  fi
  mkdir -p "$(dirname "$dest")"
  echo "  [get ] $(basename "$dest") <- $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -C - -o "${dest}.part" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "${dest}.part" "$url"
  else
    echo "Need curl or wget" >&2; exit 1
  fi
  mv "${dest}.part" "$dest"
}

echo "Bootstrapping models → ${MODELS_DIR}"
mkdir -p "${MODELS_DIR}"

# --- ORB vocab ---
if [[ -f "${MODELS_DIR}/orb_vocab.fbow" ]]; then
  echo "  [skip] orb_vocab.fbow already present"
else
  ORB_COPIED=0
  for c in \
    "/home/kodifly/Desktop/stella-vslam-dense/data/orb_vocab.fbow" \
    "${ROOT}/../stella-vslam-dense/data/orb_vocab.fbow"
  do
    if [[ -f "$c" ]]; then
      cp "$c" "${MODELS_DIR}/orb_vocab.fbow"
      echo "  [copy] orb_vocab.fbow <- $c"
      ORB_COPIED=1
      break
    fi
  done
  if [[ "${ORB_COPIED}" -eq 0 ]]; then
    fetch \
      "https://github.com/stella-cv/FBoW_orb_vocab/raw/main/orb_vocab.fbow" \
      "${MODELS_DIR}/orb_vocab.fbow" \
      1000000 || echo "  [warn] could not obtain orb_vocab.fbow — place it at models/orb_vocab.fbow"
  fi
fi

# --- YOLOE ---
YOLOE_BASE="https://huggingface.co/jameslahm/yoloe/resolve/main"
# Prefer local FARM copy when present (offline)
if [[ -f "/home/kodifly/Desktop/farm-git/FARM-Project/models/yoloe/yoloe-v8l-seg.pt" ]]; then
  mkdir -p "${MODELS_DIR}/yoloe"
  for f in yoloe-v8l-seg.pt yoloe-v8l-seg-pf.pt; do
    if [[ ! -f "${MODELS_DIR}/yoloe/$f" ]]; then
      cp "/home/kodifly/Desktop/farm-git/FARM-Project/models/yoloe/$f" "${MODELS_DIR}/yoloe/$f"
      echo "  [copy] yoloe/$f"
    else
      echo "  [skip] yoloe/$f already present"
    fi
  done
else
  fetch "${YOLOE_BASE}/yoloe-v8l-seg.pt" "${MODELS_DIR}/yoloe/yoloe-v8l-seg.pt" 1000000
  fetch "${YOLOE_BASE}/yoloe-v8l-seg-pf.pt" "${MODELS_DIR}/yoloe/yoloe-v8l-seg-pf.pt" 1000000
fi

# --- MobileCLIP ---
if [[ -f "/home/kodifly/Desktop/farm-git/FARM-Project/models/mobileclip/mobileclip_blt.pt" ]] \
   && [[ ! -f "${MODELS_DIR}/mobileclip/mobileclip_blt.pt" ]]; then
  mkdir -p "${MODELS_DIR}/mobileclip"
  cp "/home/kodifly/Desktop/farm-git/FARM-Project/models/mobileclip/mobileclip_blt.pt" \
     "${MODELS_DIR}/mobileclip/mobileclip_blt.pt"
  echo "  [copy] mobileclip_blt.pt"
elif [[ -f "${MODELS_DIR}/mobileclip/mobileclip_blt.pt" ]]; then
  echo "  [skip] mobileclip_blt.pt already present"
else
  fetch \
    "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt" \
    "${MODELS_DIR}/mobileclip/mobileclip_blt.pt" \
    1000000
fi

# --- DINOv3 ---
if [[ -d "${MODELS_DIR}/dinov3-vits16" ]] && [[ -n "$(ls -A "${MODELS_DIR}/dinov3-vits16" 2>/dev/null || true)" ]]; then
  echo "  [skip] dinov3-vits16 already present"
else
  for c in \
    "/home/kodifly/Desktop/farm-git/FARM-Project/models/dinov3-vits16" \
    "${ROOT}/../FARM-Project/models/dinov3-vits16"
  do
    if [[ -d "$c" ]]; then
      mkdir -p "${MODELS_DIR}"
      rsync -a "$c/" "${MODELS_DIR}/dinov3-vits16/"
      echo "  [copy] dinov3-vits16 <- $c"
      break
    fi
  done
  if [[ ! -d "${MODELS_DIR}/dinov3-vits16" ]]; then
    echo "  [warn] dinov3-vits16 missing — copy into models/dinov3-vits16 before Phase 2"
  fi
fi

# --- DA3METRIC-LARGE ---
if [[ "${SKIP_DA3}" -eq 1 ]]; then
  echo "  [skip] DA3 (--skip-da3); container can fetch at runtime"
elif [[ -d "${MODELS_DIR}/da3metric-large" ]] && [[ -n "$(ls -A "${MODELS_DIR}/da3metric-large" 2>/dev/null || true)" ]]; then
  echo "  [skip] da3metric-large already present"
else
  echo "  [get ] DA3METRIC-LARGE (huggingface_hub)…"
  DEST="${MODELS_DIR}/da3metric-large" python3 - <<'PY' || echo "  [warn] DA3 pre-download failed; first da3 run will try HF hub"
from pathlib import Path
import os, sys, subprocess
dest = Path(os.environ["DEST"])
dest.mkdir(parents=True, exist_ok=True)
try:
    from huggingface_hub import snapshot_download
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="depth-anything/DA3METRIC-LARGE",
    local_dir=str(dest),
)
print("  [ok  ]", dest)
PY
fi

# --- SAM3 checkpoint (gated; needs HF_TOKEN with access to facebook/sam3) ---
SAM3_DEST="${MODELS_DIR}/sam3/sam3.pt"
mkdir -p "${MODELS_DIR}/sam3"
if [[ -f "${SAM3_DEST}" ]]; then
  echo "  [skip] sam3.pt already present"
elif [[ -n "${HF_TOKEN:-}" ]]; then
  echo "  [get ] sam3.pt (huggingface_hub facebook/sam3)…"
  DEST="${MODELS_DIR}/sam3" python3 - <<'PY' || echo "  [warn] SAM3 download failed — accept the license at https://huggingface.co/facebook/sam3"
from pathlib import Path
import os, sys, subprocess, shutil
dest = Path(os.environ["DEST"])
dest.mkdir(parents=True, exist_ok=True)
final = dest / "sam3.pt"
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
    from huggingface_hub import hf_hub_download
path = Path(hf_hub_download(
    repo_id="facebook/sam3",
    filename="sam3.pt",
    local_dir=str(dest),
    token=os.environ.get("HF_TOKEN") or None,
))
# huggingface_hub may place the file under a nested cache dir; normalize to models/sam3/sam3.pt
if path.resolve() != final.resolve():
    if final.exists() or final.is_symlink():
        final.unlink()
    try:
        path.replace(final)
    except OSError:
        shutil.copy2(path, final)
print("  [ok  ]", final, f"({final.stat().st_size} bytes)")
PY
else
  echo "  [skip] sam3.pt (set HF_TOKEN to download gated facebook/sam3)"
fi

echo
echo "Done. Place a video in inputs/ then:  bash run_pipeline.sh"
