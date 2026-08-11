# Construction Spatial Memory Pipeline

**End-to-end:** equirectangular 360° walkthrough video (`.mp4`) → 3D open-vocabulary object map (`scene_state.pt`) + HTML / image visualizations.

**Repository root:**

```text
/home/kodifly/Desktop/farm-git/repo
```

---

## Table of contents

1. [What this pipeline does](#1-what-this-pipeline-does)
2. [Prerequisites](#2-prerequisites)
3. [Repository layout](#3-repository-layout)
4. [One-time setup](#4-one-time-setup)
5. [Run the full pipeline](#5-run-the-full-pipeline)
6. [Validation & point-cloud visualization](#6-validation--point-cloud-visualization)
7. [Outputs reference](#7-outputs-reference)
8. [Partial re-runs & resumability](#8-partial-re-runs--resumability)
9. [Tunable environment variables](#9-tunable-environment-variables)
10. [Logging](#10-logging)
11. [Troubleshooting](#11-troubleshooting)
12. [What this repo intentionally excludes](#12-what-this-repo-intentionally-excludes)

---

## 1. What this pipeline does

| Stage | Name | Role |
|-------|------|------|
| **Phase 1** | Stella VSLAM Dense | Monocular equirect SLAM → keyframes, poses, sparse landmarks (`out.db`, trajectories) |
| **Phase 1.5** | Depth Anything V3 (metric) | Per-keyframe cube faces + **metric** depth + FARM `frames.json` |
| **Phase 2** | Detect / segment / embed | YOLOE open-vocab masks + 3D Gaussians + DINOv3 features per face |
| **Phase 3** | Associate / fuse / map | Filter → neighbors → union-find → fused `scene_state.pt` (object memory) |
| **Validate** | Host-side scripts | Metrics, top-down scatter PNGs, interactive `overlays_3d.html` |

Captioning, pruning, and language query retrieval are **not** part of this tree (Phase 4 / 5 later).

```text
inputs/video.mp4
    │
    ▼ Phase 1 (Docker: stella)
outputs/phase1/  {out.db, keyframes/, traj/, out.ply}
    │
    ▼ Phase 1.5 (Docker: da3)
outputs/phase1.5/  {faces/, depth/, frames_json/}
    │
    ▼ Phase 2 + 3 (Docker: farm)
outputs/phase2/  detections_kf*.pt
outputs/phase3/  scene_state.pt
    │
    ▼ validate_phase2 / validate_phase3 (host conda)
outputs/validation/phase2/overlays_3d.html
outputs/validation/phase3/overlays_3d.html
```

---

## 2. Prerequisites

### Hardware / OS

- Linux host with **NVIDIA GPU** drivers installed
- Enough disk for models + run data (expect **tens of GB** for Docker images alone on first build; Stella image is large and compiles C++)

### Software

| Tool | Notes |
|------|--------|
| **Docker** + **Docker Compose v2** | Required for Phases 1–3 |
| **NVIDIA Container Toolkit** | `docker run --gpus all …` must see the GPU |
| **curl** or **wget** | Model bootstrap |
| **Python 3.10** + **conda** (optional but recommended) | Host validation / HTML viz only |

Check GPU in Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Input video

- Equirectangular (2:1) 360° monocular video (e.g. Insta360-style)
- Dropped under `inputs/` as `.mp4` or `.mov`

Default Stella config assumes the feed is resized to **1920×960** (`RESIZE=1920x960`) to match [`config/slam_config.yaml`](config/slam_config.yaml) (fast `dense_batch_1920` recipe).

---

## 3. Repository layout

```text
repo/
├── README.md                 ← this file
├── bootstrap_models.sh       ← download / copy model weights → models/
├── run_pipeline.sh           ← full e2e: stella → da3 → farm
├── docker-compose.yml        ← three GPU services
├── config/
│   └── slam_config.yaml      ← Stella SLAM + PatchMatch (1920×960 equirect)
├── vocab/
│   └── construction_vocab.txt
├── inputs/                   ← put your .mp4 here (gitignored contents)
├── models/                   ← weights (gitignored; fill via bootstrap)
├── outputs/
│   ├── phase1/               ← SLAM
│   ├── phase1.5/             ← DA3 depths + frames.json
│   ├── phase2/               ← detection packs
│   ├── phase3/               ← scene_state.pt
│   ├── validation/           ← HTML / PNG diagnostics (host)
│   └── logs/run_*/           ← per-run logs
├── phase1-slam/              ← Stella source + Dockerfile
├── phase1.5-depth/           ← DA3 source + stella_to_da3_depth.py
├── phase2/                   ← detect-segment-embed
├── phase3/                   ← associate-fuse-map
├── farm_src/                 ← FARM scene_graph + YOLOE (git stripped)
├── common/                   ← path helpers + pipeline logger
└── docker/
    ├── Dockerfile.farm
    └── farm_entrypoint.sh
```

Vendored third-party trees are **not** git submodules: nested `.git` / `.github` / pre-commit configs are stripped so pushing **this** folder alone does not re-trigger upstream CI.

---

## 4. One-time setup

All commands below assume:

```bash
cd /home/kodifly/Desktop/farm-git/repo
```

### 4.1 Bootstrap model weights

```bash
bash bootstrap_models.sh
```

Installs under `models/`:

| Path | Purpose |
|------|---------|
| `models/orb_vocab.fbow` | Stella ORB vocabulary |
| `models/yoloe/yoloe-v8l-seg.pt` | YOLOE segmentation |
| `models/yoloe/yoloe-v8l-seg-pf.pt` | YOLOE prompt-free |
| `models/mobileclip/mobileclip_blt.pt` | MobileCLIP text tower |
| `models/dinov3-vits16/` | DINOv3 appearance features |
| `models/da3metric-large/` | DA3METRIC-LARGE (optional offline cache) |

Skip DA3 pre-download (will fetch at first Phase 1.5 run via Hugging Face):

```bash
bash bootstrap_models.sh --skip-da3
```

### 4.2 Place input video

```bash
cp /path/to/your_scan.mp4 inputs/
# example used in testing:
# cp /home/kodifly/Desktop/stella-vslam-dense/inputs/scan.MP4 inputs/scan.mp4
```

Or set an explicit name later: `VIDEO_FILE=scan.mp4`.

### 4.3 Build Docker images (first time; long)

```bash
# Builds stella + da3 + farm when images are missing
FORCE_BUILD=1 bash run_pipeline.sh
# … or build without running:
docker compose build
```

Rebuild a single stage after Dockerfile changes:

```bash
docker compose build da3
docker compose build --no-cache da3   # if torch/deps look wrong
docker compose build farm
docker compose build stella
```

### 4.4 Host env for validation (HTML / plots)

Phase 2/3 **validation** is intended to run on the host (Docker often writes phase outputs as `root`, which breaks writing into those dirs).

Using the existing `farm-phase2` conda env (same deps as the original `pipeline/` work):

```bash
conda activate farm-phase2
# needs: torch, matplotlib, plotly, pillow, numpy
python -c "import torch, matplotlib, plotly; print('ok')"
```

If you do not have that env, use any Python 3.10 env with:

```bash
pip install torch matplotlib plotly pillow numpy
```

(Plotly is required for `overlays_3d.html`.)

---

## 5. Run the full pipeline

### 5.1 Standard full run

```bash
cd /home/kodifly/Desktop/farm-git/repo

# optional explicit video name
export VIDEO_FILE=scan.mp4

bash run_pipeline.sh
```

This will:

1. Resolve `VIDEO_FILE` (or auto-pick the first `.mp4` / `.mov` in `inputs/`)
2. Create `PIPELINE_RUN_ID=run_YYYYMMDD_HHMMSS`
3. Ensure models exist
4. Build images if needed
5. Run **stella → da3 → farm** sequentially (stops on first failure)

**Success criteria:**

- `outputs/phase1/out.db` and `keyframes/` present  
- `outputs/phase1.5/frames_json/frames.json` present  
- `outputs/phase2/detections_kf*.pt` present  
- **`outputs/phase3/scene_state.pt` present** ← main product  

### 5.2 Smoke test (faster)

Limit Phase 1.5 keyframes and skip more video frames:

```bash
export VIDEO_FILE=scan.mp4
export FRAME_STEP=4
export MAX_KFS=20
export WINDOW_SIZE=4
bash run_pipeline.sh
```

### 5.3 Force rebuild then run

```bash
FORCE_BUILD=1 bash run_pipeline.sh
```

---

## 6. Validation & point-cloud visualization

The Docker farm service does **not** emit HTML. After Phases 2 and 3 succeed, run the host validators. Write into **`outputs/validation/`** (user-owned) to avoid `PermissionError` on root-owned Docker volume dirs.

### 6.1 Generate Phase 2 diagnostics (per-detection Gaussians)

```bash
cd /home/kodifly/Desktop/farm-git/repo
conda activate farm-phase2

mkdir -p outputs/validation/phase2

python phase2/validate_phase2.py \
  --det-dir outputs/phase2 \
  --out-dir outputs/validation/phase2
```

### 6.2 Generate Phase 3 diagnostics (fused object map)

```bash
cd /home/kodifly/Desktop/farm-git/repo
conda activate farm-phase2

export PYTHONPATH="$(pwd)/common:$(pwd)/farm_src/src:${PYTHONPATH:-}"
mkdir -p outputs/validation

# Validators expect scene_state.pt under --output-dir; Docker phase3 may be root-owned
ln -sfn "$(pwd)/outputs/phase3/scene_state.pt" outputs/validation/scene_state.pt
ln -sfn "$(pwd)/outputs/phase3/run_stats.json" outputs/validation/run_stats.json

python phase3/validate_phase3.py \
  --output-dir outputs/validation \
  --vocab-file vocab/construction_vocab.txt

# validate_phase3 writes under <output-dir>/validation/
mkdir -p outputs/validation/phase3
mv -f outputs/validation/validation/* outputs/validation/phase3/ 2>/dev/null || true
rmdir outputs/validation/validation 2>/dev/null || true
```

### 6.3 Open the interactive point clouds

**Phase 2 — all per-detection 3D Gaussians:**

```bash
xdg-open /home/kodifly/Desktop/farm-git/repo/outputs/validation/phase2/overlays_3d.html
```

Also useful:

```bash
xdg-open /home/kodifly/Desktop/farm-git/repo/outputs/validation/phase2/world_xy_scatter.png
```

**Phase 3 — fused objects (final map):**

```bash
xdg-open /home/kodifly/Desktop/farm-git/repo/outputs/validation/phase3/overlays_3d.html
```

Also useful:

```bash
xdg-open /home/kodifly/Desktop/farm-git/repo/outputs/validation/phase3/world_xy_scatter.png
cat /home/kodifly/Desktop/farm-git/repo/outputs/validation/phase3/summary.txt
```

If `xdg-open` is unavailable, open the HTML paths in Chrome / Firefox manually.

### 6.4 What “good” looks like

| Artifact | Healthy signal |
|----------|----------------|
| Phase 2 `world_xy_scatter.png` | Detections form a coherent site footprint (metres-scale, not a blob at origin) |
| Phase 2 `overlays_3d.html` | Hover labels look construction-plausible; outliers exist but are not chaos |
| Phase 3 `object_count_growth.png` | Object count grows then plateaus |
| Phase 3 `summary.txt` | `Overall: PASS`; active objects ≪ total Phase 2 detections (merging worked) |
| Phase 3 footprint | On the order of site size (e.g. tens of metres), not kilometres |

---

## 7. Outputs reference

| Path | Contents |
|------|----------|
| `outputs/phase1/out.db` | Stella map DB (keyframes, landmarks) |
| `outputs/phase1/keyframes/` | KF RGB (`image*.png`) |
| `outputs/phase1/traj/keyframe_trajectory.txt` | TUM keyframe poses |
| `outputs/phase1/out.ply` | Stella dense point cloud |
| `outputs/phase1.5/faces/` | Cube-face JPEGs |
| `outputs/phase1.5/depth/` | Face depths (float32 metres `.npy`) |
| `outputs/phase1.5/frames_json/frames.json` | Phase 2 input manifest |
| `outputs/phase2/detections_kf*.pt` | Per-keyframe packs (means, cov6, features, masks, …) |
| `outputs/phase3/scene_state.pt` | **Global object memory** |
| `outputs/phase3/run_stats.json` | Per-keyframe merge counters |
| `outputs/validation/phase2/` | Histograms, `overlays_3d.html`, `summary.txt` |
| `outputs/validation/phase3/` | Growth plots, `overlays_3d.html`, `metrics.json`, `summary.txt` |
| `outputs/logs/run_*/` | Stage logs + `pipeline.log` + elapsed timers |

---

## 8. Partial re-runs & resumability

Phase 1 is the slowest. After a successful Stella run you can resume later stages only.

```bash
cd /home/kodifly/Desktop/farm-git/repo
export PIPELINE_RUN_ID=run_YYYYMMDD_HHMMSS   # optional; for log grouping
export VIDEO_FILE=scan.mp4                     # only needed for stella
```

| Goal | Command |
|------|---------|
| Re-run Phase 1.5 only | `docker compose run --rm da3` |
| Re-run Phase 2+3 only | `docker compose run --rm farm` |
| Re-run full stack | `bash run_pipeline.sh` |
| Rebuild DA3 image after dep fix | `docker compose build da3` then `docker compose run --rm da3` |

Phase 2 skips existing `detections_kf*.pt` files (resume-friendly). Delete them to force re-detect:

```bash
# careful: root-owned files may need sudo
sudo rm -f outputs/phase2/detections_kf*.pt
```

---

## 9. Tunable environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `VIDEO_FILE` | first `inputs/*.{mp4,mov}` | Video filename under `inputs/` |
| `RESIZE` | `1920x960` | Stella input size (must match slam config) |
| `FRAME_STEP` | `2` | Use every Nth video frame in Stella |
| `WINDOW_SIZE` | `4` | DA3 temporal window size |
| `MAX_KFS` | *(all)* | Cap keyframes in Phase 1.5 (smoke tests) |
| `DA3_MODEL` | HF id or `/models/da3metric-large` | DA3 weights path / hub id |
| `YOLOE_CONF` | `0.35` | Detection confidence threshold |
| `KF_STRIDE` | `1` | Phase 2: every Nth keyframe |
| `DEVICE` | `cuda` | Phase 2/3 torch device |
| `PIPELINE_RUN_ID` | auto timestamp | Log namespace under `outputs/logs/` |
| `FORCE_BUILD` | `0` | Set `1` to rebuild images before run |

Example:

```bash
VIDEO_FILE=scan.mp4 FRAME_STEP=2 YOLOE_CONF=0.35 bash run_pipeline.sh
```

---

## 10. Logging

Each full run creates:

```text
outputs/logs/run_YYYYMMDD_HHMMSS/
├── pipeline.log          # merged timeline + final summary
├── phase1-slam.log
├── phase1.5-depth.log
├── phase2.log
├── phase3.log
└── *.elapsed             # stage wall times (seconds)
```

Lines include ISO-ish timestamps and stage tags. Phase 1.5 / 2 / 3 emit **tqdm** progress bars on the console (also captured in stage logs).

---

## 11. Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| `ERROR: no video found` in Stella | Put a file in `inputs/`; `VIDEO_FILE` bare name is rewritten to `/inputs/<name>` after path fix |
| `missing models/...` | `bash bootstrap_models.sh` |
| `ModuleNotFoundError: moviepy` / `addict` in DA3 | Rebuild: `docker compose build da3` |
| `operator torchvision::nms does not exist` | Torch / torchvision mismatch — rebuild with `--no-cache`: `docker compose build --no-cache da3` |
| Pipeline dies after Stella on DA3 | Stella outputs intact; only re-run `docker compose run --rm da3` |
| `Permission denied` writing into `outputs/phase2` | Root-owned Docker volumes — write validation to `outputs/validation/…` |
| Farm image huge (~14 GB) | Expected with CUDA **devel** base + torch; slim later if desired; does not block correctness |
| GPU not visible | Install NVIDIA Container Toolkit; verify with `nvidia-smi` inside `docker run --gpus all` |

Inspect last run:

```bash
ls -lt outputs/logs | head
less outputs/logs/run_*/pipeline.log
```

---

## 12. What this repo intentionally excludes

- **No DAP** (relative / inconsistent depth) — metric path uses **DA3 only**
- **No FARM ROS / vLLM / captioning servers** in the farm image
- **No automated Phase 4 captions or Phase 5 retrieval** yet
- **No upstream git CI hooks** inside vendored Stella / DA3 / FARM copies

---

## Quick reference card

```bash
cd /home/kodifly/Desktop/farm-git/repo

# ── once ──
bash bootstrap_models.sh
cp /path/to/scan.mp4 inputs/
docker compose build          # optional; run_pipeline builds if missing

# ── map ──
VIDEO_FILE=scan.mp4 bash run_pipeline.sh

# ── viz ──
conda activate farm-phase2
python phase2/validate_phase2.py --det-dir outputs/phase2 --out-dir outputs/validation/phase2

export PYTHONPATH="$(pwd)/common:$(pwd)/farm_src/src"
ln -sfn "$(pwd)/outputs/phase3/scene_state.pt" outputs/validation/scene_state.pt
ln -sfn "$(pwd)/outputs/phase3/run_stats.json" outputs/validation/run_stats.json
python phase3/validate_phase3.py --output-dir outputs/validation --vocab-file vocab/construction_vocab.txt
mkdir -p outputs/validation/phase3 && mv -f outputs/validation/validation/* outputs/validation/phase3/

xdg-open outputs/validation/phase2/overlays_3d.html
xdg-open outputs/validation/phase3/overlays_3d.html
```

---

## License note

Third-party code (Stella, Depth Anything 3, FARM / YOLOE / DINOv3) retains **each upstream project’s license**. Weight downloads have their own terms. Review upstream notices before commercial use.
