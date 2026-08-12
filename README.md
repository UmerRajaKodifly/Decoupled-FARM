# Construction Spatial Memory Pipeline

**End-to-end:** equirectangular 360° walkthrough video (`.mp4`) → 3D open-vocabulary object map → interactive WebGL viewer.

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
5. [Run end-to-end on a video](#5-run-end-to-end-on-a-video)
6. [Open the 3D viewer](#6-open-the-3d-viewer)
7. [Validation & other visualizations](#7-validation--other-visualizations)
8. [Outputs reference](#8-outputs-reference)
9. [Partial re-runs & resumability](#9-partial-re-runs--resumability)
10. [Tunable environment variables](#10-tunable-environment-variables)
11. [Logging](#11-logging)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What this pipeline does

| Stage | Name | Role |
|-------|------|------|
| **Phase 1** | Stella VSLAM Dense | Monocular equirect SLAM → keyframes, poses, dense cloud (`out.db`, `out.ply`) |
| **Phase 1.5** | Depth Anything V3 (metric) | Per-keyframe cube faces + metric depth + `frames.json` |
| **Phase 2** | Detect / segment / embed | YOLOE open-vocab masks + 3D Gaussians + DINOv3 features |
| **Phase 3** | Associate / fuse / map | Multi-frame fusion → `scene_state.pt` (object memory) |
| **Phase 3.5** | Stella geometry | Refine object geometry from Stella dense cloud |
| **Phase 4a** | Best-view crops | Per-object RGB crops for inspection / captioning |
| **3D viewer** | WebGL (Three.js) | Full Stella cloud + colored object pts + crop billboards |

```text
inputs/scan.mp4
    │
    ▼ Phase 1 (Docker: stella)
outputs/runs/<RUN_ID>/phase1/  {out.db, keyframes/, out.ply}
    │
    ▼ Phase 1.5 (Docker: da3)
outputs/runs/<RUN_ID>/phase1.5/  {faces/, depth/, frames_json/}
    │
    ▼ Phases 2–4a (Docker: farm)
outputs/runs/<RUN_ID>/phase2/   detections_kf*.pt
outputs/runs/<RUN_ID>/phase3/   scene_state.pt
outputs/runs/<RUN_ID>/phase3.5/ scene_state_stella.pt
outputs/runs/<RUN_ID>/phase4/   crops/ + scene_state_with_crops.pt
    │
    ▼ 3D viewer data (auto-built inside farm container)
outputs/runs/<RUN_ID>/validation/3d-viewer/
    │
    ▼ serve.py (host)
http://127.0.0.1:8090
```

Each run is isolated under `outputs/runs/<RUN_ID>/`. A symlink `outputs/latest` always points to the most recent run. Older flat-layout outputs (`outputs/phase1/`, etc.) are archived automatically into `outputs/runs/legacy_<timestamp>/` on the first new run.

---

## 2. Prerequisites

### Hardware / OS

- Linux host with **NVIDIA GPU** drivers
- Enough disk for models + run data (Docker images alone can be tens of GB on first build)

### Software

| Tool | Notes |
|------|--------|
| **Docker** + **Docker Compose v2** | Phases 1–4a run in containers |
| **NVIDIA Container Toolkit** | `docker run --gpus all …` must see the GPU |
| **curl** or **wget** | Model bootstrap |
| **Python 3.10** + **conda** | Host validation + 3D viewer server |

Check GPU in Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Input video

- Equirectangular (2:1) 360° monocular video (e.g. Insta360-style)
- Place under `inputs/` as `.mp4` or `.mov`
- Default Stella config resizes to **1920×960** (`RESIZE=1920x960`)

---

## 3. Repository layout

```text
repo/
├── README.md
├── bootstrap_models.sh       ← download model weights → models/
├── run_pipeline.sh           ← full e2e orchestrator
├── docker-compose.yml
├── config/slam_config.yaml
├── vocab/construction_vocab.txt
├── inputs/                   ← put your .mp4 here
├── models/                   ← weights (gitignored; fill via bootstrap)
├── outputs/
│   ├── latest                → symlink to most recent run
│   └── runs/
│       └── run_YYYYMMDD_HHMMSS/
│           ├── phase1/ … phase4/
│           └── validation/
│               ├── phase2/ … phase4/
│               └── 3d-viewer/   ← WebGL viewer data
├── 3d-viewer/
│   ├── build_viewer_data.py  ← pack .pt + out.db → viewer assets
│   ├── serve.py              ← HTTP server for the viewer
│   └── static/index.html     ← Three.js viewer
├── viser-viewer/             ← legacy Viser viewer (optional)
├── phase1-slam/
├── phase1.5-depth/
├── phase2/
├── phase3/
├── phase3.5-stella-geometry/
├── phase4-caption-best-view/
├── farm_src/
├── common/
└── docker/
    ├── Dockerfile.farm
    └── farm_entrypoint.sh
```

---

## 4. One-time setup

All commands assume:

```bash
cd /home/kodifly/Desktop/farm-git/repo
```

### 4.1 Bootstrap model weights

```bash
bash bootstrap_models.sh
```

Installs under `models/`: ORB vocab, YOLOE, MobileCLIP, DINOv3, DA3 (optional offline cache).

Skip DA3 pre-download (fetches from Hugging Face on first Phase 1.5 run):

```bash
bash bootstrap_models.sh --skip-da3
```

### 4.2 Place input video

```bash
cp /path/to/your_scan.mp4 inputs/scan.mp4
```

### 4.3 Build Docker images (first time; long)

```bash
docker compose build
```

Or let `run_pipeline.sh` build automatically when images are missing. Force rebuild:

```bash
FORCE_BUILD=1 bash run_pipeline.sh
```

Rebuild a single stage after Dockerfile changes:

```bash
docker compose build farm    # after phase / viewer changes
docker compose build da3
docker compose build stella
```

### 4.4 Host conda env (validation + viewer)

The `farm-phase2` conda env is used for host-side validation and the 3D viewer server:

```bash
conda activate farm-phase2
python -c "import torch, matplotlib, plotly; print('ok')"
```

If you do not have that env:

```bash
pip install torch matplotlib plotly pillow numpy
```

---

## 5. Run end-to-end on a video

### 5.1 Full run (recommended)

```bash
cd /home/kodifly/Desktop/farm-git/repo

# place video first (if not already there)
cp /path/to/your_scan.mp4 inputs/scan.mp4

# run everything: Stella → DA3 → Phase 2 → 3 → 3.5 → 4a → viewer data build
VIDEO_FILE=scan.mp4 bash run_pipeline.sh
```

If `VIDEO_FILE` is omitted, the script auto-picks the first `.mp4` / `.mov` in `inputs/`.

**What happens:**

1. Creates `PIPELINE_RUN_ID=run_YYYYMMDD_HHMMSS`
2. Writes all outputs to `outputs/runs/<RUN_ID>/`
3. Updates `outputs/latest` symlink
4. Runs per-phase validation automatically
5. Builds 3D viewer data at `outputs/runs/<RUN_ID>/validation/3d-viewer/`

**Success criteria:**

- `outputs/runs/<RUN_ID>/phase1/out.db`
- `outputs/runs/<RUN_ID>/phase1.5/frames_json/frames.json`
- `outputs/runs/<RUN_ID>/phase2/detections_kf*.pt`
- `outputs/runs/<RUN_ID>/phase3/scene_state.pt`
- `outputs/runs/<RUN_ID>/phase3.5/scene_state_stella.pt`
- `outputs/runs/<RUN_ID>/phase4/crops/`
- `outputs/runs/<RUN_ID>/validation/3d-viewer/metadata.json`

### 5.2 Smoke test (faster)

```bash
VIDEO_FILE=scan.mp4 FRAME_STEP=4 MAX_KFS=20 WINDOW_SIZE=4 bash run_pipeline.sh
```

### 5.3 Skip stages already completed

```bash
# Re-run only farm stages (Phase 2–4a) on existing Stella + DA3 outputs
SKIP_STELLA=1 SKIP_DA3=1 bash run_pipeline.sh
```

---

## 6. Open the 3D viewer

After a successful pipeline run, viewer data is already built. Start the server:

```bash
cd /home/kodifly/Desktop/farm-git/repo
conda activate farm-phase2

# serves latest run automatically
python 3d-viewer/serve.py --data-dir outputs/latest/validation/3d-viewer
```

Or point at a specific run:

```bash
python 3d-viewer/serve.py \
  --data-dir outputs/runs/run_YYYYMMDD_HHMMSS/validation/3d-viewer \
  --port 8090
```

Then open **http://127.0.0.1:8090** in your browser (the server can auto-open it).

**Viewer features:**

- Full Stella background cloud (millions of pts, no downsampling in the viewer)
- Per-object colored point clouds from Phase 3.5
- Wireframe bounding boxes, floating class labels
- Crop billboards floating above each object
- Click object → detail panel with crop image; double-click → isolation mode
- Layer toggles: background / object pts / boxes / labels / crops
- Search/filter objects by label in the sidebar

**Rebuild viewer data manually** (e.g. after tweaking Phase 3.5 without re-running farm):

```bash
conda activate farm-phase2
python 3d-viewer/build_viewer_data.py \
  --stella-state outputs/latest/phase3.5/scene_state_stella.pt \
  --db-path      outputs/latest/phase1/out.db \
  --crops-dir    outputs/latest/phase4/crops \
  --vocab-file   vocab/construction_vocab.txt \
  --output-dir   outputs/latest/validation/3d-viewer
```

### Legacy Viser viewer (optional)

```bash
conda activate farm-phase2
python viser-viewer/run_viewer.py \
  --scene-state outputs/latest/phase4/scene_state_with_crops.pt \
  --crops-dir   outputs/latest/phase4/crops \
  --vocab       vocab/construction_vocab.txt \
  --cube-opacity 0.12
# open http://127.0.0.1:8080
```

---

## 7. Validation & other visualizations

Per-phase validation runs automatically during the pipeline. Key artifacts:

| Phase | Open / inspect |
|-------|----------------|
| Phase 3 | `outputs/latest/validation/phase3/overlays_3d.html` |
| Phase 3.5 | `outputs/latest/validation/phase3.5/summary.txt` |
| Phase 4 | `outputs/latest/validation/phase4/crop_grid.html` |
| Stella cloud QA | `outputs/latest/validation/stella_cloud/topdown.png` |
| Stella cloud QA | `outputs/latest/validation/stella_cloud/labeled_cloud.ply` (CloudCompare) |

Export labeled Stella cloud manually:

```bash
conda activate farm-phase2
python phase3.5-stella-geometry/export_labeled_cloud.py \
  --stella-state outputs/latest/phase3.5/scene_state_stella.pt \
  --db-path      outputs/latest/phase1/out.db \
  --output-dir   outputs/latest/validation/stella_cloud \
  --isolate-objects
```

Re-run host validators manually:

```bash
conda activate farm-phase2
export PYTHONPATH="$(pwd)/common:$(pwd)/farm_src/src:${PYTHONPATH:-}"

python phase2/validate_phase2.py \
  --det-dir outputs/latest/phase2 \
  --out-dir outputs/latest/validation/phase2

python phase3/validate_phase3.py \
  --output-dir outputs/latest/phase3 \
  --vocab-file vocab/construction_vocab.txt
```

---

## 8. Outputs reference

| Path | Contents |
|------|----------|
| `outputs/runs/<RUN_ID>/phase1/out.db` | Stella map DB + dense points |
| `outputs/runs/<RUN_ID>/phase1/out.ply` | Stella dense point cloud |
| `outputs/runs/<RUN_ID>/phase1.5/frames_json/frames.json` | Phase 2 input manifest |
| `outputs/runs/<RUN_ID>/phase2/detections_kf*.pt` | Per-keyframe detection packs |
| `outputs/runs/<RUN_ID>/phase3/scene_state.pt` | Fused object memory |
| `outputs/runs/<RUN_ID>/phase3.5/scene_state_stella.pt` | Geometry-refined memory |
| `outputs/runs/<RUN_ID>/phase4/scene_state_with_crops.pt` | Memory + best-view crop paths |
| `outputs/runs/<RUN_ID>/phase4/crops/` | Per-object RGB crop JPEGs |
| `outputs/runs/<RUN_ID>/validation/3d-viewer/` | WebGL viewer data (`metadata.json`, `objects.json`, `bg_cloud.bin`) |
| `outputs/logs/run_*/` | Per-run stage logs + elapsed timers |

---

## 9. Partial re-runs & resumability

```bash
cd /home/kodifly/Desktop/farm-git/repo
export VIDEO_FILE=scan.mp4
```

| Goal | Command |
|------|---------|
| Re-run Phase 1.5 only | `docker compose run --rm da3` |
| Re-run Phase 2–4a only | `SKIP_STELLA=1 SKIP_DA3=1 bash run_pipeline.sh` |
| Re-run full stack | `VIDEO_FILE=scan.mp4 bash run_pipeline.sh` |
| Compare runs | `ls outputs/runs/` |

Phase 2 skips existing `detections_kf*.pt` files. Delete them to force re-detect:

```bash
sudo rm -f outputs/runs/<RUN_ID>/phase2/detections_kf*.pt
```

---

## 10. Tunable environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `VIDEO_FILE` | first `inputs/*.{mp4,mov}` | Video filename under `inputs/` |
| `RESIZE` | `1920x960` | Stella input size |
| `FRAME_STEP` | `2` | Use every Nth video frame in Stella |
| `WINDOW_SIZE` | `4` | DA3 temporal window size |
| `MAX_KFS` | *(all)* | Cap keyframes in Phase 1.5 |
| `YOLOE_CONF` | `0.35` | Detection confidence threshold |
| `KF_STRIDE` | `1` | Phase 2: every Nth keyframe |
| `DEVICE` | `cuda` | Phase 2/3 torch device |
| `PIPELINE_RUN_ID` | auto timestamp | Run namespace under `outputs/runs/` |
| `FORCE_BUILD` | `0` | Set `1` to rebuild Docker images |
| `SKIP_STELLA` | `0` | Set `1` to skip Phase 1 |
| `SKIP_DA3` | `0` | Set `1` to skip Phase 1.5 |
| `SKIP_FARM` | `0` | Set `1` to skip Phases 2–4a |
| `STRICT_VALIDATE` | `0` | Set `1` to abort on validation failure |

Example:

```bash
VIDEO_FILE=scan.mp4 YOLOE_CONF=0.35 KF_STRIDE=2 bash run_pipeline.sh
```

---

## 11. Logging

Each run creates:

```text
outputs/logs/run_YYYYMMDD_HHMMSS/
├── pipeline.log
├── phase1-slam.log
├── phase1.5-depth.log
├── phase2.log
├── phase3.log
├── phase3.5.log
├── phase4a.log
└── *.elapsed
```

Inspect last run:

```bash
less outputs/logs/run_*/pipeline.log
```

---

## 12. Troubleshooting

| Symptom | Likely fix |
|---------|------------|
| `ERROR: no video found` | Put a file in `inputs/` or set `VIDEO_FILE=scan.mp4` |
| `missing models/...` | `bash bootstrap_models.sh` |
| GPU not visible | Install NVIDIA Container Toolkit; verify with `nvidia-smi` inside Docker |
| Viewer shows blank / 404 crops | Rebuild viewer data: `python 3d-viewer/build_viewer_data.py --output-dir outputs/latest/validation/3d-viewer` |
| Scene appears upside-down in viewer | Hard-refresh browser (`Ctrl+Shift+R`); Y-flip is applied at load time |
| `Permission denied` on outputs | Docker writes as root; use `outputs/runs/<RUN_ID>/` paths or `sudo` for cleanup |
| Farm image rebuild needed after code changes | `docker compose build farm` then re-run |

---

## Quick reference

```bash
cd /home/kodifly/Desktop/farm-git/repo

# ── once ──
bash bootstrap_models.sh
cp /path/to/scan.mp4 inputs/scan.mp4
docker compose build          # optional; run_pipeline builds if missing
conda activate farm-phase2

# ── end-to-end on a video ──
VIDEO_FILE=scan.mp4 bash run_pipeline.sh

# ── open 3D viewer ──
python 3d-viewer/serve.py --data-dir outputs/latest/validation/3d-viewer
# → http://127.0.0.1:8090
```

---

## License note

Third-party code (Stella, Depth Anything 3, FARM / YOLOE / DINOv3) retains **each upstream project's license**. Weight downloads have their own terms. Review upstream notices before commercial use.
