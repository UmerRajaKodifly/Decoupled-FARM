# Decoupled FARM

Jira: [SX-3624](https://kodiflylimited.atlassian.net/browse/SX-3624) (parent) · [SX-3625](https://kodiflylimited.atlassian.net/browse/SX-3625) (Phase 1: Stream, Segment, Embed)

Repo: [github.com/UmerRajaKodifly/Decoupled-FARM](https://github.com/UmerRajaKodifly/Decoupled-FARM)

Every commit subject starts with `SX-3624` so work shows on the overarching ticket. Template: `.gitmessage`.

FARM’s **per-frame object-mapping core** on monocular (often 360°) video:

1. Equirect → ss-3dgs **cubemap faces** (pinhole)
2. COLMAP SfM (Caspar) on the rig
3. YOLOE masks + **FARM** Hellinger/DINO/union-find (exact source wrap)
4. Depth under mask → unproject → world Gaussians + sparse voxels

No 3DGS training. No global environment cloud in the mapping loop.

This repository is self-contained: vocab lives in `data/`, and upstream FARM /
ss-3dgs are cloned under `third_party/` (not sibling folders on a specific machine).

## Docs

- [Back-projection math vs FARM](docs/BACKPROJECTION.md)
- [FARM vs greedy IoU counts](docs/FARM_VS_IOU.md)
- [Prior R&D notes](docs/prior/)

## Setup

```bash
git clone https://github.com/UmerRajaKodifly/Decoupled-FARM.git
cd Decoupled-FARM
./scripts/bootstrap_third_party.sh
# then FARM weights:
( cd third_party/FARM-Project && ./bootstrap_models.sh )
pip install -e third_party/FARM-Project/third_party/yoloe
pip install -e third_party/FARM-Project/third_party/yoloe/third_party/ml-mobileclip
source env.sh
```

Override locations if you already have checkouts:

```bash
export FARM_PROJECT_ROOT=/path/to/FARM-Project
export SS3DGS_ROOT=/path/to/ss-3dgs
export SCENE_GRAPH_MODEL_DIR=/path/to/FARM-Project/models
```

Optional: `COLMAP_ROOT` pointing at a 4.1+Caspar install (otherwise whatever
`colmap` is on `PATH`).

## Depth sources

Default / primary is **`dl`** (`dl_depth_v1`). COLMAP MVS is an **explicit
alternate** comparison flow only — it is never the default.

```bash
# primary (fails closed if DL is missing)
python -m farm_object_map e2e --video clip.mp4 --work work/run_dl --depth-source dl

# alternate comparison flow (same downstream mapping)
python -m farm_object_map e2e \
  --video clip.mp4 \
  --outputs-root outputs \
  --depth-source colmap_mvs \
  --gpu-index 0
# writes outputs/<video_stem>/colmap_mvs/  (and .../dl/ for the primary flow)
```

MVS writes `units: "sfm"` DepthMap npz under `depth_npz/`. DL stays `units: "m"`.

## Depth drop-in (`dl_depth_v1`)

Not deployed yet. Mapping **stops** rather than falling back to MVS unless you
pass `--depth-source colmap_mvs`.

When ready, any one of:

```python
from farm_object_map.dl_depth_v1 import register_infer_fn

def infer(rgb_bgr, frame_name):
    depth_m, valid = ...  # metres, same HxW as rgb_bgr (cubemap face)
    return depth_m, valid

register_infer_fn(infer)
```

```bash
export FARM_DL_DEPTH_INFER=my_pkg.depth:infer
# or precomputed maps:
python -m farm_object_map map-objects ... --dl-depth-npz-dir work/dl_depth/
```

Then run `align_poses_to_metric_depth` (COLMAP is up-to-scale; DL depth is metric).

Contract: `src/farm_object_map/depth.py` (`DepthMap` npz).

## 360 / cubemap

Default e2e path is `--image-type panorama` (`cubemap-nosfm-top-and-bottom`).
FARM unprojection runs only on the resulting pinhole faces — see the math doc.

```bash
python -m farm_object_map e2e \
  --video /path/to/export_video.mp4 \
  --work work/run1 \
  --image-type panorama \
  --fps 2
```

FPS default is **2.0** (`configs/ss3dgs_sfm_only.yaml`).

## Visualize results

```bash
source env.sh
python -m farm_object_map view-objects \
  --objects-dir outputs/export_video_2/colmap_mvs/objects \
  --port 8080
```

Open http://127.0.0.1:8080 — voxel clouds + Gaussian means/ellipsoids, colored by object.

2D masks/thumbnails (if present): `objects/image_store/` and `objects/image_store_masks/`.

FARM’s own viewer needs `scene_state.pt` (saved on newer mapping runs):

```bash
python third_party/FARM-Project/scripts/view_scene_state.py \
  --pt outputs/export_video_2/colmap_mvs/objects/scene_state.pt
```

## Association

- `--association farm` (default): FARM modules, DINO via `resolve_dino_backbone()`
- `--association greedy_iou`: 2D same-class IoU ≥ 0.3 baseline

Vocab: `data/construction_site_object_vocabulary.json` via adapter
(`src/farm_object_map/vocab.py`), not hand-edited.

## YOLOE

FARM-vendored fork, not PyPI 8.4.x:

- `third_party/FARM-Project/third_party/yoloe` @ `7ed2b05` → ultralytics **8.3.39**
- + `ml-mobileclip`, weights from that repo’s `bootstrap_models.sh`
