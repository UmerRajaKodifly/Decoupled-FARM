# Phase 2 — Detect / Segment / Embed

**Role in the pipeline:** Consumes posed RGB-D face tiles (from Phase 1 outputs) and produces one detection-pack per keyframe: 2D instance masks, per-detection 3D Gaussians in the world frame, and DINOv3 appearance features. This is the direct FARM-adaptation step that replaces FARM's `segment_and_transform` for 360 cubemap imagery.

---

## Position in the full pipeline

```
Phase 1 (inputs)          Phase 2 (this)              Phase 3 (next)
─────────────────         ──────────────────────       ─────────────────────────────
DA3 depth (.npy)    ─┐    Detect (YOLOE)               Neighbor lookup (Hellinger)
Stella poses       ─┤ →   Segment (masks)         →    Correspondence (union-find)
Face JPEGs         ─┘    Embed (DINOv3)                Map update (Gaussian fusion)
                          Unproject → Gaussian          Co-visibility graph
                          World transform (T_wc)        scene_state.pt
```

Phase 2 does **not** maintain state across keyframes. It processes each keyframe independently and writes a self-contained `.pt` file. Phase 3 reads these files in order and builds the persistent scene graph.

---

## Data flow inside Phase 2

```
frames.json
    │
    ▼ FrameLoader (phase2_runner.py)
    │   read rgb → (H,W,3) uint8
    │   read depth.npy → (H,W) float32 metres
    │   parse K → (3,3) pinhole intrinsics
    │   parse T_wc → (4,4) camera-to-world (Stella pose)
    │
    ▼ ConstructionSegmenter (segmenter.py → FARM YOLOESegmenter)
    │   YOLOE-v8l-seg-pf.pt  ← prompt-free model with fused vocab
    │   MobileCLIP text embeddings of construction_vocab.txt (one-time at init)
    │   NMS + confidence filter
    │   → 2D masks (H,W bool per detection), class ids, scores
    │
    ├── Depth unproject (inside YOLOESegmenter, using geometry.py logic)
    │       mask ∩ valid depth → 3D points (camera frame)
    │       depth-mode MAD filter  (remove bg depth leak)
    │       Mahalanobis outlier reject
    │       weighted mean + cov6  →  Gaussian (camera frame)
    │
    └── DINOv3 feature extraction (dino_extractor.py / FARM DINOFeaturesExtractor)
            ViT-S/16 forward on RGB
            dense patch-token grid  →  mask-pool under detection mask
            L2-normalise  →  384-D feature per detection
    │
    ▼ World transform (_apply_world_transform, phase2_runner.py)
        apply T_wc per detection (via batch_ids)
        means: camera frame  →  world frame
        cov6:  rotate by R from T_wc
    │
    ▼ Save detections_kfNNNNNN.pt
```

---

## Inputs

### `frames.json` (DA3 pipeline output)

| Field | Type | Notes |
|---|---|---|
| `rgb` | `str` | Relative path to 504×504 JPEG face image |
| `depth` | `str` | Relative path to 504×504 `float32` `.npy` depth map, **metric metres** |
| `K` | `[[fx,0,cx],[0,fy,cy],[0,0,1]]` | Pinhole intrinsics (same for all faces, `fx=fy=252`, `cx=cy=252`) |
| `T_wc` | 4×4 float list | Camera-to-world transform from Stella VSLAM |
| `depth_encoding` | `"float32_m"` | Must be `float32_m`; depths are already metric |
| `camera` | `"face"` | Camera id |

**Structure:** 1280 entries = 320 keyframes × 4 faces (face0, face1, face2, face3). Each face is a 90° perspective crop of the equirectangular frame reprojected to a pinhole view.

### Model weights (from `FARM-Project/models/`)

| File | Purpose |
|---|---|
| `yoloe/yoloe-v8l-seg.pt` | YOLOE base checkpoint — used once at init to extract MobileCLIP vocab embeddings |
| `yoloe/yoloe-v8l-seg-pf.pt` | Prompt-free checkpoint — actual inference model with vocab fused in |
| `dinov3-vits16/` | DINOv3 ViT-S/16 local weights (bundled in FARM repo) |
| `mobileclip/mobileclip_blt.pt` | MobileCLIP encoder used inside YOLOE for text→embedding |

Run `bootstrap_models.sh` from `FARM-Project/` if weights are missing.

### Vocabulary

`vocab/construction_vocab.txt` — ~75 construction-site labels (one per line; `#` lines are comments).

Categories:
- **PPE & people:** person, worker, helmet, safety vest
- **Plant & vehicles:** excavator, crane, forklift, truck, loader, bulldozer, roller, generator, pump, compressor, lift, tow/trailer/fire truck
- **Site structures:** scaffold, wall, pillar, column, beam, floor, ceiling, staircase, stairway
- **Raw materials:** sand, gravel, brick, concrete, pile
- **Packaged materials & storage:** bag, barrel, drum, crate, pallet, container, box, toolbox, cable, wire, pipe, conduit
- **Small tools:** ladder, shovel, wrench, tool, wheelbarrow, rope, chain
- **Boundaries & signs:** fence, gate, barrier, barricade, cone, sign
- **Waste:** dumpster, garbage

Modify to add/remove labels; re-initialise the segmenter to pick up changes.

### Hardware

- NVIDIA GPU with CUDA 12.8+ drivers (same as FARM)
- ~6–8 GB VRAM for YOLOE-v8l + DINOv3 at batch-size 4
- DA3 and FARM dependencies installed (see `FARM-Project/pyproject.toml`)

---

## Process

### Step-by-step per keyframe

1. **Load faces** — read 4 face JPEGs + 4 depth `.npy` files + `K` + `T_wc` from `frames.json`.

2. **YOLOE open-vocab detect+segment** — forward all 4 faces through YOLOE-v8l-seg-pf in a single batched call. The model detects instances from `construction_vocab.txt`, produces 2D bounding boxes and **instance masks**, filtered by confidence and NMS. No text encoder runs at inference — vocabulary was fused at init via MobileCLIP.

3. **3D Gaussian construction** (inside YOLOESegmenter, using `geometry.py`):
   - Erode each mask by 3 px to avoid depth-boundary leakage.
   - Unproject valid depth pixels under the mask using pinhole `K`: `X=(u-cx)*Z/fx`, `Y=(v-cy)*Z/fy`.
   - Apply **depth-mode MAD filter**: remove pixels whose depth is > 3 MADs from the mask-median depth. This prevents background pixels from contaminating the Gaussian when the object is near a depth discontinuity.
   - Apply **Mahalanobis outlier reject** (thresh=2.0): remove pixels > 2σ from the current fit.
   - Compute **weighted mean + cov6** (6-element packed 3×3 covariance) = the 3D Gaussian.
   - Median-robustify the mean.

4. **DINOv3 feature extraction** (`dino_extractor.py`):
   - Forward DINOv3 ViT-S/16 on the face RGB to get a dense patch-token grid.
   - For each detection, average the patch tokens under the resized mask → 384-D vector.
   - L2-normalise → `features (M, 384)`.
   - These features are the primary signal for Phase 3's neighbor lookup (cosine similarity).

5. **World transform**: apply `T_wc` from `frames.json` to rotate/translate means and covariances from camera frame into the shared Stella world frame. All 4 faces of a keyframe share the same Stella pose (the keyframe pose), so their detections are placed in a common coordinate system.

6. **Save** — write `detections_kfNNNNNN.pt` containing the detection pack.

### Key design decisions matching FARM

| Decision | FARM | Phase 2 |
|---|---|---|
| Vocabulary encoding | MobileCLIP text → fused in prompt-free YOLOE | Same |
| Merge features | DINOv3 mask-pool (`use_dino_features=True`) | Same |
| Gaussian fitting | MAD filter → Mahalanobis → weighted stats | Same |
| Output dict schema | `means, cov6, features, masks, scores, class_ids, batch_ids` | Identical |
| World transform | `transform_segmentation_to_world` (utils/geometry.py) | Inline equivalent |

**360 adaptation note:** No equirectangular warping is applied here. The DA3 pipeline already projected the 360 frame into 4 rectilinear 504×504 face tiles with known pinhole intrinsics. Phase 2 treats them as standard perspective images — this is correct.

---

## Outputs

One `detections_kfNNNNNN.pt` per keyframe, saved in `output/`. Each is a `dict` (torch-serialised) with:

| Key | Shape / Type | Description |
|---|---|---|
| `means` | `(M, 3)` float32 | 3D object centroids in **world frame** (metres) |
| `cov6` | `(M, 6)` float32 | Packed symmetric covariance `[xx,xy,xz,yy,yz,zz]` in world frame |
| `features` | `(M, 384)` float32 | DINOv3 L2-normalised appearance embeddings |
| `masks` | `list[Tensor(H,W) bool]` | 2D instance masks in original face resolution (camera frame) |
| `scores` | `(M,)` float32 | YOLOE confidence per detection |
| `class_ids` | `(M,)` int64 | Vocab index (into `vocab`) |
| `labels` | `list[str]` | Class name per detection |
| `batch_ids` | `(M,)` int64 | Which face (0–3) each detection came from |
| `intrinsics` | `list[Tensor(3,3)]` | K per face, for Phase 3 reprojection |
| `poses_world` | `list[Tensor(4,4)]` | T_wc per face, for Phase 3 covisibility |
| `face_meta` | `list[dict]` | rgb/depth paths, timestamp, face_id per face |
| `kf_id` | `str` | e.g. `"kf000042"` |
| `vocab` | `list[str]` | Full vocabulary list (for label lookup) |

`M` = total detections across all faces in this keyframe (may be 0).

A `phase2_summary.json` is also written at the end with keyframe counts, total detections, and run parameters.

---

## What Phase 3 expects

Phase 3 (neighbor lookup + correspondence + map update) loads each `.pt` and calls:

```python
# FARM's get_neighbors — uses features (cosine) + cov6/means (Hellinger)
neighbors, k_neighbors = get_neighbors(seg_outputs, scene_state, ...)

# FARM's find_object_correspondence — union-find resolves detection→object id
assignments = find_object_correspondence(seg_outputs, neighbors, ...)

# FARM's update_scene_graph_state — fuses matched detections into the map
update_scene_graph_state(scene_state, seg_outputs, assignments, ...)
```

The output dict from Phase 2 is schema-compatible with all three FARM calls. Phase 3 additionally uses `intrinsics` and `poses_world` to update the covisibility graph.

---

## Host flags that matter

### `--conf 0.35` (YOLOE confidence threshold)

This is the **NMS / detection score gate** inside YOLOE.

- Each proposal has a model confidence in `[0, 1]`.
- Detections **below** `--conf` are dropped before masks, 3D Gaussians, or DINOv3 features are computed.
- **0.35** matches FARM’s typical live-path range (~0.35–0.40): a middle ground between
  - **too low** (0.15–0.25) → lots of false positives / clutter → Phase 3 merge overloaded  
  - **too high** (0.5–0.6) → misses sparse plant / partial objects on faces  

Tune after validation: if overlays are full of junk, raise conf; if real objects are missing, lower it or expand `vocab/construction_vocab.txt`.

It is **not** a 3D quality threshold and **not** the DINOv3 feature threshold (Phase 3 uses cosine ≥ ~0.5 separately).

---

## Validate Phase 2 before Phase 3

Phase 2 is **per-keyframe proposals only**. You are **not** checking that the same truck has one ID across walks (that is Phase 3). You **are** checking that each frame’s detections look right in 2D + land in sensible 3D with merge-ready features.

### What to verify

| Check | Good | Bad (do not proceed) |
|---|---|---|
| 2D overlays | Masks on real objects; labels plausible | Blank sky as “wall”, random blobs, all faces empty |
| Scores | Most kept dets mid–high (≈0.35–0.9) | Almost everything barely above conf |
| 3D means | Site-scale cloud in shared world frame | All points at origin / km-scale / NaNs |
| Gaussian size | Object-scale metres of extent | Near-zero (no depth under mask) or giant blobs |
| DINOv3 features | \|\|f\|\| ≈ 1 for (almost) all dets | Zero vectors or norms ≪ 1 |
| Density | Some empty outdoor frames OK; most kfs have a few dets | ≥70% empty kfs, or hundreds of dets/kf |

### What is *expected*

- **Face seams:** object split across two cube faces → two detections of the same thing (OK for Phase 2; Phase 3 merges).
- **False positives at conf edge:** common; tune `--conf` and vocab.
- **Depth failures:** thin / reflective / far objects → weak Gaussians (high “zero extent” rate is a WARN).
- **Poses + metric depth:** means form a coherent site footprint as the walk advances.

### Run the validator

After Phase 2 has written some `detections_kf*.pt`:

```bash
conda activate farm-phase2
cd /home/kodifly/Desktop/farm-git/pipeline/phase2-detect-segment-embed

python validate_phase2.py --det-dir ./output
```

Artifacts under `output/validation/`:

| File | Use |
|---|---|
| `summary.txt` / `metrics.json` | PASS / WARN / FAIL rollup + numbers |
| `overlays/*.jpg` | Mask + label + score + rough world radius on each face |
| `world_xy_scatter.png` | Top-down means (color = Z) |
| `overlays_3d.html` | Interactive 3D means (if plotly installed) |
| `score_hist.png` | Confidence distribution |
| `feature_norm_hist.png` | Should peak near **1.0** |
| `class_hist.png` | What YOLOE is firing on |
| `det_count_per_kf.png` | Density over the walk |
| `mean_radii_hist.png` / `extent_hist.png` | Metric-scale sanity |

**Gate to Phase 3:** `summary.txt` status is `PASS` or only soft `WARN`s you understand, and random spot-checks of `overlays/` look correct.

```bash
# optional nicer 3D HTML
pip install plotly pandas
```

---

## How to run

### Host conda env (no Docker)

One-time setup (creates `farm-phase2`: Python 3.10 + CUDA torch + YOLOE + MobileCLIP + farm `scene_graph`):

```bash
bash /home/kodifly/Desktop/farm-git/pipeline/phase2-detect-segment-embed/setup_conda_env.sh
```

Then each session:

```bash
conda activate farm-phase2
cd /home/kodifly/Desktop/farm-git/pipeline/phase2-detect-segment-embed

python phase2_runner.py \
    --frames-json /home/kodifly/Desktop/depth-reconstr/da3_scan_depth/frames_json/frames.json \
    --data-root   /home/kodifly/Desktop/depth-reconstr/da3_scan_depth \
    --output-dir  ./output \
    --device cuda \
    --conf 0.35
```

Fast smoke (every 5th keyframe): add `--stride 5`. Interrupted runs resume automatically (existing `detections_kf*.pt` files are skipped).

Outputs land in `output/`. Prefer [`setup_conda_env.sh`](setup_conda_env.sh) over plain `conda env create -f environment.yml` so CUDA PyTorch is not replaced by a CPU build.

---

## File index

| File | Role |
|---|---|
| `phase2_runner.py` | Entry point — reads frames.json, loops keyframes, saves .pt |
| `segmenter.py` | `ConstructionSegmenter` — YOLOE + MobileCLIP + DINOv3 wrapper |
| `dino_extractor.py` | `Phase2DinoExtractor` — mask-pool DINOv3, standalone use |
| `geometry.py` | Unproject, MAD filter, Mahalanobis, weighted stats, world transform |
| `vocab/construction_vocab.txt` | ~75 construction-site object labels |
| `output/` | (gitignored) per-keyframe `detections_kf*.pt` files |
