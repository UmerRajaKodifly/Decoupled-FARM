# FARM Point / Voxel Cloud Generation — Full Pipeline

This document traces **exactly** how FARM turns raw sensor data into the per-object sparse voxel clouds stored in `scene_state.pt`. It starts at the sensor / dataset layer and ends at persistence and visualization.

**Primary code paths:**
- Depth load: `src/scene_graph/offline/frame_sources/*`
- LiDAR → depth (optional): `scripts/lidar_bag_to_frames.py`, `ros/mapping/mapping/lib/odin1_projection.py`
- Mask → 3D points: `src/scene_graph/segmentation/yoloe.py`
- Camera → world: `src/scene_graph/utils/geometry.py` (`transform_segmentation_to_world`)
- Voxelize + merge: `src/scene_graph/map_update/object_update.py`, `utils/geometry.py` voxel helpers

---

## 0. What "point cloud" means in FARM

There are three related but distinct things:

| Name | What it is | Role |
|------|------------|------|
| **A. Depth image** | Dense (or sparse) `H×W` metric depth in metres | Required input to mapping |
| **B. Per-detection / per-object point / voxel cloud** | 3D points (then voxel keys) for one object | **Core geometry** used for association, AABBs, retrieval |
| **C. Background `cloud.npz`** | Accumulated scene XYZ for Viser | Visualization only — not used to build the scene graph |

This document focuses on **A → B**. Section 8 covers **C**.

FARM does **not** run LiDAR SLAM or build a global dense reconstruction as its memory. Object geometry is always: **mask ∩ depth → unproject → (filter) → voxelize → merge across views**.

---

## 1. Stage 0 — Raw data: how depth arrives

Every mapping frame eventually becomes:

```text
rgb:        (H, W, 3) uint8
depth_f32:  (H, W) float32   # metres; 0 / NaN = invalid
K:          3×3 pinhole (or equivalent fx,fy,cx,cy)
T_world_cam: 4×4 camera-to-world
```

Depth can come from three families of sources.

### 1.1 Native depth camera / dataset depth (most common)

**Offline `frames-json`** (`offline/frame_sources/frames_json.py`):

- RGB from JPEG/PNG via `rgb_path`
- Depth from:
  - `.npy` — already float32 metres, or
  - `.png` uint16 — converted as  
    `depth_m = png_count * scale_to_metres`  
    where `scale_to_metres` comes from `frames.json`'s `depth_encoding` block (default **0.001** = 1 mm/count if omitted)
- Invalid handling: non-finite → 0; optional clip above `_depth_clip_m`; negatives → 0

**FARM-Scenes** uses uint16 PNG + per-scene `depth_encoding` because outdoor LiDAR-derived depth can exceed the 65.535 m range of plain millimetre uint16.

**NPZ** (`frame_sources/npz.py`): `depths` array `(N,H,W)` float32 metres; NaN/0 invalid.

**ScanNet `.sens`**: depth frames decoded from the archive alongside RGB and poses.

**ROS depth camera** (Spot, etc.): aligned depth image topic synchronized with RGB by `frame_pub` into `RGBDFrame`.

### 1.2 LiDAR `PointCloud2` → synthesized depth (no depth camera)

Used when the rig has RGB + LiDAR + odometry but **no** depth image (e.g. Berkeley Odin1).

**Offline converter:** `scripts/lidar_bag_to_frames.py`  
**Online node:** `odin1_depth_pub` using `ros/mapping/mapping/lib/odin1_projection.py`

Pipeline per RGB frame:

1. Read LiDAR `sensor_msgs/PointCloud2` (default topic `/odin1/cloud_slam`) — points already in **world/odom** frame.
2. Interpolate camera pose from odometry at the image timestamp (`interpolate_pose` + extrinsic `T_base_cam`).
3. Transform world points into the camera frame; project with fisheye (`project_fishpoly`) or pinhole.
4. Splat into an `(H,W)` z-buffer: each pixel stores a valid camera-frame `z` (depth image).
5. Optional **dilation** (`dilate_sparse_depth`, `--depth-dilate-px`) fills empty pixels from nearby valid depths — LiDAR is sparse.
6. Write depth into a `frames-json` scene (offline), or republish as a depth image topic (online).

After this step, the rest of FARM sees a normal `depth_f32` image — it does **not** keep the raw LiDAR cloud as object geometry.

### 1.3 What is *not* a depth source

- Stella / monocular SLAM point clouds (those are Spatial GPT / VPA, not FARM)
- The viz `cloud.npz` file (see §8)
- RGB alone — without depth, YOLOE cannot form 3D Gaussians / voxel support

---

## 2. Stage 1 — Frame enters the mapper

Both online and offline call into `StreamingMapper._run_mapping_batch` (or the equivalent `pipeline/steps.py` path).

Per batch:

1. **Decode** RGB-D + pose (`_decode_batch` / `FrameSource`)
2. **Prepare tensors** — depth resized to RGB resolution if needed; poses as 4×4 world←cam
3. **Segment** — `segmenter(colors, depths, intrinsics)` → YOLOE backend

At this point depth is still an image. Points do not exist yet.

---

## 3. Stage 2 — 2D open-vocab masks (YOLOE)

**File:** `segmentation/yoloe.py` → `YoloESegmenter.__call__`

The goal of this stage is **not** yet 3D. It produces, for each RGB-D frame, a set of **instance masks in a shared pixel grid** that will later select which depth pixels become an object’s point cloud.

Conceptually:

```text
RGB image  →  (letterbox)  →  YOLOE  →  boxes + soft masks
Depth map  →  (same letterbox)  →  aligned depth + adjusted K
Soft masks →  threshold →  binary letterbox masks
Binary masks →  morphological erosion  →  masks used for 3D
```

Everything below happens **per frame** (or per batch of frames on GPU), still in **image / camera coordinates**.

> **Note on math notation:** equations below are plain monospace / Unicode so they render in Cursor’s Markdown preview (which does not enable LaTeX/KaTeX by default).

---

### 3.1 Why letterbox exists (geometric motivation)

YOLOE expects a fixed square input, typically `H_t = W_t = 640` (`imgsz`). Real cameras have arbitrary original size `(H_0, W_0)`.

**Letterbox** means: *uniformly scale the image so it fits inside the square, then pad the unused borders*, instead of stretching (which would warp aspect ratio and break the pinhole model).

Define the scale (gain):

```text
g = min(H_t / H_0,  W_t / W_0)
```

Resized content size:

```text
H_r = round(H_0 * g)
W_r = round(W_0 * g)
```

Symmetric padding into the target canvas:

```text
d_h = H_t - H_r
d_w = W_t - W_r

top    = round(d_h/2 - 0.1)
bottom = round(d_h/2 + 0.1)
left   = round(d_w/2 - 0.1)
right  = round(d_w/2 + 0.1)
```

So a pixel `(u_0, v_0)` in the **original** image maps to letterbox coordinates approximately:

```text
u = g * u_0 + left
v = g * v_0 + top
```

RGB is resized with **bilinear** interpolation and padded with a constant gray (`114/255`). Depth uses the **same** `g`, `left`, `top` but **nearest-neighbor** resize (so depth values are not blended across surfaces) and pad value **0** (invalid depth).

**Why this matters for points later:** mask pixels and depth pixels must live on the **same** `(u, v)` grid. FARM therefore does 3D on the **letterbox grid**, not on the original resolution, so a batch of differently sized cameras can be stacked as tensors of shape `(B, H_t, W_t)`.

---

### 3.2 Intrinsics must be transformed with the letterbox

Original pinhole projection (OpenCV / optical frame convention used here):

```text
u_0 = f_x * (X / Z) + c_x
v_0 = f_y * (Y / Z) + c_y
```

After letterbox, the same 3D point projects to:

```text
u = g * u_0 + left
v = g * v_0 + top
```

which is equivalent to a new camera matrix `K'`:

```text
f_x' = g * f_x
f_y' = g * f_y
c_x' = g * c_x + left
c_y' = g * c_y + top
```

FARM builds exactly this adjusted `K'` when letterboxing depth (`_letterbox_depth` + the block around lines 1123–1137). Unprojection in §4 uses `f_x'`, `f_y'`, `c_x'`, `c_y'` on the letterbox grid — **not** the original `K`.

If you forgot this adjustment, every 3D ray would be wrong by the pad/scale, and object Gaussians would systematically shift.

---

### 3.3 What YOLOE actually outputs (instance segmentation theory)

YOLOE is an **open-vocabulary** detector/segmenter: class names come from a text vocabulary (~1516 prompts in `configs/yoloe_vocabulary.txt`), not a fixed closed set of COCO IDs only.

Forward pass (Ultralytics-style) yields, per image:

| Output | Meaning |
|--------|---------|
| Boxes | Axis-aligned rectangles in letterbox pixels |
| Class scores | Confidence per vocab class |
| Mask coefficients + prototypes | Soft instance masks via `ops.process_mask` |

**Mask decoding (high level):** YOLO-seg family models predict a small set of **mask coefficients** per detection and a shared **prototype tensor**. The soft mask is a linear combination of prototypes, upsampled to the letterbox resolution, then thresholded:

```text
M_soft_i(u,v) ∈ [0, 1]

M_i(u,v) = 1  if  M_soft_i(u,v) > 0.5
         = 0  otherwise
```

FARM keeps these binary masks as `masks_letterbox` with shape `(M, H_t, W_t)` for all `M` detections in the batch.

**NMS** (`conf_thres`, `iou_thres`, `max_det`) removes overlapping boxes of the same/similar classes so you do not get dozens of near-duplicate masks for one physical object in one frame. (Cross-frame identity is handled later by union-find — not here.)

**Two mask resolutions in code:**

- `masks_letterbox` — used for **depth / 3D** (aligned with letterboxed depth).
- Masks scaled back to original `(H_0, W_0)` — used for 2D viz / evidence crops, not for unprojection.

---

### 3.4 Morphological erosion — why shrink the mask before 3D?

Segmentation masks are soft at boundaries. Near the silhouette, many pixels are:

- half object / half background in RGB, and/or  
- **background depth** (the wall behind a chair, the floor under a table edge)

If you unproject the full mask, those boundary depth values pull the 3D mean toward the background plane and inflate the covariance. That breaks Hellinger association later.

FARM therefore **erodes** each binary mask by `mask_erosion_px` pixels (default **3**):

```text
M_eroded = M ⊖ B_r
```

where `B_r` is a square structuring element of radius `r` (implemented as: dilate the *complement* of `M` with max-pool of kernel `2r+1`, then invert — equivalent to binary erosion).

**Geometric meaning:** only pixels at least `r` pixels inside the silhouette may contribute depth. You trade some surface coverage for cleaner depth statistics.

```text
masks_depth = _erode_masks(masks_letterbox, mask_erosion_px)
```

---

### 3.5 End state of Stage 2

For each detection `i` on frame `b`, FARM now has:

- Binary eroded mask `M_eroded_i` on the letterbox grid  
- Letterboxed depth `Z_b(u,v)` and adjusted intrinsics `K'_b`  
- Class id / score / visual features (DINOv3 or YOLOE embeddings) — for association later, not for unprojection  

**No 3D points yet.** Stage 4 turns mask ∩ depth into XYZ.

---

## 4. Stage 3 — Unproject depth under each mask (camera-frame points)

This is where the **per-detection point cloud** is born. The theoretical model is classical pinhole back-projection, followed by **robust statistical cleaning** of the resulting point set.

Throughout this section, coordinates are in the **camera optical frame** (OpenCV-style): **+X** right, **+Y** down, **+Z** forward along the optical axis. Depth `Z` is the metric Z-coordinate of the scene point (metres), not a disparity.

---

### 4.1 Pinhole unprojection (theory)

Given a pixel `(u, v)` with valid depth `Z > 0` and intrinsics `(f_x, f_y, c_x, c_y)` (the **letterbox-adjusted** ones from §3.2):

```text
X = ((u - c_x) / f_x) * Z
Y = ((v - c_y) / f_y) * Z
Z = Z
```

Equivalently, the camera ray through the pixel is

```text
r(u,v) = [ (u - c_x)/f_x ,
           (v - c_y)/f_y ,
           1 ]

p = Z * r(u,v)
```

**Implementation:** FARM does **not** loop over mask pixels first. It builds dense XYZ maps for the whole letterbox image once:

```text
ZB[b,v,u] = depth_letterbox[b,v,u]
XB[b,v,u] = (u - cx'_b) * ZB / fx'_b
YB[b,v,u] = (v - cy'_b) * ZB / fy'_b
```

Then, for each detection, it **indexes** those maps with the detection’s mask. That is mathematically identical to unprojecting only masked pixels, but fully vectorized on GPU.

**Valid depth:**

```text
valid(u,v)  ⟺  (Z(u,v) > 0)  AND  isfinite(Z(u,v))
```

Pad regions from letterbox have `Z = 0` and never unproject.

---

### 4.2 Selecting pixels: the initial inlier set

For detection `i` on image `b`, the initial weight map is the indicator of the eroded mask and valid depth:

```text
w_i^(0)(u,v) = 1  if  M_eroded_i(u,v) = 1  AND  valid_b(u,v)
             = 0  otherwise
```

Only these pixels may enter the object’s point cloud. Call this set `P_i^(0)`.

If `|P_i^(0)|` is tiny (holes in depth, over-erosion, bad mask), later median/Mahalanobis steps will fail the `min_depth_points` check (default 50) and the detection becomes unusable for geometry.

---

### 4.3 Depth-mode filter — 1D robust gate on Z (theory)

**Problem:** Even after erosion, a mask can still cover two depth modes — e.g. chair seat at 2 m and wall through gaps / soft edges at 5 m. A mean of `{2, 2, 2, 5, 5}` is nonsense for “where is the chair.”

**Approach:** Treat depth as a 1D sample and keep only values near the **dominant mode**, using median + MAD (median absolute deviation), which are robust to outliers (unlike mean + std).

Let `{Z_j}` be depths under the current weights (j ∈ `P_i^(0)`).

```text
m   = median({Z_j})
MAD = median({ |Z_j − m| })
```

Clamp MAD so very flat depth patches do not collapse the gate to a single bin:

```text
MAD' = max(MAD, δ_min)

δ_min = depth_mode_min_mad_m = 0.03 m   (default)
```

Keep pixel `j` iff

```text
Z_j ∈ [ m − k * MAD' ,  m + k * MAD' ]

k = depth_mode_k_mad = 3   (default)
```

Update:

```text
w_i^(1)(u,v) = w_i^(0)(u,v) * 1[Z(u,v) is in the band]
```

**Interpretation:** this is a robust, nonparametric “same-surface” filter along the optical axis before any 3D ellipsoid is fit. It does **not** yet use `X` or `Y` — only `Z`.

---

### 4.4 Sample covariance of the 3D points (theory)

On the surviving set `P_i^(1)` with weights `w`, FARM fits a **3D Gaussian** in camera coordinates. With binary weights this is ordinary sample mean / covariance; the code is written in weighted form:

```text
n = Σ_j w_j

μ = (1/n) * Σ_j w_j * p_j

Σ = (1/(n−1)) * Σ_j w_j * (p_j − μ)(p_j − μ)ᵀ     for n > 1
```

`Σ` is stored packed as 6 floats (upper triangle):

```text
cov6 = [ Σ_xx, Σ_xy, Σ_xz, Σ_yy, Σ_yz, Σ_zz ]
```

**Role of this Gaussian:**

1. **Immediate cleanup** via Mahalanobis (§4.5)  
2. **Later association** via Hellinger distance between Gaussians in the world frame (after Stage 4 transform)  
3. Compact object state even after voxels are coarsened  

It is **not** the final object centre used downstream for “mean” — see §4.6 (median).

---

### 4.5 Mahalanobis outlier rejection (theory)

**Problem:** Residual outliers that survived the 1D depth gate (e.g. a few wrong depths still near the median `Z`, or lateral noise) still pollute `Σ`.

**Approach:** Reject points that are far from `μ` **relative to the fitted covariance shape** — i.e. use the Mahalanobis distance, which measures distance in units of the ellipsoid’s own axes.

For a point `p`:

```text
d_M²(p) = (p − μ)ᵀ  Σ⁻¹  (p − μ)
```

(Implementation adds a tiny ridge `ε I` with `ε = 1e-6` before inverting so `Σ` is numerically SPD, then symmetrizes `Σ⁻¹`.)

Keep pixel `j` iff

```text
d_M²(p_j) ≤ τ²   AND   w_j^(1) > 0

τ = mahalanobis_thresh = 2.0   (default)
```

So `τ = 2` means “keep points inside roughly a **2-σ** ellipsoid” of the current fit (exact probabilistic coverage depends on dimension; in 3D this is a fixed geometric gate, not a calibrated χ² test).

Then **refit** `μ`, `Σ` on the survivors `w^(2)`.

**Order matters:** depth-mode (1D, robust) → Gaussian fit → Mahalanobis (3D, covariance-shaped) → refit. Depth-mode prevents a bimodal `Z` from producing a huge `Σ` that would then “accept” both modes under Mahalanobis.

---

### 4.6 Replace the mean with a coordinate-wise median (theory)

After Mahalanobis, the code **discards the sample mean as the object centre** and sets:

```text
μ_x = median({X_j})
μ_y = median({Y_j})
μ_z = median({Z_j})
```

over remaining inliers (requires at least `min_depth_points`, else NaNs).

**Why:** Even after filtering, the mean is pulled by remaining asymmetric tails (e.g. more pixels on a visible side of an object). The coordinate-wise median is a more robust location estimate for “where is this blob,” especially under depth noise. The **covariance `cov6` is left as the (post-Mahalanobis) sample covariance** — it still describes spread; only the centre used as `means` becomes the median.

---

### 4.7 Materialize the point list (what becomes the voxel cloud’s raw input)

Every pixel that still has `w^(2) > 0` contributes one camera-frame point:

```text
p_j = (X_j, Y_j, Z_j)
```

These are packed into CSR storage (not one Python list per detection):

```text
det_points_flat:     (P, 3) float32   # concatenation of all inlier points
det_points_offsets:  (M+1,) int64     # detection i owns flat[off[i] : off[i+1]]
```

**This list is the raw per-detection point cloud.** Stage 5–7 will:

1. Rotate/translate it to world (§5)  
2. Voxelize / unique (§7)  
3. Merge into the persistent object buffer  

Also emitted alongside (for association / viz, not for voxelization itself):

- `means` — median centre (camera frame for now)  
- `cov6` — sample covariance  
- masks, class ids, visual features  

---

### 4.8 Summary of the cleaning pipeline (one detection)

```text
letterbox depth Z(u,v), K'
        │
eroded mask M
        │
P0 = { pixels in M with Z valid }          ← geometric support
        │
depth-mode: keep Z near median±k·MAD       ← kill second depth surface
        │
fit N(μ, Σ) on XYZ                         ← elliptical model of the blob
        │
Mahalanobis: keep d_M ≤ τ                  ← kill 3D outliers vs that ellipse
        │
refit Σ; set μ ← coordinate-wise median    ← robust centre
        │
det_points = remaining XYZ                 ← camera-frame point cloud
```

**Still camera frame.** World alignment is the next stage.

---


## 5. Stage 4 — Transform points into the world frame

**File:** `utils/geometry.py` → `transform_segmentation_to_world(seg_outputs, poses_world)`

For each frame index `b` with pose `T_world_cam` (4×4):

```text
R = T[:3,:3],  t = T[:3,3]

means_world   = means_cam @ Rᵀ + t
cov_world     = R @ cov_cam @ Rᵀ          # then re-pack to cov6
points_world  = points_cam @ Rᵀ + t       # same R,t applied to det_points_flat
```

After this call, `seg_outputs["det_points_flat"]` is in the **shared world frame** used by the whole scene graph. Downstream voxelization always assumes world coordinates.

---

## 6. Stage 5 — Filtering before association

**File:** `map_update/filtering.py` (+ StreamingMapper filter stage)

Detections may be dropped for:

- mask touching the image border
- too few pixels / too few depth points
- too far from the camera
- uninformative labels (walls, floors, clutter — configurable)
- near-duplicate masks by 2D IoU in the same frame

Survivors keep their world-frame `det_points_*`, means, cov6, and features.

---

## 7. Stage 6 — Voxelize into the persistent object memory

**Files:** `map_update/object_update.py`, `utils/geometry.py` (voxel helpers)

Association (`get_neighbors` + `union_find` + `cannot_link`) decides whether a detection updates an **existing** object or creates a **new** one. Either way, the detection's world points are ingested via `_ingest_points_into_object`.

### 7.1 Voxel grid parameters

| Constant | Value | Meaning |
|----------|-------|---------|
| `VOXEL_BASE_V` | **0.005 m** (5 mm) | Level-0 voxel edge length |
| `VOXEL_CAP_PER_OBJECT` | **1000** | Max unique voxels per object |
| `VOXEL_K_INIT` | **32** | Target voxels along longest AABB axis for a new object |
| `VOXEL_MAX_LEVEL` | **15** | Coarsest allowed level |
| Packing | 21 bits/axis | Signed indices packed into `int64` keys |

Effective voxel size at level `L`:

```text
v(L) = VOXEL_BASE_V * 2^L
```

Levels are **monotone non-decreasing**: once coarsened, an object never refines again.

### 7.2 Quantization (`voxelize_points`)

```text
v = base_v * 2^L
qxyz = floor(points_xyz / v)     # integer grid indices
key  = pack(qxyz)                # 21+21+21 bit int64
keys = unique(keys)              # deduplicate
```

Decode back to world centers:

```text
xyz_center = qxyz * v + 0.5 * v
```

(`voxel_keys_to_world` / `decode_voxel_keys_numpy`)

### 7.3 New object — choose initial level

`init_voxel_level(pts)`:

1. Compute AABB of first observation's points; let `longest` be the longest edge.
2. Choose `L ≈ round(log2((longest / k_init) / base_v))` so the longest axis spans ~`k_init` voxels.
3. If unique voxels at that `L` still exceed `cap` (1000), bump `L` until it fits or `max_level`.

Then store those keys + level on the object.

### 7.4 Existing object — append and maybe promote

```text
new_keys = voxelize(pts, current_level)
merged   = unique(old_keys ∪ new_keys)
while |merged| > 1000 and level < max_level:
    merged = promote(merged)   # >> 1 on each axis index → 2× coarser
    level += 1
```

`promote_voxel_keys`: unpack → arithmetic right-shift by `levels` → repack → unique.

### 7.5 Object–object merge (union-find winners)

When two object IDs merge:

- `merge_voxel_buffers` promotes both to `max(level_a, level_b)`, unions keys, promotes further if over cap.
- Optional geometry guards refuse merges that explode AABB / covariance relative to members.
- Optionally **recompute the compact Gaussian from the merged voxel support** (`SCENE_GRAPH_RECOMPUTE_MERGED_GAUSSIAN_FROM_VOXELS`, default true) so Hellinger matching uses the tight evidence, not a stale pre-merge ellipsoid.

### 7.6 Storage layout in `scene_state.pt`

CSR-flat across all objects:

```text
object_voxel_keys_flat:    (M_total,) int64   # all keys concatenated
object_voxel_keys_offsets: (N+1,) int64       # object i → flat[off[i]:off[i+1]]
object_voxel_levels:       (N,) int8          # level L_i for object i
```

Plus the compact Gaussian (`means`, `cov6`, `count`) used for neighbor search.

**Tight AABB for viz / retrieval** = decode voxel keys → min/max (± half voxel), via `voxel_cloud_aabb`.

---

## 8. The separate background cloud (`cloud.npz`)

**Not part of stages 3–7.**

Used by `scripts/view_scene_state.py --cloud …/cloud.npz` as a static backdrop.

### How it is produced

**FARM-Scenes:** Precomputed and shipped with each scene (aggregated from that capture's depth/LiDAR).

**`lidar_bag_to_frames.py`:** While synthesizing depth, also concatenates LiDAR XYZ scans, optionally downsamples to ≤400k points, writes:

```text
cloud.npz  with key xyz (or points / cloud)
```

and may record `"cloud_path": "cloud.npz"` in `frames.json`.

Mapping can run **without** this file. Retrieval does not read it. Only the viewer uses it for human context over the object boxes / per-object voxel clouds.

---

## 9. End-to-end diagram

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ RAW                                                                     │
│  • Depth camera / dataset depth PNG·npy·sens                            │
│  • OR LiDAR PointCloud2 + odometry → project → depth image (§1.2)       │
│  • RGB + intrinsics K + pose T_world_cam                                │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ YOLOE (2D)                                                              │
│  letterbox RGB → detect/segment → masks                                 │
│  letterbox depth + adjust K                                             │
│  erode masks (mask_erosion_px)                                          │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ UNPROJECT (camera frame)          yoloe.py                              │
│  X=(u−cx)Z/fx, Y=(v−cy)Z/fy, Z=depth                                    │
│  keep: eroded_mask ∩ valid_depth                                        │
│  depth-mode filter (median ± k·MAD)                                     │
│  weighted Gaussian → Mahalanobis reject → median mean                   │
│  emit det_points_flat (P×3) + CSR offsets                               │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ WORLD TRANSFORM                   geometry.transform_segmentation_…     │
│  p_w = R p_c + t   (and cov similarly)                                  │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ ASSOCIATE                             get_neighbors + union_find        │
│  DINO features + Hellinger(Gaussians); cannot-link blocks               │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ VOXELIZE / MERGE                      object_update._ingest_points_…    │
│  q = floor(p_w / (0.005·2^L)); pack int64; unique                       │
│  new object: choose L from AABB (~32 voxels on long axis)               │
│  update: union keys; if >1000 unique → promote L                        │
│  store CSR keys + levels on SceneState                                  │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   ▼
                         scene_state.pt
              (per-object sparse voxel clouds + Gaussians)
```

---

## 10. Numerical defaults cheat sheet

| Knob | Typical default | Where |
|------|-----------------|-------|
| Depth PNG scale | 0.001 m/count (or scene `depth_encoding`) | `frames_json.py` |
| Mask erosion | 3 px | YOLOE / `configs/replica.yaml` |
| Min depth points | 50 | `yoloe.py` |
| Depth-mode k·MAD | 3.0; min MAD 0.03 m | `yoloe.py` |
| Mahalanobis thresh | 2.0 | `yoloe.py` / replica YAML |
| Base voxel size | 5 mm | `VOXEL_BASE_V` |
| Cap per object | 1000 voxels | `VOXEL_CAP_PER_OBJECT` |
| Init longest-axis voxels | 32 | `VOXEL_K_INIT` |
| Feature / Hellinger (association) | 0.5 / 0.8 | `pipeline/steps.py` |
| Max merge centre distance | 1.0 m (env overrideable) | `object_update.py` |

---

## 11. Practical implications

1. **No depth ⇒ no object voxels.** Bad depth (holes, wrong scale, misaligned to RGB) directly corrupts Gaussians and voxel AABBs.
2. **Pose errors** shear the world cloud: association and covisibility assume a consistent metric frame.
3. **LiDAR-synthesized depth is sparse** even after dilation — outdoor FARM-Scenes objects often have thinner support than indoor RGB-D.
4. **Coarse levels are permanent** for an object once the 1000-voxel cap forces promotion — large walls become coarse blobs by design.
5. **Background `cloud.npz` is unrelated** to object memory quality; improving viz cloud does not improve retrieval Acc@IoU.

---

## 12. File index

| Step | Path |
|------|------|
| frames-json depth decode | `src/scene_graph/offline/frame_sources/frames_json.py` |
| NPZ / sens ingress | `…/npz.py`, `…/sens.py` |
| LiDAR → depth + `cloud.npz` | `scripts/lidar_bag_to_frames.py` |
| Online LiDAR projection | `ros/mapping/mapping/lib/odin1_projection.py` |
| Mask + unproject + det points | `src/scene_graph/segmentation/yoloe.py` |
| Camera → world | `src/scene_graph/utils/geometry.py` |
| Voxel pack / promote / AABB | same `geometry.py` |
| Ingest / merge into SceneState | `src/scene_graph/map_update/object_update.py` |
| SceneState schema | `src/scene_graph/map_update/models.py` |
| Viewer background cloud | `scripts/view_scene_state.py` (`load_cloud`) |
