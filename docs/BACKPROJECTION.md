# Back-projection math: cubemap faces and FARM pinhole unprojection

This note is the contract for how 360° Insta360 frames become 3D points in
`farm-object-map`. Two stages are distinct on purpose:

1. **Equirect → perspective cubemap faces** (ss-3dgs). This is *not* FARM.
2. **Pixel + optical-axis depth → camera XYZ → world** (FARM
   `YOLOESegmenter`). This *is* FARM, copied, not re-derived.

FARM never ingests an equirectangular image. It assumes each input is already a
pinhole RGB-D frame with known \(K\) and \(T_{\mathrm{world}\leftarrow\mathrm{cam}}\).
Cubemap rendering exists only so that assumption becomes true.

---

## 0. Coordinates and symbols

| Symbol | Meaning |
|---|---|
| \((u,v)\) | Pixel column / row. Origin top-left. \(u\) right, \(v\) down. |
| \(Z\) | **Optical-axis depth** (OpenCV \(+Z\)), not ray length. |
| \(K\) | Pinhole intrinsics \(\begin{bmatrix}f_x&0&c_x\\0&f_y&c_y\\0&0&1\end{bmatrix}\) |
| \(T_{\mathrm{cam}\leftarrow\mathrm{world}}\) | COLMAP `image.cam_from_world`, world → camera SE(3) |
| \(T_{\mathrm{world}\leftarrow\mathrm{cam}}\) | Inverse; camera centre is the translation of this matrix |
| Equirect size | \(W_p \times H_p\) with \(W_p = 2 H_p\) (true 360×180) |

Camera frame (OpenCV / COLMAP / FARM): \(+X\) right, \(+Y\) down, \(+Z\) forward.

---

## 1. Stage A — ss-3dgs cubemap (not FARM)

Source: `ss-3dgs/utils/pano_utils.py`, `ss-3dgs/src/pano_processing.py`.

Default render type matches `configs/ss3dgs_sfm_only.yaml`:

`cubemap-nosfm-top-and-bottom`

- 6 virtual cameras: bottom, 4 sides, top
- HFOV = VFOV = \(90^\circ\)
- Top and bottom are **rendered** but **excluded from SfM** (zeroed COLMAP masks)
- Horizontal ring yaws: \(0^\circ, 90^\circ, 180^\circ, 270^\circ\) (front, right, back, left)

### 1.1 Virtual-camera orientation

For yaw \(\alpha\) and pitch \(\beta\) (degrees):

\[
R_{\mathrm{cam}\leftarrow\mathrm{pano}}
= R_X(-\beta)\, R_Y(-\alpha)
\]

(`scipy.spatial.transform.Rotation.from_euler("XY", [-pitch, -yaw])`).

All virtual cameras share the **same optical centre** as the panorama (pure
rotation rig, zero translation). COLMAP is told this via `apply_rig_config`.

### 1.2 Face resolution and intrinsics

Face width / height from equirect FOV fractions:

\[
w = \mathrm{round}(W_p \cdot \mathrm{HFOV}/360),\quad
h = \mathrm{round}(H_p \cdot \mathrm{VFOV}/180)
\]

At \(3840\times1920\) and \(90^\circ\): \(w=h=960\).

Focal length (SIMPLE_PINHOLE, \(f_x=f_y=f\)):

\[
f = \frac{w}{2\tan(\mathrm{HFOV}/2)}
= \frac{w}{2}
\quad\text{when HFOV}=90^\circ.
\]

Principal point is the image centre in COLMAP’s SIMPLE_PINHOLE convention
(\(c_x=w/2\), \(c_y=h/2\)).

### 1.3 Pixel rays used **only for sampling the equirect**

ss-3dgs builds a unit ray per **pixel centre**:

\[
\begin{aligned}
\tilde u &= u + 0.5,\quad \tilde v = v + 0.5,\\
x_n &= (\tilde u - c_x)/f,\quad
y_n = (\tilde v - c_y)/f,\\
\mathbf{r}_{\mathrm{cam}} &= \mathrm{normalize}(x_n,\, y_n,\, 1).
\end{aligned}
\]

Rotate into panorama frame:

\[
\mathbf{r}_{\mathrm{pano}} = \mathbf{r}_{\mathrm{cam}}^\top R_{\mathrm{cam}\leftarrow\mathrm{pano}}
\quad\text{(row-vector form in the code: } \texttt{rays\_in\_cam @ cam\_from\_pano\_r}\text{)}.
\]

Map to equirect pixel coordinates (spherical):

\[
\begin{aligned}
\mathrm{yaw} &= \mathrm{atan2}(r_x, r_z),\\
\mathrm{pitch} &= -\mathrm{atan2}\bigl(r_y, \sqrt{r_x^2+r_z^2}\bigr),\\
U &= \frac{W_p}{2}\bigl(1 + \mathrm{yaw}/\pi\bigr),\\
V &= \frac{H_p}{2}\bigl(1 - 2\,\mathrm{pitch}/\pi\bigr).
\end{aligned}
\]

`cv2.remap` with wrap at the 0/360 seam (optional yaw seam-roll first).

**This +0.5 is a rendering convention only.** It is *not* part of FARM
unprojection.

### 1.4 What COLMAP sees

Each face is a genuine pinhole image. SfM estimates one pose per timestamp for
the rig reference sensor; other faces are locked by the known relative
rotations. After export we still emit per-face

\[
T_{\mathrm{world}\leftarrow\mathrm{cam}} = \bigl(T_{\mathrm{cam}\leftarrow\mathrm{world}}\bigr)^{-1}
\]

exactly as for ordinary pinhole frames.

---

## 2. Stage B — FARM back-projection (exact)

Source: `FARM-Project/src/scene_graph/segmentation/yoloe.py` around 1147–1151,
plus `transform_segmentation_to_world` in `scene_graph/utils/geometry.py`.

When `--association farm`, we do **not** reimplement this. We call
`YOLOESegmenter` and `PipelineOrchestrator`. The greedy-IoU path uses
`geometry.unproject_pixels`, which is the same algebraic formula.

### 2.1 Pixel grid

FARM builds

```text
u = 0, 1, …, W-1
v = 0, 1, …, H-1
```

with `torch.arange` / `meshgrid(..., indexing="ij")`. **No +0.5.** Pixel
\((0,0)\) is treated as the top-left integer index, not the pixel centre.

### 2.2 Letterboxed \(K\)

YOLOE letterboxes RGB and depth to `imgsz=640` (pad value 114/255 for RGB, 0
for depth). Intrinsics are scaled:

\[
\begin{aligned}
f' &= g\, f,\\
c_x' &= g\, c_x + p_{\mathrm{left}},\\
c_y' &= g\, c_y + p_{\mathrm{top}},
\end{aligned}
\]

where \(g=\min(640/H_{\mathrm{orig}}, 640/W_{\mathrm{orig}})\).

Unprojection is performed **in letterbox space** with \(K'\), then the resulting
camera-frame Gaussians / points are used as-is (still in the original camera
axes; letterbox is a 2D warp of the image plane, compensated by \(K'\)).

### 2.3 Unprojection formula (FARM)

\[
\boxed{
X = (u - c_x)\, Z / f_x,\qquad
Y = (v - c_y)\, Z / f_y,\qquad
Z = z
}
\]

\(Z\) is the depth map value: **along the optical axis**, not the Euclidean
range \(\|\mathbf{X}\|\).

Invalid depth: \(Z\le 0\) or non-finite. Masks are eroded by 3 px first.
Then:

1. 1-D depth-mode MAD filter (\(k=3.0\) live default)
2. Weighted mean / cov of inlier \((X,Y,Z)\)
3. Mahalanobis reject (\(2.0\sigma\))
4. **Mean replaced by coordinate-wise median** of surviving points
5. Inlier points stored for voxel ingest

### 2.4 Camera → world (FARM)

`transform_segmentation_to_world`:

\[
\boldsymbol{\mu}_w = R\, \boldsymbol{\mu}_c + \mathbf{t},\qquad
\Sigma_w = R\, \Sigma_c\, R^\top
\]

where \(R,t\) come from \(T_{\mathrm{world}\leftarrow\mathrm{cam}}\) (the pose
FARM calls `poses_world`). Offline orchestrator further left-multiplies by the
inverse of the first frame pose (`relative_pose`) so the map origin is the first
keyframe.

Points:

\[
\mathbf{p}_w = R\, \mathbf{p}_c + \mathbf{t}.
\]

Voxel keys use FARM’s 5 mm base grid (`VOXEL_BASE_V = 0.005`).

---

## 3. Where we match FARM vs where we diverge

| Step | FARM | This experiment |
|---|---|---|
| Input image | Pinhole RGB-D | Cubemap **face** (pinhole) after ss-3dgs render |
| \(K\) | Dataset / SLAM pinhole | SIMPLE_PINHOLE of that face (ss-3dgs formula) |
| Pose | Metric SLAM \(T_{wc}\) | COLMAP face pose, up-to-scale until DL depth alignment |
| Unprojection | §2.3 | **Identical** (call FARM code on farm path) |
| Depth units | Metres | Metres after `dl_depth_v1` + scale align; SfM units if MVS opt-in |
| Pixel convention | Integer index, no +0.5 | Same for unprojection. +0.5 only inside cubemap **resample** |
| Equirect unprojection | Does not exist | We **do not** unproject on the sphere |
| DINO | `resolve_dino_backbone()` | Same call path |
| Association | Hellinger² + feat + UF | Same FARM modules |

Divergences that remain even after cubemap:

1. **Scale.** FARM’s SLAM is metric. Monocular COLMAP is not. `dl_depth_v1`
   depth is metric and independent. `align_poses_to_metric_depth` estimates
   \(s=\mathrm{median}(Z_{\mathrm{DL}}/Z_{\mathrm{SfM}})\) on paired pixels and
   multiplies \(T_{wc}\) translations by \(s\). Skip this and Hellinger
   thresholds (metres) are meaningless.
2. **Top/bottom faces** are not in SfM (`cubemap-nosfm-top-and-bottom`). They
   can still be rendered for detection later if we assign rig poses, but they
   are not used to constrain the trajectory.
3. **Letterbox vs face size.** Faces are 960²; YOLOE letterboxes to 640². FARM
   does the same letterbox on Replica 1200×680. No extra change.
4. **Seam / 90° FOV overlap.** Adjacent 90° faces share a ray at the edge.
   ss-3dgs optionally masks a few edge pixels for SfM (`cubemap_edge_ignore_margin_px`).
   Object masks near a face edge may be split across two virtual cameras — FARM
   never sees that split on a single pinhole stream. Association (Hellinger +
   DINO) is what glues those fragments in world space.

---

## 4. What we deliberately do *not* do

- Spherical / equirect unprojection \( \mathbf{d}(\mathrm{yaw},\mathrm{pitch}) \cdot r \).
  That would be a different camera model from FARM and would invalidate
  Hellinger Gaussians trained/tuned on pinhole RGB-D.
- Treating the full 3840×1920 equirect as one PINHOLE \(K\) (the Phase-1 smoke
  that registered 24 frames). That SfM can look numerically healthy and still
  be geometrically wrong.

---

## 5. Drop-in depth (ready, not yet live)

`farm_object_map.dl_depth_v1`:

- `register_infer_fn(fn)` or env `FARM_DL_DEPTH_INFER=module:fn`
- `fn(rgb_bgr, frame_name) -> (depth_m, valid_mask)` on the **same grid as the
  RGB that FARM will see** (cubemap face, not the equirect)
- `--dl-depth-npz-dir` for precomputed `DepthMap` files
- `align_poses_to_metric_scale` before mapping

Until one of those is wired, mapping stops. No silent MVS.

---

## 6. “FARM vs greedy IoU object counts” (what that comparison is)

This is **not** a 2D detection metric and **not** “which detector is better.”

On one posed RGB-D sequence we run **the same YOLOE masks** through two
identity rules:

| Method | What counts as “the same object” across frames |
|---|---|
| `greedy_iou` | Same class label + 2D bbox IoU ≥ 0.3 vs last seen track (cheap baseline) |
| `farm` | DINO (or LRPC) cosine > 0.5 **and** Gaussian Hellinger² < 0.8, then union-find merge with cannot-link / one-to-one |

We report:

- \(N_{\mathrm{farm}}\) = number of active scene-graph objects after the clip
- \(N_{\mathrm{iou}}\) = number of distinct IoU track IDs created on the same dets

If \(N_{\mathrm{iou}} \gg N_{\mathrm{farm}}\), FARM is merging views that the
2D tracker splits (expected when the camera orbits / faces change). If
\(N_{\mathrm{farm}} \gg N_{\mathrm{iou}}\), Hellinger/DINO is failing to link
(depth mis-scale, bad Gaussians, or DINO off). If they are close, either the
motion is tiny or both rules are under-associating.

It cannot be computed without depth, because FARM’s link test is 3D.
)
