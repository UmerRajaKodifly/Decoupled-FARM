#!/usr/bin/env python3
"""
stella_to_da3_depth.py
======================
Convert Stella vSLAM keyframe outputs → per-face DA3METRIC depth maps → FARM frames-json.

Pipeline
--------
1. Read KF trajectory (TUM format) + Stella sparse landmarks (SQLite out.db)
2. For each KF: project equirect → 4 perspective cube faces (90° × 90°, 504 × 504)
3. Run DA3METRIC-LARGE on all face images (windowed batches)
4. Per-face scale alignment using Stella sparse landmark z-depths as anchors
5. Write FARM frames-json: (rgb JPEG, depth float32 .npy meters, K 3×3, T_wc 4×4)

Usage
-----
    conda run -n myenv python stella_to_da3_depth.py [OPTIONS]

    # Smoke test (first 20 KFs, 1 face each):
    conda run -n myenv python stella_to_da3_depth.py --max-kfs 20

    # Full run:
    conda run -n myenv python stella_to_da3_depth.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **_kw):  # type: ignore
        return x

# ---------------------------------------------------------------------------
# Paths — defaults empty; container entrypoints pass absolute paths via CLI.
# ---------------------------------------------------------------------------
def _env_path(name: str) -> Optional[Path]:
    v = os.environ.get(name, "").strip()
    return Path(v) if v else None


STELLA_DB = _env_path("STELLA_DB")
KF_IMAGE_DIR = _env_path("STELLA_KF_DIR")
KF_TRAJ = _env_path("STELLA_KF_TRAJ")
OUT_ROOT = _env_path("DA3_OUT_DIR") or Path("./outputs/phase1.5")

FACE_SIZE = 504       # pixels per cube-face side (matches DA3 process_res)
HFOV_DEG = 90.0       # horizontal FOV of each cube face
NUM_FACES = 4         # front, right, back, left

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit quaternion → 3×3 rotation matrix (column-major SO3)."""
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def face_rotations_cam_from_pano(
    num_yaw: int = NUM_FACES,
    pitches_deg: Tuple[float, ...] = (0.0,),
) -> List[np.ndarray]:
    """
    Return cam_from_pano rotation matrices matching COLMAP panorama_sfm convention.
    Yaws are evenly spaced starting at 0° (front).
    """
    rots: List[np.ndarray] = []
    yaws = np.linspace(0, 360, num_yaw, endpoint=False)
    for pitch_deg in pitches_deg:
        yaw_offset = (360 / num_yaw / 2) if pitch_deg > 0 else 0.0
        for yaw_deg in yaws + yaw_offset:
            pitch, yaw = np.deg2rad([-float(pitch_deg), -float(yaw_deg)])
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)
            Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
            rots.append(Rx @ Ry)
    return rots


def face_K(face_size: int = FACE_SIZE, hfov_deg: float = HFOV_DEG) -> np.ndarray:
    """Intrinsic matrix for a square face camera."""
    focal = face_size / (2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0))
    cx = cy = face_size / 2.0
    return np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)


def build_equirect_to_face_maps(
    equirect_hw: Tuple[int, int],
    face_size: int,
    R_cam_from_pano: np.ndarray,
    hfov_deg: float = HFOV_DEG,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build cv2.remap maps to warp equirect → perspective face.

    For each face pixel (u, v):
      1. Compute 3D direction in face frame
      2. Rotate to pano frame via R_cam_from_pano.T
      3. Project to equirect longitude/latitude → equirect pixel
    """
    H_eq, W_eq = equirect_hw
    K = face_K(face_size, hfov_deg)
    R_pano_from_face = R_cam_from_pano.T

    # Face pixel grid
    us = np.arange(face_size, dtype=np.float64)
    vs = np.arange(face_size, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)  # [face_size, face_size]

    # Direction in face camera frame.
    # Negate dy so that y+ points UP, matching the COLMAP equirectangular
    # geographic convention (latitude increases upward). Without this negation
    # the extracted face images appear upside-down.
    fx, cx = K[0, 0], K[0, 2]
    fy, cy = K[1, 1], K[1, 2]
    dx = (uu - cx) / fx
    dy = -(vv - cy) / fy   # y+ UP (geographic / COLMAP convention)
    dz = np.ones_like(dx)

    # Rotate to pano frame
    D = np.stack([dx, dy, dz], axis=-1)  # [H, W, 3]
    D_pano = D @ R_pano_from_face.T       # [H, W, 3]

    # Normalise
    norm = np.linalg.norm(D_pano, axis=-1, keepdims=True)
    D_pano = D_pano / np.maximum(norm, 1e-9)
    dpx, dpy, dpz = D_pano[..., 0], D_pano[..., 1], D_pano[..., 2]

    # Longitude / latitude → equirect pixel
    lon = np.arctan2(dpx, dpz)                # [-π, π]
    lat = np.arcsin(np.clip(dpy, -1.0, 1.0))  # [-π/2, π/2]

    map_x = ((lon / (2 * np.pi)) + 0.5) * W_eq
    map_y = (0.5 - lat / np.pi) * H_eq

    return map_x.astype(np.float32), map_y.astype(np.float32)


def extract_face(
    equirect_bgr: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    """Warp equirect image to perspective face using precomputed remap maps."""
    return cv2.remap(
        equirect_bgr, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


# ---------------------------------------------------------------------------
# Stella data loading
# ---------------------------------------------------------------------------

def load_kf_trajectory(traj_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load TUM keyframe trajectory.
    Returns {timestamp_str: (t_wc [3], R_wc [3,3])} — camera-to-world.
    """
    poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for line in traj_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        ts = parts[0]
        tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
        qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
        R_wc = quat_to_rotmat(qx, qy, qz, qw)
        t_wc = np.array([tx, ty, tz], dtype=np.float64)
        poses[ts] = (t_wc, R_wc)
    return poses


def load_stella_landmarks(db_path: Path) -> Dict[int, np.ndarray]:
    """Return {landmark_id: pos_world [3]} from Stella SQLite DB."""
    conn = sqlite3.connect(str(db_path))
    landmarks: Dict[int, np.ndarray] = {}
    for lid, pb in conn.execute("SELECT id, pos_w FROM landmarks"):
        landmarks[int(lid)] = np.frombuffer(pb, np.float64).copy()
    conn.close()
    return landmarks


def load_kf_landmark_ids(db_path: Path) -> Dict[int, np.ndarray]:
    """Return {kf_db_id: lm_id_array} from the associations table."""
    conn = sqlite3.connect(str(db_path))
    result: Dict[int, np.ndarray] = {}
    for kid, lm_b in conn.execute("SELECT id, lm_ids FROM associations"):
        result[int(kid)] = np.frombuffer(lm_b, np.int32).copy()
    conn.close()
    return result


def load_kf_metadata(db_path: Path) -> List[dict]:
    """
    Return list of {id, ts, img_name, pose_cw_4x4} for all keyframes,
    ordered by id.
    Image names follow the convention: image{id}.png (Stella dense output).
    """
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT id, ts, pose_cw FROM keyframes ORDER BY id"
    ).fetchall()
    conn.close()
    kfs = []
    for kid, ts, pose_b in rows:
        T_cw = np.frombuffer(pose_b, np.float64).reshape(4, 4).T.copy()
        kfs.append({
            "id": int(kid),
            "ts": float(ts),
            "img_name": f"image{kid}.png",  # Stella dense saves as image{id}.png
            "T_cw": T_cw,
        })
    return kfs


# ---------------------------------------------------------------------------
# Scale alignment: Stella sparse landmarks → metric scale for DA3 depth
# ---------------------------------------------------------------------------

def fit_scale_alpha(
    da3_depth_map: np.ndarray,
    T_cw_face: np.ndarray,
    K_face: np.ndarray,
    landmarks: Dict[int, np.ndarray],
    lm_ids: np.ndarray,
    face_size: int = FACE_SIZE,
    min_points: int = 5,
) -> float:
    """
    Fit a global scale alpha so that: da3_depth * alpha ≈ z_slam_landmark

    Projects visible Stella landmarks into the face camera, samples DA3 depth
    at those pixels, and returns the median ratio z_slam / da3_pred.
    Returns 1.0 if not enough valid anchors.
    """
    R = T_cw_face[:3, :3]
    t = T_cw_face[:3, 3]
    fx, fy = K_face[0, 0], K_face[1, 1]
    cx, cy = K_face[0, 2], K_face[1, 2]

    H = W = face_size
    slam_depths, pred_depths = [], []

    for lid in lm_ids:
        if lid < 0 or lid not in landmarks:
            continue
        p_w = landmarks[lid]
        p_c = R @ p_w + t
        z = float(p_c[2])
        if z <= 0.01:
            continue
        u = fx * p_c[0] / z + cx
        v = fy * p_c[1] / z + cy
        pi, pj = int(round(v)), int(round(u))
        if not (0 <= pi < H and 0 <= pj < W):
            continue
        d_pred = float(da3_depth_map[pi, pj])
        if d_pred <= 0 or not np.isfinite(d_pred):
            continue
        slam_depths.append(z)
        pred_depths.append(d_pred)

    if len(slam_depths) < min_points:
        return 1.0

    ratios = np.array(slam_depths) / np.array(pred_depths)
    # Robust: clip 10th/90th percentile outliers before median
    lo, hi = np.percentile(ratios, 10), np.percentile(ratios, 90)
    inliers = ratios[(ratios >= lo) & (ratios <= hi)]
    if len(inliers) < min_points:
        inliers = ratios
    alpha = float(np.median(inliers))
    log.debug(
        "Scale alignment: %d anchors → alpha=%.4f (inliers=%d)",
        len(slam_depths), alpha, len(inliers),
    )
    return alpha


# ---------------------------------------------------------------------------
# DA3 inference
# ---------------------------------------------------------------------------

def load_da3_runner(model_name: str = "depth-anything/DA3METRIC-LARGE", device: str = "cuda"):
    """Lazily import and initialise DA3Runner (requires myenv)."""
    da3_src = Path(__file__).parent / "Depth-Anything-3/colmap_depth_pipeline/src"
    da3_repo = Path(__file__).parent / "Depth-Anything-3/src"
    for p in (str(da3_src), str(da3_repo)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from da3_runner import DA3Runner, raw_to_metric_depth  # noqa: PLC0415
    runner = DA3Runner(model_name=model_name, device=device, process_res=FACE_SIZE)
    return runner, raw_to_metric_depth


# ---------------------------------------------------------------------------
# FARM frames-json writer
# ---------------------------------------------------------------------------

def write_frames_json(
    out_dir: Path,
    records: List[dict],
) -> Path:
    """
    Write FARM-compatible frames.json.
    Each record: {image_path, depth_path, K [3,3 list], T_wc [4,4 list], ts}
    """
    frames_json_dir = out_dir / "frames_json"
    frames_json_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for r in records:
        # Copy image and depth into frames_json subfolders
        rel_img = Path(r["image_path"]).name
        rel_dep = Path(r["depth_path"]).name
        entries.append({
            "timestamp": r["ts"],
            "rgb": rel_img,
            "depth": rel_dep,
            "depth_encoding": "float32_m",   # FARM convention: float32 metres
            "K": r["K"],
            "T_wc": r["T_wc"],
            "camera": r.get("camera", "face"),
        })

    manifest_path = frames_json_dir / "frames.json"
    with open(manifest_path, "w") as f:
        json.dump({"frames": entries}, f, indent=2)
    log.info("Wrote frames.json with %d entries → %s", len(entries), manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    import torch

    stella_db = Path(args.db)
    kf_image_dir = Path(args.kf_image_dir)
    kf_traj = Path(args.kf_traj)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    faces_dir = out_root / "faces"
    depth_dir = out_root / "depth"
    vis_dir = out_root / "depth_vis"
    frames_dir = out_root / "frames_json" / "images"
    depths_out_dir = out_root / "frames_json" / "depths"
    for d in (faces_dir, depth_dir, vis_dir, frames_dir, depths_out_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- Load Stella data ---
    log.info("Loading Stella data from %s", stella_db)
    kf_meta = load_kf_metadata(stella_db)
    landmarks = load_stella_landmarks(stella_db)
    kf_lm_ids = load_kf_landmark_ids(stella_db)
    log.info("KFs: %d  Landmarks: %d", len(kf_meta), len(landmarks))

    if args.max_kfs:
        kf_meta = kf_meta[: args.max_kfs]
        log.info("Capped to %d KFs (--max-kfs)", len(kf_meta))

    # --- Face geometry ---
    R_faces = face_rotations_cam_from_pano()          # [4] cam_from_pano
    K_face = face_K(FACE_SIZE, HFOV_DEG)
    log.info("Face K:\n%s", K_face)

    # Build remap maps once per face (equirect → face)
    equirect_hw = (960, 1920)  # from dense_batch_1920.yaml: rows=960, cols=1920
    log.info("Building %d equirect→face remap maps …", NUM_FACES)
    remap_maps = [
        build_equirect_to_face_maps(equirect_hw, FACE_SIZE, R_faces[i], HFOV_DEG)
        for i in range(NUM_FACES)
    ]

    # --- Extract face images ---
    log.info("Extracting face images …")
    face_image_paths: Dict[Tuple[int, int], Path] = {}  # (kf_id, face_id) → path

    for kf in tqdm(kf_meta, desc="Phase1.5 faces", dynamic_ncols=True, leave=True):
        kid = kf["id"]
        img_name = kf["img_name"]

        # KF images are named e.g. "frame000001.png" or similar; find by id
        # Stella saves keyframe images as e.g. keyframe{id:06d}.png
        # Look for any matching pattern
        img_path = kf_image_dir / img_name  # image{id}.png
        if not img_path.is_file():
            # Try legacy patterns just in case
            for pat in [f"frame{kid:06d}.png", f"frame{kid:04d}.png",
                        f"keyframe{kid:06d}.png", f"{kid:06d}.png",
                        f"image{kid:04d}.png"]:
                cand = kf_image_dir / pat
                if cand.is_file():
                    img_path = cand
                    break
        if not img_path.is_file():
            log.warning("KF %d: image not found at %s, skipping", kid, img_path)
            continue

        equirect = cv2.imread(str(img_path))
        if equirect is None:
            log.warning("KF %d: failed to read %s", kid, img_path)
            continue

        # Resize to expected equirect resolution if needed
        if equirect.shape[:2] != equirect_hw:
            equirect = cv2.resize(equirect, (equirect_hw[1], equirect_hw[0]))

        for fi, (mx, my) in enumerate(remap_maps):
            face_img = extract_face(equirect, mx, my)
            out_p = faces_dir / f"kf{kid:06d}_face{fi}.jpg"
            cv2.imwrite(str(out_p), face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            face_image_paths[(kid, fi)] = out_p

    log.info(
        "Extracted %d face images (%d KFs × %d faces)",
        len(face_image_paths), len(kf_meta), NUM_FACES,
    )

    # --- DA3 inference ---
    log.info("Loading DA3METRIC model …")
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    runner, raw_to_metric_depth = load_da3_runner(args.model, device)

    # Build simple FaceEntry-like objects for DA3Runner
    # (DA3Runner only needs image_path, extrinsics, intrinsics, width, height)
    da3_src = Path(__file__).parent / "Depth-Anything-3/colmap_depth_pipeline/src"
    if str(da3_src) not in sys.path:
        sys.path.insert(0, str(da3_src))
    from colmap_io import FaceEntry  # noqa: PLC0415

    # Prepare T_cw for each (kf_id, face_id)
    log.info("Computing face extrinsics from Stella poses …")
    face_T_cw: Dict[Tuple[int, int], np.ndarray] = {}
    for kf in kf_meta:
        kid = kf["id"]
        T_cw = kf["T_cw"]  # 4×4 world-to-camera
        R_cw = T_cw[:3, :3]
        t_cw = T_cw[:3, 3]
        for fi, R_face in enumerate(R_faces):
            R_cw_face = R_face @ R_cw
            t_cw_face = R_face @ t_cw
            T_cw_face = np.eye(4, dtype=np.float64)
            T_cw_face[:3, :3] = R_cw_face
            T_cw_face[:3, 3] = t_cw_face
            face_T_cw[(kid, fi)] = T_cw_face

    # Build FaceEntry list in order
    all_face_entries: List[FaceEntry] = []
    entry_map: Dict[Tuple[int, int], int] = {}  # (kf_id, face_id) → list index
    img_id = 0
    for kf in kf_meta:
        kid = kf["id"]
        for fi in range(NUM_FACES):
            if (kid, fi) not in face_image_paths:
                continue
            fe = FaceEntry(
                frame_id=f"kf{kid:06d}",
                face_id=fi,
                image_id=img_id,
                image_name=f"kf{kid:06d}_face{fi}.jpg",
                image_path=face_image_paths[(kid, fi)],
                extrinsics=face_T_cw[(kid, fi)],
                intrinsics=K_face,
                camera_id=1,
                width=FACE_SIZE,
                height=FACE_SIZE,
                qvec=np.zeros(4),  # unused by DA3Runner
                tvec=np.zeros(3),
            )
            entry_map[(kid, fi)] = img_id
            all_face_entries.append(fe)
            img_id += 1

    # Run inference in windows
    window_size = args.window_size
    overlap = max(0, window_size - 1)  # ensure progress

    log.info(
        "Running DA3 on %d face images (window=%d, overlap=%d) …",
        len(all_face_entries), window_size, overlap,
    )

    # Stores (d_metric, sky_mask) per (kf_id, face_id).
    # sky_mask is True where pixels ARE sky (depth should be zeroed).
    raw_depths: Dict[Tuple[int, int], tuple] = {}
    step = max(1, window_size - overlap)

    for w_start in tqdm(
        range(0, len(all_face_entries), step),
        desc="Phase1.5 DA3",
        dynamic_ncols=True,
        leave=True,
    ):
        window = all_face_entries[w_start: w_start + window_size]
        if not window:
            break

        depth_batch, conf_batch, sky_batch = runner.infer_faces(
            window, pose_condition=False
        )
        for idx, fe in enumerate(window):
            d_raw = depth_batch[idx]  # [H, W] raw network depth
            d_metric = runner.to_metric_face_depth(d_raw, K_face)

            # sky_batch: bool [N,H,W] where True = sky
            # (DA3 OutputProcessor: sky_logits >= 0.3). Zero those pixels.
            if sky_batch is not None:
                sky_mask = sky_batch[idx].astype(bool)
            else:
                sky_mask = np.zeros(d_metric.shape, dtype=bool)

            raw_depths[(
                int(fe.frame_id.replace("kf", "")),
                fe.face_id,
            )] = (d_metric, sky_mask)

        if (w_start // step) % 10 == 0:
            log.info(
                "  window %d/%d processed", w_start // step + 1,
                (len(all_face_entries) + step - 1) // step,
            )

    log.info("DA3 inference done. Applying scale alignment …")

    # --- Scale alignment + save ---
    records: List[dict] = []

    for kf in kf_meta:
        kid = kf["id"]
        T_cw_pano = kf["T_cw"]
        # Camera-to-world pose of pano camera (for FARM)
        R_cw = T_cw_pano[:3, :3]
        t_cw = T_cw_pano[:3, 3]
        R_wc = R_cw.T
        t_wc = R_wc @ (-t_cw)  # = -R_cw.T @ t_cw = camera centre in world

        lm_ids = kf_lm_ids.get(kid, np.array([], dtype=np.int32))

        for fi, R_face in enumerate(R_faces):
            if (kid, fi) not in raw_depths:
                continue
            d_metric, sky_mask = raw_depths[(kid, fi)]
            T_cw_face = face_T_cw[(kid, fi)]

            # Scale alignment (use only non-sky pixels as anchors implicitly —
            # landmarks in sky are already filtered by z>0 check in fit_scale_alpha)
            alpha = fit_scale_alpha(
                d_metric, T_cw_face, K_face, landmarks, lm_ids
            )
            d_aligned = d_metric * alpha

            # Zero out sky pixels so FARM doesn't fuse garbage far-depth sky points
            d_aligned[sky_mask] = 0.0

            # Save depth .npy
            depth_fname = f"kf{kid:06d}_face{fi}.npy"
            depth_path = depth_dir / depth_fname
            np.save(str(depth_path), d_aligned.astype(np.float32))

            # Save depth visualisation (Turbo); zeroed/sky pixels drawn grey
            valid = d_aligned[d_aligned > 0]
            if valid.size > 0:
                vmin, vmax = float(np.percentile(valid, 2)), float(np.percentile(valid, 98))
                d_norm = np.clip((d_aligned - vmin) / max(vmax - vmin, 1e-6), 0, 1)
                vis = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                vis[d_aligned <= 0] = (40, 40, 40)
                cv2.imwrite(str(vis_dir / f"kf{kid:06d}_face{fi}.jpg"), vis)

            # Copy face image to frames_json/images
            src_img = face_image_paths[(kid, fi)]
            dst_img = frames_dir / src_img.name
            import shutil
            shutil.copy2(src_img, dst_img)

            # Copy depth to frames_json/depths
            import shutil as sh
            sh.copy2(depth_path, depths_out_dir / depth_fname)

            # Camera-to-world for face (FARM convention)
            R_wc_face = R_wc @ R_face.T
            T_wc_face = np.eye(4, dtype=np.float64)
            T_wc_face[:3, :3] = R_wc_face
            T_wc_face[:3, 3] = t_wc  # same camera centre for all faces

            records.append({
                "ts": kf["ts"] + fi * 1e-6,   # tiny offset to keep unique
                "image_path": str(frames_dir / src_img.name),
                "depth_path": str(depths_out_dir / depth_fname),
                "K": K_face.tolist(),
                "T_wc": T_wc_face.tolist(),
                "kf_id": kid,
                "face_id": fi,
                "scale_alpha": alpha,
            })

    log.info("Saving FARM frames-json …")
    manifest = write_frames_json(out_root, records)

    # --- Summary ---
    alphas = [r["scale_alpha"] for r in records]
    log.info(
        "\n=== DONE ===\n"
        "  Output dir : %s\n"
        "  Frames     : %d (%d KFs × %d faces)\n"
        "  Scale alpha: med=%.4f std=%.4f (1.0 = no correction needed)\n"
        "  frames.json: %s\n",
        out_root, len(records), len(kf_meta), NUM_FACES,
        float(np.median(alphas)), float(np.std(alphas)),
        manifest,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=str(STELLA_DB) if STELLA_DB else None,
                   help="Path to Stella out.db")
    p.add_argument("--kf-image-dir", default=str(KF_IMAGE_DIR) if KF_IMAGE_DIR else None,
                   help="Keyframe JPEG/PNG dir")
    p.add_argument("--kf-traj", default=str(KF_TRAJ) if KF_TRAJ else None,
                   help="TUM keyframe trajectory file")
    p.add_argument("--out-dir", default=str(OUT_ROOT), help="Output root directory")
    p.add_argument("--max-kfs", type=int, default=None, help="Limit KFs (smoke test)")
    p.add_argument("--window-size", type=int, default=4, help="DA3 inference window (frames per batch)")
    p.add_argument("--model", default="depth-anything/DA3METRIC-LARGE", help="DA3 model name or local path")
    p.add_argument("--cpu", action="store_true", help="Force CPU (very slow)")
    args = p.parse_args()
    if not args.db or not args.kf_image_dir or not args.kf_traj:
        p.error("--db, --kf-image-dir, and --kf-traj are required (or set STELLA_* env vars)")
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
