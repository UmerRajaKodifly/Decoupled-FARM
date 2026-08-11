"""World-space point cloud from face depths + COLMAP intrinsics and poses."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from colmap_io import load_colmap_model
from io_utils import (
    load_depth,
    resolve_face_conf_path,
    resolve_face_depth_path,
    resolve_face_sky_path,
)
from scale_align import camera_center_from_w2c, scale_extrinsics

logger = logging.getLogger(__name__)

_CAMERA_SPHERE_RGB = np.array([255, 0, 220], dtype=np.uint8)


def write_ply_binary(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(points.shape[0])
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    verts = np.empty(
        n,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    verts["x"] = points[:, 0].astype(np.float32)
    verts["y"] = points[:, 1].astype(np.float32)
    verts["z"] = points[:, 2].astype(np.float32)
    verts["red"] = colors[:, 0]
    verts["green"] = colors[:, 1]
    verts["blue"] = colors[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(verts.tobytes())


def load_manifest_scale(out_dir: Path) -> tuple[float, dict[str, float]]:
    """Return (global_alpha, per_frame_alpha)."""
    man = out_dir / "manifest.json"
    per_frame: dict[str, float] = {}
    global_alpha = 1.0
    if not man.is_file():
        return global_alpha, per_frame
    data = json.loads(man.read_text())
    if "alpha" in data:
        global_alpha = float(data["alpha"])
    for fr in data.get("frames", []):
        if "alpha" in fr:
            per_frame[fr["frame_id"]] = float(fr["alpha"])
    for fr in data.get("scale_details", []):
        if "alpha" in fr:
            per_frame[fr["frame_id"]] = float(fr["alpha"])
    return global_alpha, per_frame


def sky_mask_hsv(
    rgb: np.ndarray,
    *,
    v_min: float = 160.0,
    s_max: float = 80.0,
    top_fraction: float = 0.65,
    dilate: int = 5,
) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    bright_desat = (v >= v_min) & (s <= s_max)
    blueish = (h >= 85.0) & (h <= 135.0) & (v >= 110.0) & (s <= 130.0)
    sky = bright_desat | blueish
    H = sky.shape[0]
    row_lim = int(np.clip(top_fraction, 0.0, 1.0) * H)
    upper = np.zeros_like(sky, dtype=bool)
    upper[:row_lim, :] = True
    sky &= upper
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        sky = cv2.dilate(sky.astype(np.uint8), k, iterations=1).astype(bool)
    return sky


def sample_sphere_points(
    center: np.ndarray,
    radius: float,
    *,
    n_points: int = 256,
) -> np.ndarray:
    """Fibonacci-lattice points on a sphere of fixed radius around ``center``."""
    center = np.asarray(center, dtype=np.float64).reshape(3)
    i = np.arange(n_points, dtype=np.float64)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (2.0 * i + 1.0) / n_points
    r = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    theta = i * phi
    pts = np.stack([np.cos(theta) * r, y, np.sin(theta) * r], axis=1)
    return (center + radius * pts).astype(np.float32)


def collect_scaled_camera_centers(
    colmap_dir: Path,
    out_dir: Path,
    *,
    every_n: int = 1,
    pose_scale: float | None = None,
) -> tuple[np.ndarray, float]:
    """One camera center per frame after alpha scaling (rig faces share a center)."""
    model = load_colmap_model(colmap_dir, expected_faces=4)
    global_alpha, per_frame = load_manifest_scale(out_dir)
    frame_ids = model.frame_ids[:: max(every_n, 1)]
    centers = []
    alphas = []
    for frame_id in frame_ids:
        faces = model.frames.get(frame_id) or []
        if not faces:
            continue
        alpha = pose_scale if pose_scale is not None else per_frame.get(frame_id, global_alpha)
        face = next((f for f in faces if f.face_id == 0), faces[0])
        w2c = scale_extrinsics(face.extrinsics, alpha)
        centers.append(camera_center_from_w2c(w2c))
        alphas.append(alpha)
    if not centers:
        return np.zeros((0, 3), dtype=np.float64), float(global_alpha)
    return np.stack(centers, axis=0), float(np.median(alphas))


def camera_spheres_pointcloud(
    centers: np.ndarray,
    *,
    radius: float = 0.08,
    n_points_per_sphere: int = 256,
    rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-size spheres at scaled camera centers (points + colors)."""
    if centers.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    color = np.asarray(rgb if rgb is not None else _CAMERA_SPHERE_RGB, dtype=np.uint8)
    pts_list = []
    col_list = []
    for c in centers:
        sp = sample_sphere_points(c, radius, n_points=n_points_per_sphere)
        pts_list.append(sp)
        col_list.append(np.tile(color, (sp.shape[0], 1)))
    return np.concatenate(pts_list, axis=0), np.concatenate(col_list, axis=0)


def unproject_face(
    depth_planar: np.ndarray,
    K: np.ndarray,
    w2c: np.ndarray,
    rgb: np.ndarray,
    *,
    stride: int,
    max_depth: float | None,
    conf: np.ndarray | None,
    min_conf: float,
    mask_sky: bool,
    sky_mask: np.ndarray | None,
    sky_v_min: float,
    sky_s_max: float,
    sky_top_fraction: float,
    sky_dilate: int,
) -> tuple[np.ndarray, np.ndarray]:
    H, W = depth_planar.shape
    if rgb.shape[0] != H or rgb.shape[1] != W:
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    z = depth_planar[yy, xx]
    valid = np.isfinite(z) & (z > 1e-6)
    if max_depth is not None and max_depth > 0:
        valid &= z <= max_depth
    if conf is not None:
        c = conf
        if c.shape != depth_planar.shape:
            c = cv2.resize(c.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        valid &= c[yy, xx] >= min_conf
    if sky_mask is not None:
        sm = sky_mask
        if sm.shape != depth_planar.shape:
            sm = cv2.resize(sm.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        valid &= ~sm[yy, xx]
    elif mask_sky:
        sky = sky_mask_hsv(
            rgb,
            v_min=sky_v_min,
            s_max=sky_s_max,
            top_fraction=sky_top_fraction,
            dilate=sky_dilate,
        )
        valid &= ~sky[yy, xx]
    if not np.any(valid):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)

    u = xx[valid].astype(np.float64)
    v = yy[valid].astype(np.float64)
    z_v = z[valid].astype(np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u - cx) / fx * z_v
    y = (v - cy) / fy * z_v
    Xc = np.stack([x, y, z_v], axis=1)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    Xw = (Xc - t) @ R
    cols = rgb[yy, xx][valid]
    return Xw.astype(np.float32), cols.astype(np.uint8)


def build_pointcloud(
    colmap_dir: Path,
    out_dir: Path,
    *,
    stride: int = 4,
    every_n: int = 1,
    max_depth: float | None = None,
    min_conf: float = 0.0,
    max_points: int | None = 5_000_000,
    mask_sky: bool = True,
    sky_v_min: float = 160.0,
    sky_s_max: float = 80.0,
    sky_top_fraction: float = 0.65,
    sky_dilate: int = 5,
    pose_scale: float | None = None,
    show_cameras: bool = True,
    camera_sphere_radius: float = 0.08,
    camera_sphere_points: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject face depths with COLMAP K + alpha-scaled COLMAP poses."""
    if max_depth is not None and max_depth <= 0:
        max_depth = None

    model = load_colmap_model(colmap_dir, expected_faces=4)
    global_alpha, per_frame = load_manifest_scale(out_dir)
    face_depth_dir = out_dir / "face_depth"
    face_conf_dir = out_dir / "face_conf"
    face_sky_dir = out_dir / "face_sky"

    frame_ids = model.frame_ids[:: max(every_n, 1)]
    all_pts: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []

    for i, frame_id in enumerate(frame_ids):
        alpha = pose_scale if pose_scale is not None else per_frame.get(frame_id, global_alpha)
        n_frame = 0
        for face in model.frames[frame_id]:
            stem = Path(frame_id).stem
            depth_path = resolve_face_depth_path(face_depth_dir, stem, face.face_id)
            if depth_path is None:
                logger.warning(
                    "skip missing depth %s/face%d/%s.npy",
                    face_depth_dir.name,
                    face.face_id,
                    stem,
                )
                continue
            depth = load_depth(depth_path).astype(np.float64)

            conf = None
            if min_conf > 0:
                cp = resolve_face_conf_path(face_conf_dir, stem, face.face_id)
                if cp is not None:
                    conf = load_depth(cp).astype(np.float64)

            sky = None
            sp = resolve_face_sky_path(face_sky_dir, stem, face.face_id)
            if sp is not None:
                sky = np.load(sp).astype(bool)

            bgr = cv2.imread(str(face.image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            H, W = depth.shape[:2]
            K = np.asarray(face.intrinsics, dtype=np.float64)
            if face.width and face.height and (W != face.width or H != face.height):
                sx = W / float(face.width)
                sy = H / float(face.height)
                K = K.copy()
                K[0, 0] *= sx
                K[1, 1] *= sy
                K[0, 2] *= sx
                K[1, 2] *= sy

            w2c = scale_extrinsics(face.extrinsics, alpha)
            pts, cols = unproject_face(
                depth,
                K,
                w2c,
                rgb,
                stride=stride,
                max_depth=max_depth,
                conf=conf,
                min_conf=min_conf,
                mask_sky=mask_sky,
                sky_mask=sky,
                sky_v_min=sky_v_min,
                sky_s_max=sky_s_max,
                sky_top_fraction=sky_top_fraction,
                sky_dilate=sky_dilate,
            )
            if pts.shape[0] == 0:
                continue
            all_pts.append(pts)
            all_cols.append(cols)
            n_frame += pts.shape[0]

        logger.info(
            "[%d/%d] %s: %s pts  alpha=%.4f",
            i + 1,
            len(frame_ids),
            frame_id,
            f"{n_frame:,}",
            alpha,
        )

    if not all_pts:
        raise RuntimeError("No points generated")

    points = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_cols, axis=0)
    if max_points is not None and points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points, colors = points[idx], colors[idx]
        logger.info("Downsampled to %s points", f"{max_points:,}")

    if show_cameras:
        centers, alpha_med = collect_scaled_camera_centers(
            colmap_dir,
            out_dir,
            every_n=every_n,
            pose_scale=pose_scale,
        )
        cam_pts, cam_cols = camera_spheres_pointcloud(
            centers,
            radius=camera_sphere_radius,
            n_points_per_sphere=camera_sphere_points,
        )
        if cam_pts.shape[0]:
            points = np.concatenate([points, cam_pts], axis=0)
            colors = np.concatenate([colors, cam_cols], axis=0)
            logger.info(
                "Appended %d camera spheres (r=%.3f, alpha≈%.4f) — magenta markers",
                centers.shape[0],
                camera_sphere_radius,
                alpha_med,
            )
    return points, colors


def export_face_pointcloud(
    colmap_dir: Path,
    out_dir: Path,
    output: Path | None = None,
    *,
    stride: int = 4,
    every_n: int = 1,
    max_depth: float | None = None,
    min_conf: float = 0.0,
    max_points: int = 5_000_000,
    mask_sky: bool = True,
    sky_v_min: float = 160.0,
    sky_s_max: float = 80.0,
    sky_top_fraction: float = 0.65,
    sky_dilate: int = 5,
    pose_scale: float | None = None,
    show_cameras: bool = True,
    camera_sphere_radius: float = 0.08,
) -> Path:
    """Write ``pointcloud.ply`` from COLMAP-posed face depths."""
    out_dir = Path(out_dir)
    output = Path(output) if output is not None else out_dir / "pointcloud.ply"
    points, colors = build_pointcloud(
        colmap_dir,
        out_dir,
        stride=stride,
        every_n=every_n,
        max_depth=max_depth,
        min_conf=min_conf,
        max_points=max_points,
        mask_sky=mask_sky,
        sky_v_min=sky_v_min,
        sky_s_max=sky_s_max,
        sky_top_fraction=sky_top_fraction,
        sky_dilate=sky_dilate,
        pose_scale=pose_scale,
        show_cameras=show_cameras,
        camera_sphere_radius=camera_sphere_radius,
    )
    write_ply_binary(output, points, colors)
    logger.info("Wrote %s (%s points)", output, f"{points.shape[0]:,}")
    return Path(output)
