#!/usr/bin/env python3
"""DA3-only baseline: same cube-face images, no COLMAP poses / scale.

Runs pose-free DA3 on sliding windows of the 4-face images, stitches windows
via Umeyama on the overlapping cameras, and writes a colored PLY for comparison
against the COLMAP-conditioned pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_REPO = _ROOT.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_DA3 = _REPO / "src"
if str(_DA3) not in sys.path:
    sys.path.insert(0, str(_DA3))

from colmap_io import load_colmap_model  # noqa: E402
from da3_runner import DA3Runner, iter_frame_windows  # noqa: E402

logger = logging.getLogger(__name__)


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


def _as_44(ext: np.ndarray) -> np.ndarray:
    ext = np.asarray(ext, dtype=np.float64)
    if ext.shape == (4, 4):
        return ext
    if ext.shape == (3, 4):
        H = np.eye(4)
        H[:3, :4] = ext
        return H
    raise ValueError(f"Bad extrinsics shape {ext.shape}")


def _w2c_centers(exts: np.ndarray) -> np.ndarray:
    out = []
    for e in exts:
        E = _as_44(e)
        R, t = E[:3, :3], E[:3, 3]
        out.append(-R.T @ t)
    return np.stack(out, axis=0)


def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Sim(3): dst ≈ s * R @ src + t. src/dst (N,3)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape and src.shape[0] >= 3
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    xs = src - mu_s
    xd = dst - mu_d
    var_s = (xs**2).sum() / src.shape[0]
    cov = (xd.T @ xs) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / max(var_s, 1e-12)
    t = mu_d - s * R @ mu_s
    return float(s), R, t


def apply_sim3(points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (s * (points.astype(np.float64) @ R.T)) + t


def unproject_views(
    depth: np.ndarray,
    K: np.ndarray,
    ext_w2c: np.ndarray,
    images_rgb: list[np.ndarray],
    *,
    stride: int,
    conf: np.ndarray | None,
    conf_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    N = depth.shape[0]
    conf_thr = 0.0
    if conf is not None:
        conf_thr = float(np.percentile(conf, conf_percentile))

    all_pts, all_cols = [], []
    for i in range(N):
        d = depth[i]
        H, W = d.shape
        rgb = images_rgb[i]
        if rgb.shape[0] != H or rgb.shape[1] != W:
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)

        ys = np.arange(0, H, stride)
        xs = np.arange(0, W, stride)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        z = d[yy, xx]
        valid = np.isfinite(z) & (z > 1e-6)
        if conf is not None:
            valid &= conf[i][yy, xx] >= conf_thr
        if not np.any(valid):
            continue

        u = xx[valid].astype(np.float64)
        v = yy[valid].astype(np.float64)
        z_v = z[valid].astype(np.float64)
        Ki = np.asarray(K[i], dtype=np.float64)
        fx, fy, cx, cy = Ki[0, 0], Ki[1, 1], Ki[0, 2], Ki[1, 2]
        x = (u - cx) / fx * z_v
        y = (v - cy) / fy * z_v
        Xc = np.stack([x, y, z_v], axis=1)
        E = _as_44(ext_w2c[i])
        R, t = E[:3, :3], E[:3, 3]
        Xw = (Xc - t) @ R
        cols = rgb[yy, xx][valid]
        all_pts.append(Xw.astype(np.float32))
        all_cols.append(cols.astype(np.uint8))

    if not all_pts:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    return np.concatenate(all_pts, 0), np.concatenate(all_cols, 0)


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _unique_frame_centers(
    keys: list[tuple[str, int]],
    centers: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """One camera center per frame_id (prefer face 0)."""
    chosen: dict[str, np.ndarray] = {}
    for (fid, face_id), c in zip(keys, centers):
        if fid not in chosen or face_id == 0:
            chosen[fid] = c
    fids = list(chosen.keys())
    return fids, np.stack([chosen[f] for f in fids], axis=0)


def run_baseline(
    colmap_dir: Path,
    out_dir: Path,
    *,
    model_name: str,
    window_size: int,
    overlap: int,
    stride: int,
    max_points: int,
    conf_percentile: float,
    device: str | None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = out_dir / "face_depth"
    depth_dir.mkdir(exist_ok=True)

    # Face images come from the COLMAP project layout; poses are NOT used.
    model = load_colmap_model(colmap_dir, expected_faces=4)
    frame_ids = model.frame_ids
    windows = iter_frame_windows(frame_ids, window_size, overlap)
    logger.info(
        "DA3-only baseline: %d frames, %d windows (size=%d overlap=%d)",
        len(frame_ids),
        len(windows),
        window_size,
        overlap,
    )

    runner = DA3Runner(model_name=model_name, device=device)

    world_pts: list[np.ndarray] = []
    world_cols: list[np.ndarray] = []
    prev_keys: list[tuple[str, int]] | None = None
    prev_frame_ids: list[str] | None = None
    prev_uniq_centers: np.ndarray | None = None
    # cumulative Sim3 into first-window frame: X_world = s * R @ X_local + t
    cum_s, cum_R, cum_t = 1.0, np.eye(3), np.zeros(3)

    manifest_windows = []

    for wi, w_frames in enumerate(windows):
        faces = []
        keys = []
        for fid in w_frames:
            for face in model.frames[fid]:
                faces.append(face)
                keys.append((fid, face.face_id))

        image_paths = [str(f.image_path) for f in faces]
        logger.info("Window %d/%d: %d faces (no COLMAP poses)", wi + 1, len(windows), len(faces))

        prediction = runner.model.inference(
            image=image_paths,
            extrinsics=None,
            intrinsics=None,
            align_to_input_ext_scale=False,
            process_res=runner.process_res,
            ref_view_strategy=runner.ref_view_strategy,
        )
        depth = np.asarray(prediction.depth, dtype=np.float32)
        conf = np.asarray(prediction.conf, dtype=np.float32) if prediction.conf is not None else None
        ext = np.asarray(prediction.extrinsics)
        K = np.asarray(prediction.intrinsics)

        # Save raw depths for inspection
        for i, (fid, face_id) in enumerate(keys):
            stem = Path(fid).stem
            np.save(depth_dir / f"{stem}_face{face_id}_w{wi:02d}.npy", depth[i])

        rgbs = [load_rgb(Path(p)) for p in image_paths]
        centers = _w2c_centers(ext)
        frame_ids_w, uniq_centers = _unique_frame_centers(keys, centers)

        if prev_frame_ids is not None and prev_uniq_centers is not None:
            shared = [f for f in frame_ids_w if f in set(prev_frame_ids)]
            if len(shared) >= 3:
                src = np.stack(
                    [uniq_centers[frame_ids_w.index(f)] for f in shared], axis=0
                )
                dst = np.stack(
                    [prev_uniq_centers[prev_frame_ids.index(f)] for f in shared],
                    axis=0,
                )
                s, R, t = umeyama_sim3(src, dst)
                cum_s, cum_R, cum_t = (
                    cum_s * s,
                    cum_R @ R,
                    cum_s * (cum_R @ t) + cum_t,
                )
                logger.info(
                    "  stitch overlap=%d frames  s=%.4f cum_s=%.4f",
                    len(shared),
                    s,
                    cum_s,
                )
            else:
                logger.warning(
                    "  only %d shared frames (need >=3); keeping previous Sim3",
                    len(shared),
                )

        # Prefer only newly seen faces so overlap isn't double-counted.
        if prev_keys is not None:
            prev_set = set(prev_keys)
            keep_idx = [i for i, k in enumerate(keys) if k not in prev_set]
        else:
            keep_idx = list(range(len(keys)))

        n_points = 0
        if keep_idx:
            keep = np.asarray(keep_idx, dtype=np.int64)
            pts_local, cols = unproject_views(
                depth[keep],
                K[keep],
                ext[keep],
                [rgbs[i] for i in keep_idx],
                stride=stride,
                conf=conf[keep] if conf is not None else None,
                conf_percentile=conf_percentile,
            )
            pts_w = apply_sim3(pts_local, cum_s, cum_R, cum_t).astype(np.float32)
            n_points = int(pts_w.shape[0])
            if n_points:
                world_pts.append(pts_w)
                world_cols.append(cols)

        prev_keys = keys
        prev_frame_ids = frame_ids_w
        prev_uniq_centers = uniq_centers

        manifest_windows.append(
            {
                "window": wi,
                "frames": w_frames,
                "n_faces": len(faces),
                "n_points": n_points,
                "cum_scale": cum_s,
            }
        )

    if not world_pts:
        raise RuntimeError("No points from DA3-only baseline")

    points = np.concatenate(world_pts, 0)
    colors = np.concatenate(world_cols, 0)
    if points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points, colors = points[idx], colors[idx]

    ply_path = out_dir / "pointcloud_da3_only.ply"
    write_ply_binary(ply_path, points, colors)
    meta = {
        "mode": "da3_only_no_colmap_poses",
        "colmap_dir_images_only": str(colmap_dir),
        "model_name": model_name,
        "window_size": window_size,
        "overlap": overlap,
        "n_points": int(points.shape[0]),
        "windows": manifest_windows,
        "note": (
            "Cube-face images reused from COLMAP project layout; "
            "DA3 ran with extrinsics=None/intrinsics=None. "
            "Windows stitched by Umeyama on overlapping camera centers."
        ),
    }
    (out_dir / "manifest_da3_only.json").write_text(json.dumps(meta, indent=2))
    logger.info("Wrote %s (%s points)", ply_path, f"{points.shape[0]:,}")
    return ply_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--colmap_dir",
        type=Path,
        required=True,
        help="Project with images/pano_camera*/ (poses ignored)",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="depth-anything/DA3-LARGE-1.1")
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument(
        "--overlap",
        type=int,
        default=3,
        help="Frame overlap for window stitch (need >=3 unique centers)",
    )
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max_points", type=int, default=5_000_000)
    parser.add_argument("--conf_percentile", type=float, default=40.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_baseline(
        args.colmap_dir,
        args.out_dir,
        model_name=args.model_name,
        window_size=args.window_size,
        overlap=args.overlap,
        stride=args.stride,
        max_points=args.max_points,
        conf_percentile=args.conf_percentile,
        device=args.device,
    )


if __name__ == "__main__":
    main()
