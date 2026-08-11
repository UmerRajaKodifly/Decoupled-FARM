#!/usr/bin/env python3
"""Overlay map landmarks onto keyframe images (equirectangular).

Visual pose check without GT:
  - green cross  = projected 3D landmark (using keyframe pose)
  - red  circle  = observed 2D keypoint associated with that landmark
  - yellow line  = residual; short = good pose/obs consistency

Also computes median / mean reprojection error in pixels for associated points.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np


def load_pose_cw(blob: bytes) -> np.ndarray:
    """4x4 camera-from-world (column-major float64 blob in DB)."""
    return np.frombuffer(blob, np.float64).reshape((4, 4)).T.copy()


def load_keypoints(blob: bytes, n: int) -> np.ndarray:
    """(N,2) float32 xy; storage is 7 floats per keypoint."""
    arr = np.frombuffer(blob, np.float32).reshape(n, 7)
    return arr[:, :2].copy()


def load_lm_ids(blob: bytes, n: int) -> np.ndarray:
    return np.frombuffer(blob, np.int32).reshape(n).copy()


def load_landmarks(conn: sqlite3.Connection) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for lid, pos_b in conn.execute("SELECT id, pos_w FROM landmarks"):
        out[int(lid)] = np.frombuffer(pos_b, np.float64).copy()
    return out


def equirect_project(pos_c: np.ndarray, cols: int, rows: int) -> Tuple[float, float]:
    """Camera-frame 3D → equirect pixel (same as stella equirectangular)."""
    n = np.linalg.norm(pos_c)
    if n < 1e-12:
        return float("nan"), float("nan")
    b = pos_c / n
    # clamp for asin
    y = float(np.clip(b[1], -1.0, 1.0))
    lat = -np.arcsin(y)
    lon = np.arctan2(b[0], b[2])
    u = cols * (0.5 + lon / (2.0 * np.pi))
    v = rows * (0.5 - lat / np.pi)
    return float(u), float(v)


def project_world(T_cw: np.ndarray, pos_w: np.ndarray, cols: int, rows: int) -> Tuple[float, float, float]:
    """Return (u, v, depth_along_ray_approx). Depth = ||pos_c||."""
    R = T_cw[:3, :3]
    t = T_cw[:3, 3]
    pos_c = R @ pos_w + t
    depth = float(np.linalg.norm(pos_c))
    # behind camera hemisphere is still valid for equirect (full sphere)
    u, v = equirect_project(pos_c, cols, rows)
    return u, v, depth


def draw_overlay(
    img_bgr: np.ndarray,
    observed_xy: np.ndarray,
    projected_uv: np.ndarray,
    max_draw: int = 800,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Draw residuals; return vis image and error stats (in pixels)."""
    h, w = img_bgr.shape[:2]
    errs = np.linalg.norm(observed_xy - projected_uv, axis=1)
    stats = {
        "n": float(len(errs)),
        "mean_px": float(errs.mean()) if len(errs) else float("nan"),
        "median_px": float(np.median(errs)) if len(errs) else float("nan"),
        "p95_px": float(np.percentile(errs, 95)) if len(errs) else float("nan"),
        "max_px": float(errs.max()) if len(errs) else float("nan"),
    }

    vis = img_bgr.copy()
    n = len(errs)
    if n == 0:
        return vis, stats

    rng = np.random.default_rng(seed)
    if n > max_draw:
        idx = rng.choice(n, size=max_draw, replace=False)
    else:
        idx = np.arange(n)

    for i in idx:
        ox, oy = observed_xy[i]
        px, py = projected_uv[i]
        if not (np.isfinite(ox) and np.isfinite(oy) and np.isfinite(px) and np.isfinite(py)):
            continue
        # wrap u if needed for equirect residual drawing (simple clamp for view)
        oi = (int(round(ox)), int(round(oy)))
        pi = (int(round(px)), int(round(py)))
        if 0 <= oi[0] < w and 0 <= oi[1] < h and 0 <= pi[0] < w and 0 <= pi[1] < h:
            cv.line(vis, oi, pi, (0, 255, 255), 1, cv.LINE_AA)
            cv.circle(vis, oi, 3, (0, 0, 255), 1, cv.LINE_AA)  # observed red
            cv.drawMarker(vis, pi, (0, 255, 0), cv.MARKER_CROSS, 8, 1, cv.LINE_AA)  # projected green

    # legend
    cv.rectangle(vis, (8, 8), (420, 88), (0, 0, 0), -1)
    cv.putText(vis, "red=observed keypoint  green=projected 3D", (16, 32), cv.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv.LINE_AA)
    cv.putText(
        vis,
        f"n={int(stats['n'])}  median={stats['median_px']:.1f}px  p95={stats['p95_px']:.1f}px",
        (16, 56),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv.LINE_AA,
    )
    cv.putText(vis, "short yellow lines = good pose/obs match", (16, 76), cv.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv.LINE_AA)
    return vis, stats


def select_keyframe_ids(all_ids: List[int], count: int) -> List[int]:
    if count <= 0 or not all_ids:
        return []
    if len(all_ids) <= count:
        return list(all_ids)
    # evenly spaced indices
    idxs = np.linspace(0, len(all_ids) - 1, count)
    return [all_ids[int(round(i))] for i in idxs]


def process_db(
    db_path: Path,
    out_dir: Path,
    num_frames: int = 6,
    max_draw: int = 800,
    keyframe_ids: Optional[List[int]] = None,
    image_dir: Optional[Path] = None,
) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cam = conn.execute("SELECT cols, rows FROM cameras LIMIT 1").fetchone()
    cols, rows = int(cam[0]), int(cam[1])

    landmarks = load_landmarks(conn)
    all_ids = [int(r[0]) for r in conn.execute("SELECT id FROM keyframes ORDER BY id")]
    ids = keyframe_ids if keyframe_ids is not None else select_keyframe_ids(all_ids, num_frames)

    report_lines: List[str] = []
    report_lines.append(f"db: {db_path}")
    report_lines.append(f"camera equirect {cols}x{rows}")
    report_lines.append(f"landmarks: {len(landmarks)}")
    report_lines.append(f"keyframes selected: {ids}")
    report_lines.append("")
    report_lines.append("id\tn_assoc\tmedian_px\tmean_px\tp95_px\tmax_px\tout_png")

    summary_medians: List[float] = []

    for kid in ids:
        row = conn.execute(
            "SELECT pose_cw, n_keypts, undist_keypts, image FROM keyframes WHERE id=?",
            (kid,),
        ).fetchone()
        if row is None:
            report_lines.append(f"{kid}\tMISSING_KF")
            continue
        pose_b, n_keypts, uk_b, img_b = row
        n_keypts = int(n_keypts)
        assoc = conn.execute("SELECT lm_ids FROM associations WHERE id=?", (kid,)).fetchone()
        if assoc is None:
            report_lines.append(f"{kid}\tMISSING_ASSOC")
            continue

        T_cw = load_pose_cw(pose_b)
        kpts = load_keypoints(uk_b, n_keypts)
        lm_ids = load_lm_ids(assoc[0], n_keypts)

        # prefer on-disk keyframe PNG if provided (same resolution)
        img = None
        if image_dir is not None:
            p = image_dir / f"image{kid}.png"
            if p.is_file():
                img = cv.imread(str(p), cv.IMREAD_COLOR)
        if img is None:
            img = cv.imdecode(np.frombuffer(img_b, np.uint8), cv.IMREAD_COLOR)
        if img is None:
            report_lines.append(f"{kid}\tDECODE_FAIL")
            continue

        h, w = img.shape[:2]
        # if keyframe image size differs from camera cols/rows, project into camera size then scale
        scale_x = w / float(cols)
        scale_y = h / float(rows)

        obs_list = []
        proj_list = []
        for i, lid in enumerate(lm_ids):
            if lid < 0:
                continue
            if lid not in landmarks:
                continue
            pos_w = landmarks[lid]
            u, v, _depth = project_world(T_cw, pos_w, cols, rows)
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            # residual uses image coordinates of observed kpt (already in image px of saved frame)
            # projected coords: map from camera resolution to saved image size
            pu, pv = u * scale_x, v * scale_y
            ox, oy = float(kpts[i, 0]) * scale_x, float(kpts[i, 1]) * scale_y
            # handle simple u wrap for equirect (shortest residual)
            du = pu - ox
            if du > w * 0.5:
                pu -= w
            elif du < -w * 0.5:
                pu += w
            obs_list.append([ox, oy])
            proj_list.append([pu, pv])

        if not obs_list:
            report_lines.append(f"{kid}\t0\tnan\tnan\tnan\tnan\t-")
            continue

        observed = np.asarray(obs_list, dtype=np.float64)
        projected = np.asarray(proj_list, dtype=np.float64)
        vis, stats = draw_overlay(img, observed, projected, max_draw=max_draw, seed=kid)
        out_png = out_dir / f"reproj_kf{kid:04d}.png"
        cv.imwrite(str(out_png), vis)
        summary_medians.append(stats["median_px"])
        report_lines.append(
            f"{kid}\t{int(stats['n'])}\t{stats['median_px']:.3f}\t{stats['mean_px']:.3f}\t"
            f"{stats['p95_px']:.3f}\t{stats['max_px']:.3f}\t{out_png.name}"
        )

    report_lines.append("")
    if summary_medians:
        report_lines.append(
            f"OVERALL median of per-keyframe median errors: {float(np.median(summary_medians)):.3f} px"
        )
        report_lines.append(
            "Guide: few px (≲ a few pixels at 1920-wide equirect) = tight observation model match; "
            "tens of px or huge scatter = pose/landmark inconsistency for that frame."
        )
    report_lines.append("Legend: red=observed ORB keypoint, green=3D landmark reprojected with that keyframe pose.")
    text = "\n".join(report_lines) + "\n"
    (out_dir / "reproj_report.txt").write_text(text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproject map landmarks onto keyframe images")
    ap.add_argument("--db", type=Path, required=True, help="stella out.db (sqlite)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory for overlay PNGs + report")
    ap.add_argument("--num-frames", type=int, default=6, help="Number of evenly spaced keyframes")
    ap.add_argument("--keyframe-ids", type=int, nargs="*", default=None, help="Optional explicit KF ids")
    ap.add_argument("--max-draw", type=int, default=800, help="Max associations drawn per image")
    ap.add_argument("--image-dir", type=Path, default=None, help="Optional keyframes/ folder with imageN.png")
    args = ap.parse_args()

    report = process_db(
        args.db,
        args.out_dir,
        num_frames=args.num_frames,
        max_draw=args.max_draw,
        keyframe_ids=args.keyframe_ids,
        image_dir=args.image_dir,
    )
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
