#!/usr/bin/env python3
"""Measure Euclidean distance between two picked points on a PLY point cloud.

Open3D controls:
  - Shift + left-click  → pick a point (pick exactly 2)
  - Q / Esc / close window → finish and print distance

COLMAP SfM scale is only metric if the reconstruction was scaled (e.g. known
baseline). Otherwise the number is in COLMAP units — still useful to compare
known real-world lengths (tape measure) vs cloud distance (ratio = scale error).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def measure(ply_path: Path, known_meters: float | None = None) -> None:
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        raise SystemExit(f"Empty or unreadable point cloud: {ply_path}")

    print(f"Loaded {ply_path}  ({np.asarray(pcd.points).shape[0]:,} points)")
    print()
    print("Instructions:")
    print("  1. Hold Shift and left-click to pick point A")
    print("  2. Hold Shift and left-click to pick point B")
    print("  3. Close the window (Q) to print the distance")
    print("  Tip: scroll = zoom, left-drag = orbit, Ctrl+left-drag = pan")
    print()

    # VisualizerWithEditing: Shift+LMB picks; closing returns picked point indices
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=f"measure — {ply_path.name}", width=1280, height=800)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.08, 0.08, 0.1])
    vis.run()  # blocks until window closed
    vis.destroy_window()

    idxs = vis.get_picked_points()
    pts = np.asarray(pcd.points)
    if len(idxs) < 2:
        raise SystemExit(
            f"Need 2 picked points (got {len(idxs)}). "
            "Shift+left-click twice, then close the window."
        )
    if len(idxs) > 2:
        print(f"Note: {len(idxs)} points picked; using the first two.")

    a = pts[idxs[0]]
    b = pts[idxs[1]]
    dist = float(np.linalg.norm(a - b))

    print("=" * 50)
    print(f"Point A (idx {idxs[0]}): [{a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f}]")
    print(f"Point B (idx {idxs[1]}): [{b[0]:.4f}, {b[1]:.4f}, {b[2]:.4f}]")
    print(f"Distance: {dist:.4f} m  (reconstruction units; expect meters with DA3METRIC)")
    if known_meters is not None and known_meters > 0:
        scale = known_meters / dist
        print(f"Known real length: {known_meters:.4f} m")
        print(f"Implied scale factor (real / cloud): {scale:.4f}")
        if abs(scale - 1.0) < 0.05:
            print("  → scale looks metric (within ~5%)")
        else:
            print(f"  → multiply cloud coords by {scale:.4f} to match meters")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ply",
        type=Path,
        nargs="?",
        default=Path("vkip3046/depth_v2/pointcloud_faces.ply"),
        help="Path to point cloud PLY",
    )
    parser.add_argument(
        "--known_m",
        type=float,
        default=None,
        help="Optional real-world length in meters between the two points "
        "(tape measure). Prints cloud→meter scale factor.",
    )
    args = parser.parse_args()
    if not args.ply.is_file():
        raise SystemExit(f"File not found: {args.ply}")
    measure(args.ply, known_meters=args.known_m)


if __name__ == "__main__":
    main()
