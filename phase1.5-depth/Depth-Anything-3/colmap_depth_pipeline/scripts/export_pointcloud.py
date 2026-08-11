#!/usr/bin/env python3
"""Export pointcloud.ply from face depths + COLMAP intrinsics and poses."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_REPO = _ROOT.parent
for p in (_SRC, _REPO / "src", _REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pointcloud_export import (  # noqa: E402
    collect_scaled_camera_centers,
    export_face_pointcloud,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--every_n", type=int, default=1)
    parser.add_argument("--max_depth", type=float, default=0.0, help="0 = no depth trunc")
    parser.add_argument("--min_conf", type=float, default=0.0)
    parser.add_argument("--max_points", type=int, default=5_000_000)
    parser.add_argument("--no_mask_sky", action="store_true")
    parser.add_argument("--pose_scale", type=float, default=None, help="Override manifest alpha")
    parser.add_argument("--no_cameras", action="store_true")
    parser.add_argument("--camera_radius", type=float, default=0.08)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    max_depth = None if args.max_depth <= 0 else args.max_depth
    output = args.output or (args.out_dir / "pointcloud.ply")

    path = export_face_pointcloud(
        args.colmap_dir,
        args.out_dir,
        output,
        stride=args.stride,
        every_n=args.every_n,
        max_depth=max_depth,
        min_conf=args.min_conf,
        max_points=args.max_points,
        mask_sky=not args.no_mask_sky,
        pose_scale=args.pose_scale,
        show_cameras=not args.no_cameras,
        camera_sphere_radius=args.camera_radius,
    )
    print(f"Wrote {path}")

    if args.view:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(str(path))
        geoms = [pcd]
        if not args.no_cameras:
            centers, alpha_med = collect_scaled_camera_centers(
                args.colmap_dir,
                args.out_dir,
                every_n=args.every_n,
                pose_scale=args.pose_scale,
            )
            print(
                f"Scaled cameras: {centers.shape[0]} centers, "
                f"alpha≈{alpha_med:.4f}, sphere r={args.camera_radius}"
            )
            for c in centers:
                sph = o3d.geometry.TriangleMesh.create_sphere(radius=args.camera_radius)
                sph.translate(c.tolist())
                sph.paint_uniform_color([1.0, 0.0, 0.85])
                sph.compute_vertex_normals()
                geoms.append(sph)
        o3d.visualization.draw_geometries(geoms, window_name=Path(path).name)


if __name__ == "__main__":
    main()
