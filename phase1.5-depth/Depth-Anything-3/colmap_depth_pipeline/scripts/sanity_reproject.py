#!/usr/bin/env python3
"""Sanity-check: print one frame's faces and reproject a 3D point (<1px)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from colmap_io import load_colmap_model, reproject_point_to_face  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--frame_id", type=str, default=None)
    parser.add_argument("--max_error_px", type=float, default=1.0)
    args = parser.parse_args()

    model = load_colmap_model(args.colmap_dir, expected_faces=4)
    frame_id = args.frame_id or model.frame_ids[0]
    faces = model.frames[frame_id]
    print(f"Frame: {frame_id}  ({len(faces)} faces)")
    for f in faces:
        print(
            f"  face{f.face_id}: {f.image_name}\n"
            f"    path={f.image_path}\n"
            f"    K=\n{f.intrinsics}\n"
            f"    w2c t={f.extrinsics[:3, 3]}"
        )

    # Find a 3D point observed in one of these faces
    best = None
    for face in faces:
        image = model.images[face.image_id]
        for xy, p3d_id in zip(image.xys, image.point3D_ids):
            if p3d_id < 0:
                continue
            p3d = model.points3D.get(p3d_id)
            if p3d is None:
                continue
            uv_pred, z = reproject_point_to_face(p3d.xyz, face.extrinsics, face.intrinsics)
            err = float(np.linalg.norm(uv_pred - np.asarray(xy)))
            best = (face, xy, uv_pred, err, z, p3d_id)
            break
        if best is not None:
            break

    if best is None:
        print("No 3D observations found on this frame's faces.")
        sys.exit(2)

    face, xy, uv_pred, err, z, p3d_id = best
    print(
        f"Reprojection check face{face.face_id} point3D={p3d_id}:\n"
        f"  observed={xy}  predicted={uv_pred}  error={err:.4f}px  z={z:.4f}"
    )
    if err > args.max_error_px:
        print(f"FAIL: error {err:.4f} > {args.max_error_px}")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
