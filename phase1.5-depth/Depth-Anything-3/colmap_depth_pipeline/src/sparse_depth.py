"""Project COLMAP sparse points into per-face sparse depth."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from colmap_io import ColmapModel, FaceEntry


def get_sparse_depth(
    model: ColmapModel,
    face: FaceEntry,
    *,
    min_track_length: int = 3,
    max_reproj_error: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (pixel_coords [M,2], depths [M]) for points observed in this face.

    Depth is planar (camera Z). Filters by track length and reprojection error.
    """
    image = model.images[face.image_id]
    coords = []
    depths = []

    for xy, p3d_id in zip(image.xys, image.point3D_ids):
        if p3d_id < 0:
            continue
        p3d = model.points3D.get(p3d_id)
        if p3d is None:
            continue
        track_len = len(p3d.image_ids)
        if track_len < min_track_length:
            continue
        if p3d.error > max_reproj_error:
            continue

        R = face.extrinsics[:3, :3]
        t = face.extrinsics[:3, 3]
        Xc = R @ np.asarray(p3d.xyz, dtype=np.float64) + t
        z = float(Xc[2])
        if z <= 1e-6:
            continue
        coords.append([float(xy[0]), float(xy[1])])
        depths.append(z)

    if not coords:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    return np.asarray(coords, dtype=np.float64), np.asarray(depths, dtype=np.float64)


def get_sparse_depth_for_frame(
    model: ColmapModel,
    frame_id: str,
    **kwargs,
) -> dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Sparse depth for each face_id in a frame."""
    out = {}
    for face in model.frames[frame_id]:
        out[face.face_id] = get_sparse_depth(model, face, **kwargs)
    return out
