"""Optional background cloud.npz (viz only — not an input to object mapping)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData


def ply_to_cloud_npz(ply_path: str | Path, npz_path: str | Path) -> Path:
    ply_path = Path(ply_path)
    npz_path = Path(npz_path)
    ply = PlyData.read(str(ply_path))
    vertex = ply["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    colors = None
    names = set(vertex.data.dtype.names or [])
    if {"red", "green", "blue"} <= names:
        colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.uint8)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    if colors is not None:
        np.savez_compressed(npz_path, points=xyz, colors=colors)
    else:
        np.savez_compressed(npz_path, points=xyz)
    return npz_path
