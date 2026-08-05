"""Browser viewer for exported FARM object npzs (Gaussian mean + sparse voxels)."""

from __future__ import annotations

import colorsys
import json
import time
from pathlib import Path

import numpy as np


def _resolve_npz_dir(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file() and path.name == "summary.json":
        path = path.parent
    nested = path / "objects"
    if nested.is_dir() and any(nested.glob("object_*.npz")):
        return nested
    if path.is_dir() and any(path.glob("object_*.npz")):
        return path
    raise FileNotFoundError(f"No object_*.npz under {path}")


def _color(i: int) -> tuple[int, int, int]:
    h = (i * 0.15) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _ellipsoid_points(mean: np.ndarray, cov: np.ndarray, n_u: int = 16, n_v: int = 10) -> np.ndarray:
    vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    vals = np.clip(vals, 1e-8, None)
    radii = 2.0 * np.sqrt(vals)
    us = np.linspace(0.0, 2.0 * np.pi, n_u, endpoint=False)
    vs = np.linspace(0.0, np.pi, n_v)
    uu, vv = np.meshgrid(us, vs)
    local = np.stack(
        [
            radii[0] * np.cos(uu) * np.sin(vv),
            radii[1] * np.sin(uu) * np.sin(vv),
            radii[2] * np.cos(vv),
        ],
        axis=-1,
    ).reshape(-1, 3)
    return (local @ vecs.T + np.asarray(mean, dtype=np.float64)).astype(np.float32)


def load_objects(npz_dir: str | Path) -> list[dict]:
    npz_dir = _resolve_npz_dir(npz_dir)
    objects = []
    for path in sorted(npz_dir.glob("object_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            label = str(data["label"])
            mean = np.asarray(data["mean"], dtype=np.float32)
            cov = np.asarray(data["cov"], dtype=np.float32)
            ijk = np.asarray(data["voxels_ijk"], dtype=np.int32)
            voxel_size = float(data["voxel_size"])
            pts = (ijk.astype(np.float32) + 0.5) * voxel_size if ijk.size else np.zeros((0, 3), np.float32)
            objects.append(
                {
                    "object_id": int(data["object_id"]),
                    "label": label,
                    "mean": mean,
                    "cov": cov,
                    "points": pts,
                    "voxel_size": voxel_size,
                    "path": str(path),
                }
            )
    if not objects:
        raise FileNotFoundError(f"No object_*.npz in {npz_dir}")
    return objects


def serve(
    objects_dir: str | Path,
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
    max_points_per_object: int = 4000,
    background_cloud: str | Path | None = None,
) -> None:
    import viser

    objects = load_objects(objects_dir)
    server = viser.ViserServer(host=host, port=port)
    server.scene.add_frame("/world", wxyz=(1.0, 0.0, 0.0, 0.0), position=(0.0, 0.0, 0.0))

    if background_cloud:
        cloud_path = Path(background_cloud)
        with np.load(cloud_path) as data:
            bg = np.asarray(data["points"], dtype=np.float32)
        if bg.shape[0] > 80_000:
            bg = bg[np.random.default_rng(0).choice(bg.shape[0], 80_000, replace=False)]
        server.scene.add_point_cloud(
            "/sfm_sparse",
            points=bg,
            colors=np.full((bg.shape[0], 3), 180, dtype=np.uint8),
            point_size=0.04,
        )

    for i, obj in enumerate(objects):
        rgb = _color(i)
        name = f"{obj['object_id']:04d}_{obj['label'].replace(' ', '_')}"
        pts = obj["points"]
        if pts.shape[0] > max_points_per_object:
            idx = np.random.default_rng(obj["object_id"]).choice(pts.shape[0], max_points_per_object, replace=False)
            pts = pts[idx]
        if pts.shape[0]:
            colors = np.tile(np.asarray(rgb, dtype=np.uint8), (pts.shape[0], 1))
            server.scene.add_point_cloud(
                f"/objects/{name}/voxels",
                points=pts,
                colors=colors,
                point_size=max(0.03, obj["voxel_size"] * 0.6),
            )
        server.scene.add_icosphere(
            f"/objects/{name}/mean",
            radius=max(0.08, obj["voxel_size"]),
            color=tuple(c / 255.0 for c in rgb),
            position=tuple(float(x) for x in obj["mean"]),
        )
        ell = _ellipsoid_points(obj["mean"], obj["cov"])
        server.scene.add_point_cloud(
            f"/objects/{name}/gaussian",
            points=ell,
            colors=np.tile(np.asarray(rgb, dtype=np.uint8), (ell.shape[0], 1)),
            point_size=0.02,
        )
        server.scene.add_label(
            f"/objects/{name}/label",
            text=f"{obj['object_id']}: {obj['label']}",
            position=tuple(float(x) for x in obj["mean"]),
        )

    summary = {
        "n_objects": len(objects),
        "url": f"http://127.0.0.1:{port}",
        "labels": [f"{o['object_id']}: {o['label']}" for o in objects],
    }
    print(json.dumps(summary, indent=2))
    print(f"Open {summary['url']}  (Ctrl+C to stop)")
    while True:
        time.sleep(1.0)
