"""Unprojection, SE(3) transform, Gaussian stats, and sparse voxels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .depth import DepthMap


@dataclass(frozen=True)
class ObjectGaussian:
    mean: np.ndarray  # (3,) world or camera, depending on caller
    cov: np.ndarray  # (3, 3)
    n_points: int


@dataclass(frozen=True)
class SparseVoxels:
    ijk: np.ndarray  # (N, 3) int32 voxel indices
    voxel_size: float
    origin: np.ndarray  # (3,) world origin of the integer grid (usually zeros)


def unproject_pixels(
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """FARM-identical pinhole unprojection (YOLOESegmenter lines 1148–1151).

    ``u``/``v`` are integer pixel indices (``torch.arange`` / ``np.nonzero``),
    **not** pixel centres. ``z`` is optical-axis depth. No +0.5 offset.

    .. math::

        X = (u - c_x)\\, Z / f_x,\\quad
        Y = (v - c_y)\\, Z / f_y,\\quad
        Z = z
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (u.astype(np.float64) - cx) * z / fx
    y = (v.astype(np.float64) - cy) * z / fy
    return np.stack([x, y, z.astype(np.float64)], axis=-1).astype(np.float32)


def unproject_masked_depth(
    depth: DepthMap,
    mask: np.ndarray,
    K: np.ndarray,
    *,
    min_depth: float = 1e-4,
    max_depth: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Unproject valid depth pixels under a 2D instance mask.

    Returns camera-frame points ``(N, 3)`` and a drop-count dict for logging.
    """
    if mask.shape != depth.depth_m.shape:
        raise ValueError(
            f"mask shape {mask.shape} != depth shape {depth.depth_m.shape}"
        )
    mask_bool = np.asarray(mask, dtype=bool)
    n_mask = int(mask_bool.sum())
    valid = depth.validity() & mask_bool
    z = depth.depth_m
    valid &= z > float(min_depth)
    if max_depth is not None:
        valid &= z < float(max_depth)

    n_valid = int(valid.sum())
    stats = {
        "mask_pixels": n_mask,
        "valid_depth_pixels": n_valid,
        "dropped_invalid_depth": n_mask - n_valid,
    }
    if n_valid == 0:
        return np.zeros((0, 3), dtype=np.float32), stats

    v_idx, u_idx = np.nonzero(valid)
    pts = unproject_pixels(u_idx.astype(np.float64), v_idx.astype(np.float64), z[valid], K)
    return pts, stats


def invert_se3(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_points_cam_to_world(points_cam: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    """Apply camera-to-world SE(3): ``p_w = R p_c + t``."""
    if points_cam.size == 0:
        return points_cam.astype(np.float32, copy=False)
    R = np.asarray(T_world_cam[:3, :3], dtype=np.float64)
    t = np.asarray(T_world_cam[:3, 3], dtype=np.float64)
    return (points_cam.astype(np.float64) @ R.T + t).astype(np.float32)


def points_to_gaussian(points: np.ndarray, *, min_points: int = 8) -> ObjectGaussian | None:
    if points.shape[0] < min_points:
        return None
    pts = points.astype(np.float64, copy=False)
    mean = pts.mean(axis=0)
    centered = pts - mean
    if pts.shape[0] < 2:
        cov = np.eye(3, dtype=np.float64) * 1e-6
    else:
        cov = (centered.T @ centered) / max(pts.shape[0] - 1, 1)
        cov = cov + np.eye(3) * 1e-8
    return ObjectGaussian(
        mean=mean.astype(np.float32),
        cov=cov.astype(np.float32),
        n_points=int(pts.shape[0]),
    )


def voxelize_points(
    points_world: np.ndarray,
    voxel_size: float,
    *,
    origin: np.ndarray | None = None,
) -> SparseVoxels:
    """Unique integer voxels. Default ``voxel_size=0.05`` (5 cm indoor assumption)."""
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be > 0, got {voxel_size}")
    origin_v = np.zeros(3, dtype=np.float64) if origin is None else np.asarray(origin, dtype=np.float64)
    if points_world.size == 0:
        return SparseVoxels(
            ijk=np.zeros((0, 3), dtype=np.int32),
            voxel_size=float(voxel_size),
            origin=origin_v.astype(np.float32),
        )
    q = np.floor((points_world.astype(np.float64) - origin_v) / float(voxel_size)).astype(np.int32)
    uniq = np.unique(q, axis=0)
    return SparseVoxels(ijk=uniq, voxel_size=float(voxel_size), origin=origin_v.astype(np.float32))


def merge_voxels(a: SparseVoxels, b: SparseVoxels) -> SparseVoxels:
    if abs(a.voxel_size - b.voxel_size) > 1e-12:
        raise ValueError("Cannot merge voxels with different voxel_size")
    if not np.allclose(a.origin, b.origin):
        raise ValueError("Cannot merge voxels with different origins")
    if a.ijk.size == 0:
        return b
    if b.ijk.size == 0:
        return a
    uniq = np.unique(np.concatenate([a.ijk, b.ijk], axis=0), axis=0)
    return SparseVoxels(ijk=uniq, voxel_size=a.voxel_size, origin=a.origin)


def project_world_point(
    point_world: np.ndarray,
    T_world_cam: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Project a world point into the image. Returns ``(u, v)`` and camera Z."""
    T_cam_world = invert_se3(T_world_cam)
    p = np.asarray(point_world, dtype=np.float64).reshape(3)
    p_cam = T_cam_world[:3, :3] @ p + T_cam_world[:3, 3]
    z = float(p_cam[2])
    u = float(K[0, 0] * p_cam[0] / z + K[0, 2])
    v = float(K[1, 1] * p_cam[1] / z + K[1, 2])
    return np.array([u, v], dtype=np.float64), z
