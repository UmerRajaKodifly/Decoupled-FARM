"""Rewrite scene_state geometry using per-object Stella point clouds.

For each active object with ≥ min_pts Stella inliers:
  - means[i]  = median XYZ of inlier points (robust, no outlier bias)
  - cov6[i]   = empirical covariance, clamped so max eigenvalue ≤ max_sigma²
  - object_stella_pts[i] = subsampled inlier points stored for Viser display

Objects with < min_pts points are left unchanged (original Gaussian geometry).

This is a **non-destructive** update: we write to a new scene_state dict copy
and keep `scene_state_with_stella_pts` for the Viser viewer.
"""

from __future__ import annotations

import copy
import logging
from typing import List, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

_COV6_IDX = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def _pack_cov6(cov3x3: np.ndarray) -> np.ndarray:
    """3×3 → 6-packed (xx, xy, xz, yy, yz, zz)."""
    return np.array([cov3x3[r, c] for r, c in _COV6_IDX], dtype=np.float32)


def _unpack_cov6(c6: np.ndarray) -> np.ndarray:
    """6-packed → 3×3 symmetric."""
    m = np.zeros((3, 3), dtype=np.float64)
    for k, (r, c) in enumerate(_COV6_IDX):
        m[r, c] = c6[k]
        m[c, r] = c6[k]
    return m


def _empirical_cov6(
    pts: np.ndarray,
    *,
    ridge: float = 1e-4,
    max_sigma: float = 1.5,
) -> np.ndarray:
    """Compute 6-packed covariance from point set; clamp max eigenvalue.

    Parameters
    ----------
    pts       : (M, 3) float32/float64.
    ridge     : additive diagonal regularisation.
    max_sigma : cap on sqrt(eigenvalue) = σ per axis (metres).
                5σ box side ≤ 5 * max_sigma * 2 per axis.
    """
    pts = pts.astype(np.float64)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = (centered.T @ centered) / max(len(pts) - 1, 1) + ridge * np.eye(3)

    # Clamp eigenvalues
    eigvals, eigvecs = np.linalg.eigh(cov)
    max_var = max_sigma ** 2
    eigvals = np.clip(eigvals, ridge, max_var)
    cov_clamped = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return _pack_cov6(cov_clamped.astype(np.float32))


def subsample_points(pts: np.ndarray, max_pts: int = 500) -> np.ndarray:
    """Uniform random subsample if over max_pts."""
    if pts.shape[0] <= max_pts:
        return pts
    rng = np.random.default_rng(42)
    idx = rng.choice(pts.shape[0], size=max_pts, replace=False)
    return pts[idx]


def rewrite_geometry(
    scene_state: dict,
    object_point_arrays: List[Optional[np.ndarray]],
    *,
    min_pts: int = 5,
    max_sigma: float = 1.5,
    max_voxel_pts_per_object: int = 500,
) -> dict:
    """Return a modified copy of scene_state with Stella-derived geometry.

    New / updated fields
    --------------------
    means        : updated for objects with Stella support
    cov6         : updated for objects with Stella support
    stella_means : (N, 3) tensor  — pre-update backup of original Gaussian means
    stella_n_pts : (N,) int tensor — Stella inlier count (0 = not updated)
    object_stella_pts : list of (M_i, 3) float32 numpy or None  — for Viser display
    """
    ss = copy.deepcopy(scene_state)

    means = ss["means"]   # (N, 3) tensor
    cov6 = ss["cov6"]     # (N, 6) tensor
    n = int(means.shape[0])

    # Back up original geometry
    ss["stella_means"] = means.clone()
    ss["stella_n_pts"] = torch.zeros(n, dtype=torch.long)

    stella_pts_list: List[Optional[np.ndarray]] = []
    n_updated = 0

    for i in range(n):
        pts = object_point_arrays[i] if i < len(object_point_arrays) else None
        stella_pts_list.append(None)

        if pts is None or pts.shape[0] < min_pts:
            continue

        # Median mean (robust to outlier clusters)
        new_mean = np.median(pts, axis=0).astype(np.float32)
        new_cov6 = _empirical_cov6(pts, max_sigma=max_sigma)

        means[i] = torch.from_numpy(new_mean)
        cov6[i] = torch.from_numpy(new_cov6)
        ss["stella_n_pts"][i] = pts.shape[0]

        # Subsampled for display
        stella_pts_list[-1] = subsample_points(pts, max_voxel_pts_per_object)
        n_updated += 1

    ss["means"] = means
    ss["cov6"] = cov6
    ss["object_stella_pts"] = stella_pts_list

    log.info(
        "Geometry rewrite: %d / %d objects updated with Stella PCD (≥%d pts)",
        n_updated, n, min_pts,
    )
    return ss
