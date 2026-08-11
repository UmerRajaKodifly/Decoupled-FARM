"""3D geometry helpers for Phase 2: unproject depth → 3D Gaussian, world transform.

Ported directly from FARM's YOLOESegmenter._compute_weighted_stats,
._depth_mode_filter, ._mahalanobis_reject (yoloe.py) and
transform_segmentation_to_world (utils/geometry.py).

All functions operate on PyTorch tensors. Device agnostic — callers place
tensors on the desired device before calling.

Usage in the pipeline
---------------------
Given per-frame (depth, K, masks) in camera frame and T_wc list:

    XB, YB, ZB = unproject_depth(depth_batch, K_batch)
    weights     = build_mask_weights(masks, XB.shape[1:], depth_valid_B)
    weights     = depth_mode_mad_filter(ZB, weights)
    n, means, cov6 = compute_weighted_stats(XB, YB, ZB, weights)
    weights     = mahalanobis_reject(XB, YB, ZB, weights, means, cov6)
    n, means, cov6 = compute_weighted_stats(XB, YB, ZB, weights)
    means       = compute_mask_medians(XB, YB, ZB, weights, min_points=50)
    means_w, cov6_w = transform_to_world(means, cov6, batch_ids, T_wc_list)
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Covariance pack/unpack  (identical to FARM utils/geometry.py)
# ---------------------------------------------------------------------------

def cov6_to_matrix(cov6: torch.Tensor) -> torch.Tensor:
    """(N,6) packed symmetric → (N,3,3)."""
    S = torch.zeros(cov6.shape[0], 3, 3, device=cov6.device, dtype=cov6.dtype)
    S[:, 0, 0] = cov6[:, 0]
    S[:, 0, 1] = S[:, 1, 0] = cov6[:, 1]
    S[:, 0, 2] = S[:, 2, 0] = cov6[:, 2]
    S[:, 1, 1] = cov6[:, 3]
    S[:, 1, 2] = S[:, 2, 1] = cov6[:, 4]
    S[:, 2, 2] = cov6[:, 5]
    return S


def matrix_to_cov6(cov: torch.Tensor) -> torch.Tensor:
    """(N,3,3) → (N,6) packed symmetric."""
    return torch.stack([
        cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
        cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2],
    ], dim=-1)


# ---------------------------------------------------------------------------
# Pixel-grid cache
# ---------------------------------------------------------------------------

_PIXEL_GRID_CACHE: dict = {}


def _get_pixel_grid(H: int, W: int, device: torch.device, dtype: torch.dtype):
    key = (H, W, device, dtype)
    if key not in _PIXEL_GRID_CACHE:
        y_coords = torch.arange(H, device=device, dtype=dtype)
        x_coords = torch.arange(W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        _PIXEL_GRID_CACHE[key] = (grid_y, grid_x)
    return _PIXEL_GRID_CACHE[key]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def unproject_depth(
    depth: torch.Tensor,       # (B, H, W) float32 metric metres
    K: torch.Tensor,           # (B, 3, 3) or (3, 3) pinhole intrinsics
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backproject a batch of depth maps into camera-frame 3-D point grids.

    Returns (XB, YB, ZB) each of shape (B, H, W).
    Invalid pixels (depth <= 0 or non-finite) are kept as-is; callers should
    mask with `depth_valid_B = (ZB > 0) & torch.isfinite(ZB)`.
    """
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    B, H, W = depth.shape
    device, dtype = depth.device, depth.dtype

    if K.ndim == 2:
        K = K.unsqueeze(0).expand(B, -1, -1)
    K = K.to(device=device, dtype=dtype)

    gy, gx = _get_pixel_grid(H, W, device, dtype)
    ZB = depth
    XB = (gx - K[:, 0, 2].view(B, 1, 1)) * ZB / K[:, 0, 0].view(B, 1, 1)
    YB = (gy - K[:, 1, 2].view(B, 1, 1)) * ZB / K[:, 1, 1].view(B, 1, 1)
    return XB, YB, ZB


def build_mask_weights(
    masks: torch.Tensor,         # (M, H, W) bool — per-detection instance masks
    depth_valid: torch.Tensor,   # (B, H, W) bool — valid depth pixels
    batch_ids: torch.Tensor,     # (M,) int — which image each detection belongs to
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Intersection of instance mask and valid-depth pixels → float weights (M,H,W)."""
    if masks.numel() == 0:
        return torch.zeros(0, masks.shape[-2], masks.shape[-1], dtype=dtype, device=masks.device)
    batch_ids_long = batch_ids.to(torch.long)
    dv = depth_valid[batch_ids_long]           # (M, H, W)
    return (masks & dv).to(dtype)


def erode_masks(masks: torch.Tensor, radius: int = 3) -> torch.Tensor:
    """Binary erosion with a square (2r+1)×(2r+1) kernel.

    Matches FARM's YOLOESegmenter._erode_masks exactly.
    """
    if radius <= 0 or masks.numel() == 0:
        return masks
    kernel = int(radius) * 2 + 1
    masks_f = masks.to(dtype=torch.float32)
    inv = 1.0 - masks_f
    dilated = F.max_pool2d(inv.unsqueeze(1), kernel_size=kernel, stride=1, padding=radius)
    eroded = (1.0 - dilated).squeeze(1)
    return (eroded > 0.5).to(torch.bool)


def depth_mode_mad_filter(
    ZB: torch.Tensor,         # (M, H, W) depths for each detection
    weights: torch.Tensor,    # (M, H, W) float mask weights
    k_mad: float = 3.0,
    min_mad_m: float = 0.03,
    min_depth_points: int = 50,
) -> torch.Tensor:
    """Remove mask pixels whose depth is more than k_mad MADs from the median.

    Ported from FARM's YOLOESegmenter._depth_mode_filter.
    Prevents background depth leakage when a detection box/mask crosses a
    foreground-background depth boundary.
    """
    if not weights.any():
        return weights
    out = weights.clone()
    M = weights.shape[0]
    for i in range(M):
        w_i = weights[i]
        if w_i.sum() < min_depth_points:
            continue
        z_vals = ZB[i][w_i > 0]
        median = z_vals.median()
        mad = (z_vals - median).abs().median().clamp(min=min_mad_m)
        lo = median - k_mad * mad
        hi = median + k_mad * mad
        depth_keep = (ZB[i] >= lo) & (ZB[i] <= hi)
        out[i] = out[i] * depth_keep.to(out.dtype)
    return out


def compute_weighted_stats(
    XB: torch.Tensor,  # (M, H, W)
    YB: torch.Tensor,
    ZB: torch.Tensor,
    weights: torch.Tensor,  # (M, H, W) float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted mean and 6-packed covariance for each detection.

    Returns (n, means, cov6) with shapes (M,), (M,3), (M,6).
    Ported from FARM's YOLOESegmenter._compute_weighted_stats.
    """
    n = weights.sum(dim=(1, 2))
    n_safe = n.clamp_min(1.0)
    inv_n = 1.0 / n_safe

    sx = (XB * weights).sum(dim=(1, 2))
    sy = (YB * weights).sum(dim=(1, 2))
    sz = (ZB * weights).sum(dim=(1, 2))
    mx, my, mz = sx * inv_n, sy * inv_n, sz * inv_n
    means = torch.stack([mx, my, mz], dim=1)

    nm1 = (n - 1.0).clamp_min(1.0)
    c_xx = ((XB * XB * weights).sum(dim=(1, 2)) - sx * sx * inv_n) / nm1
    c_yy = ((YB * YB * weights).sum(dim=(1, 2)) - sy * sy * inv_n) / nm1
    c_zz = ((ZB * ZB * weights).sum(dim=(1, 2)) - sz * sz * inv_n) / nm1
    c_xy = ((XB * YB * weights).sum(dim=(1, 2)) - sx * sy * inv_n) / nm1
    c_xz = ((XB * ZB * weights).sum(dim=(1, 2)) - sx * sz * inv_n) / nm1
    c_yz = ((YB * ZB * weights).sum(dim=(1, 2)) - sy * sz * inv_n) / nm1
    cov6 = torch.stack([c_xx, c_xy, c_xz, c_yy, c_yz, c_zz], dim=1)
    return n, means, cov6


def mahalanobis_reject(
    XB: torch.Tensor,
    YB: torch.Tensor,
    ZB: torch.Tensor,
    weights: torch.Tensor,
    means: torch.Tensor,     # (M, 3)
    cov6: torch.Tensor,      # (M, 6)
    thresh: float = 2.0,
    ridge: float = 1e-4,
) -> torch.Tensor:
    """Remove pixels beyond `thresh` Mahalanobis distances from the 3D mean.

    Ported from FARM's YOLOESegmenter._mahalanobis_reject.
    Returns updated weights (M, H, W).
    """
    if means.numel() == 0:
        return weights
    M = means.shape[0]
    dtype = weights.dtype
    cov_mats = cov6_to_matrix(cov6)                          # (M, 3, 3)
    eye3 = ridge * torch.eye(3, device=cov6.device, dtype=cov6.dtype)
    inv_cov = torch.linalg.inv(cov_mats + eye3.unsqueeze(0))  # (M, 3, 3)

    dx = XB - means[:, 0].view(M, 1, 1)
    dy = YB - means[:, 1].view(M, 1, 1)
    dz = ZB - means[:, 2].view(M, 1, 1)

    a = inv_cov[:, 0, 0].view(M, 1, 1)
    b = inv_cov[:, 0, 1].view(M, 1, 1)
    c = inv_cov[:, 0, 2].view(M, 1, 1)
    d = inv_cov[:, 1, 1].view(M, 1, 1)
    e = inv_cov[:, 1, 2].view(M, 1, 1)
    f = inv_cov[:, 2, 2].view(M, 1, 1)
    d2 = (a * dx * dx + d * dy * dy + f * dz * dz
          + 2.0 * b * dx * dy + 2.0 * c * dx * dz + 2.0 * e * dy * dz)

    inliers = (d2 <= thresh ** 2) & (weights > 0)
    return inliers.to(dtype=dtype)


def compute_mask_medians(
    XB: torch.Tensor,
    YB: torch.Tensor,
    ZB: torch.Tensor,
    weights: torch.Tensor,
    min_points: int = 50,
) -> torch.Tensor:
    """Return per-detection 3D medians as a robust alternative mean.

    Ported from FARM's YOLOESegmenter._compute_mask_medians.
    Used as the final `means` output (more robust to outlier clusters).
    """
    M = weights.shape[0]
    device, dtype = weights.device, weights.dtype
    medians = torch.zeros(M, 3, device=device, dtype=dtype)
    for i in range(M):
        mask_i = weights[i] > 0
        if mask_i.sum() < min_points:
            continue
        medians[i, 0] = XB[i][mask_i].median()
        medians[i, 1] = YB[i][mask_i].median()
        medians[i, 2] = ZB[i][mask_i].median()
    return medians


# ---------------------------------------------------------------------------
# World transform  (identical to FARM utils/geometry.transform_segmentation_to_world)
# ---------------------------------------------------------------------------

def transform_to_world(
    means: torch.Tensor,           # (M, 3) camera frame
    cov6: torch.Tensor,            # (M, 6) camera frame
    batch_ids: torch.Tensor,       # (M,) int — which pose each detection uses
    poses_world: Sequence[torch.Tensor],  # list of (4,4) T_world_cam
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply per-detection camera-to-world poses to means and covariances.

    Returns (means_world, cov6_world) both in the shared world frame.
    Identical math to FARM's transform_segmentation_to_world.
    """
    if means.numel() == 0 or len(poses_world) == 0:
        return means.clone(), cov6.clone()

    device, dtype = means.device, means.dtype
    batch_ids_long = batch_ids.long()
    means_w = means.clone()
    cov6_w = cov6.clone()

    for batch_idx in torch.unique(batch_ids_long).tolist():
        if batch_idx < 0 or batch_idx >= len(poses_world):
            continue
        mask = batch_ids_long == batch_idx
        if not mask.any():
            continue
        pose = poses_world[batch_idx].to(device=device, dtype=dtype)
        R, t = pose[:3, :3], pose[:3, 3]

        means_sub = means[mask]
        if means_sub.numel() > 0:
            means_w[mask] = means_sub @ R.T + t

        cov_sub = cov6[mask]
        if cov_sub.numel() > 0:
            cov_mats = cov6_to_matrix(cov_sub)
            rot = R.unsqueeze(0).expand(cov_mats.shape[0], -1, -1)
            cov_w = rot @ cov_mats @ rot.transpose(1, 2)
            cov6_w[mask] = matrix_to_cov6(cov_w)

    return means_w, cov6_w
