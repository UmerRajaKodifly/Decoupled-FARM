"""Project Stella dense cloud into face cameras; label points via mask + depth.

For each keyframe face we:
  1. Project all cloud points into the face camera frame.
  2. Keep points with Z > 0, within image bounds.
  3. Apply depth occlusion: |Z_cloud - depth[v,u]| < tau.
  4. For each YOLOE detection mask in this face, collect inlier points.

Returns a mapping  detection_key → point indices into the downsampled cloud.
A detection_key is (kf_enum_index, face_index, det_in_face_index).

Depth tolerance tau
-------------------
We use a *relative + absolute* combined tolerance:
    tau = max(tau_abs, tau_rel * Z_cloud)
Default tau_abs=0.15m, tau_rel=0.05 (5 % of the range).
This allows a slightly larger window for distant objects (10 m → ±0.5 m) while
staying tight at short range (2 m → ±0.15 m).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)

# Type alias for clarity
DetKey = Tuple[int, int, int]  # (kf_enum_idx, face_idx, det_local_idx)


def project_face_label(
    pts: np.ndarray,           # (N, 3) float32 world XYZ
    T_cw: np.ndarray,          # (4, 4) float64 camera-from-world  (T_cw)
    K: np.ndarray,             # (3, 3) float32 face intrinsics
    depth: np.ndarray,         # (H, W) float32 metric depth (metres)
    masks: Optional[List],     # list of (H, W) bool/uint8 masks, or None
    *,
    tau_abs: float = 0.15,
    tau_rel: float = 0.05,
) -> Dict[int, np.ndarray]:
    """Project cloud into one face; return {mask_idx: point_indices}.

    Parameters
    ----------
    pts       : (N, 3) float32 world XYZ (already downsampled).
    T_cw      : camera-from-world 4×4.
    K         : 3×3 pinhole intrinsics.
    depth     : (H, W) float32 metric depth.
    masks     : list of (H, W) boolean arrays, one per detection.
    tau_abs   : absolute depth tolerance (metres).
    tau_rel   : relative depth tolerance (fraction of camera-Z).

    Returns
    -------
    dict mapping mask_index → np.ndarray of point indices (into `pts`).
    Empty dict if masks is None / empty.
    """
    if masks is None or len(masks) == 0:
        return {}

    N = pts.shape[0]
    if N == 0:
        return {}

    H, W = depth.shape
    R_cw = T_cw[:3, :3].astype(np.float32)
    t_cw = T_cw[:3, 3].astype(np.float32)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    # Camera-frame coords: P_cam = R_cw @ pts.T + t_cw
    # pts shape (N,3); result (3,N)
    P_cam = (R_cw @ pts.T) + t_cw[:, None]   # (3, N)
    Z = P_cam[2]

    # Keep in front of camera
    front = Z > 0.01
    if not front.any():
        return {}

    # Project to pixel coords
    U = fx * P_cam[0] / np.where(Z > 0, Z, 1.0) + cx
    V = fy * P_cam[1] / np.where(Z > 0, Z, 1.0) + cy

    Ui = np.round(U).astype(np.int32)
    Vi = np.round(V).astype(np.int32)

    # In-bounds
    in_bounds = (
        front
        & (Ui >= 0) & (Ui < W)
        & (Vi >= 0) & (Vi < H)
    )

    valid_idx = np.where(in_bounds)[0]
    if valid_idx.size == 0:
        return {}

    # Depth occlusion check
    Ui_v = Ui[valid_idx].clip(0, W - 1)
    Vi_v = Vi[valid_idx].clip(0, H - 1)
    Z_v = Z[valid_idx]
    depth_v = depth[Vi_v, Ui_v]

    # Skip pixels with invalid depth
    depth_valid = (depth_v > 0) & np.isfinite(depth_v)
    tau = np.maximum(tau_abs, tau_rel * Z_v)
    depth_ok = depth_valid & (np.abs(Z_v - depth_v) < tau)

    inlier_idx = valid_idx[depth_ok]   # indices into pts
    Ui_in = Ui[inlier_idx]
    Vi_in = Vi[inlier_idx]

    if inlier_idx.size == 0:
        return {}

    result: Dict[int, np.ndarray] = {}
    for mi, mask in enumerate(masks):
        if mask is None:
            continue
        mask_arr: np.ndarray
        if isinstance(mask, torch.Tensor):
            mask_arr = mask.detach().cpu().numpy().astype(bool)
        else:
            mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != (H, W):
            # Resize if needed (should not happen in normal flow)
            continue
        inside = mask_arr[Vi_in, Ui_in]
        if inside.any():
            result[mi] = inlier_idx[inside]

    return result
