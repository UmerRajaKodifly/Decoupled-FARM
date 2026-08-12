"""Map Phase 2 face-detections to fused Phase 3 objects and accumulate Stella points.

For each fused object we know which global image IDs (kf_index*4+face_index) observed
it via `scene_state["object_image_ids"]`. For each such face we find the best-matching
detection (highest feature cosine among same-class candidates) and collect the Stella
inlier points that fall under that detection's mask.

Returns object_point_arrays: list of length N_objects, each entry is
  None                   (no Stella support found)
  np.ndarray (M, 3)      float32 world XYZ of Stella inliers for this object
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from project_label import project_face_label

log = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _T_cw_from_pack(pack: dict, face_idx: int) -> Optional[np.ndarray]:
    """Return T_cw (4×4) for a face.  poses_world stores T_wc so we invert."""
    pw = pack.get("poses_world")
    if pw is None or face_idx >= len(pw):
        return None
    T_wc = pw[face_idx]
    if isinstance(T_wc, torch.Tensor):
        T_wc = T_wc.detach().cpu().numpy()
    T_wc = np.asarray(T_wc, dtype=np.float64)
    if T_wc.shape != (4, 4):
        return None
    return np.linalg.inv(T_wc)


def _K_from_pack(pack: dict, face_idx: int) -> Optional[np.ndarray]:
    kl = pack.get("intrinsics")
    if kl is None or face_idx >= len(kl):
        return None
    K = kl[face_idx]
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    return np.asarray(K, dtype=np.float32)


def build_object_point_arrays(
    pts: np.ndarray,                # (N, 3) float32 downsampled world pts
    scene_state: dict,
    pack_paths: List[Path],
    *,
    tau_abs: float = 0.15,
    tau_rel: float = 0.05,
    feat_sim_min: float = 0.30,     # loose gate — geometry first pass
    max_center_dist_m: float = 6.0,
    min_points_per_object: int = 3,
) -> List[Optional[np.ndarray]]:
    """Collect Stella dense points per fused object.

    Parameters
    ----------
    pts           : downsampled Stella dense cloud, (N, 3) float32.
    scene_state   : Phase 3 fused map.
    pack_paths    : sorted list of Phase 2 .pt paths (index = kf_enum_index).

    Returns
    -------
    List of length N_objects.  Entry i is None or (M_i, 3) float32 world pts.
    """
    means = scene_state.get("means")
    features = scene_state.get("features")
    class_ids_ss = scene_state.get("class_ids")
    active = scene_state.get("active")
    image_ids_list = scene_state.get("object_image_ids") or []

    if not isinstance(means, torch.Tensor) or means.numel() == 0:
        return []

    n_obj = int(means.shape[0])
    means_np = means.detach().cpu().numpy().astype(np.float32)
    feats_np = (
        features.detach().cpu().numpy().astype(np.float32)
        if isinstance(features, torch.Tensor)
        else None
    )
    active_np = (
        active.detach().cpu().numpy().astype(bool)
        if isinstance(active, torch.Tensor)
        else np.ones(n_obj, dtype=bool)
    )

    # Accumulate points per object across all views
    # Use a dict of lists then stack later
    obj_pts_chunks: List[List[np.ndarray]] = [[] for _ in range(n_obj)]

    # Cache loaded packs to avoid double-loading
    _pack_cache: Dict[int, Optional[dict]] = {}

    def _get_pack(kf_idx: int) -> Optional[dict]:
        if kf_idx not in _pack_cache:
            if kf_idx < 0 or kf_idx >= len(pack_paths):
                _pack_cache[kf_idx] = None
            else:
                try:
                    _pack_cache[kf_idx] = torch.load(
                        pack_paths[kf_idx], map_location="cpu", weights_only=False
                    )
                except Exception as exc:
                    log.warning("Failed to load pack %s: %s", pack_paths[kf_idx], exc)
                    _pack_cache[kf_idx] = None
        return _pack_cache[kf_idx]

    n_processed = 0

    for obj_idx in range(n_obj):
        if not active_np[obj_idx]:
            continue

        obj_mean = means_np[obj_idx]
        obj_feat = feats_np[obj_idx] if feats_np is not None else None
        obj_cls = (
            int(class_ids_ss[obj_idx].item())
            if isinstance(class_ids_ss, torch.Tensor) and class_ids_ss.numel() > obj_idx
            else -1
        )
        image_ids = image_ids_list[obj_idx] if obj_idx < len(image_ids_list) else []
        if not image_ids:
            continue

        # Process each face that observed this object
        for raw_iid in image_ids:
            try:
                image_id = int(raw_iid)
            except (TypeError, ValueError):
                continue
            kf_idx = image_id // 4
            face_idx = image_id % 4

            pack = _get_pack(kf_idx)
            if pack is None:
                continue

            T_cw = _T_cw_from_pack(pack, face_idx)
            K = _K_from_pack(pack, face_idx)
            if T_cw is None or K is None:
                continue

            # Load depth for this face
            fm = pack.get("face_meta") or []
            if face_idx >= len(fm):
                continue
            dep_path = fm[face_idx].get("depth") if isinstance(fm[face_idx], dict) else None
            if not dep_path:
                continue
            dep_p = Path(dep_path)
            if not dep_p.is_file():
                continue
            try:
                import numpy as _np
                depth = _np.load(str(dep_p)).astype(np.float32)
                if depth.ndim == 3:
                    depth = depth[..., 0]
            except Exception:
                continue

            # Build mask list for detections on this face
            batch_ids = pack.get("batch_ids")
            if not isinstance(batch_ids, torch.Tensor) or batch_ids.numel() == 0:
                continue

            det_indices = [
                i for i, b in enumerate(batch_ids.tolist())
                if int(b) == face_idx
            ]
            if not det_indices:
                continue

            # Build masks list in detection order for this face
            pack_masks = pack.get("masks")
            pack_feats = pack.get("features")
            pack_means = pack.get("means")
            pack_cls = pack.get("class_ids")

            if pack_masks is None:
                continue

            # Find best matching detection for this object on this face
            best_di = None
            best_sim = feat_sim_min - 1e-9

            for di in det_indices:
                if di >= len(pack_masks):
                    continue
                # Center distance gate
                if isinstance(pack_means, torch.Tensor) and di < pack_means.shape[0]:
                    det_mean = pack_means[di].detach().cpu().numpy().astype(np.float32)
                    dist = float(np.linalg.norm(det_mean - obj_mean))
                    if dist > max_center_dist_m:
                        continue
                # Feature similarity gate
                sim = 0.0
                if (
                    obj_feat is not None
                    and isinstance(pack_feats, torch.Tensor)
                    and di < pack_feats.shape[0]
                ):
                    det_feat = pack_feats[di].detach().cpu().numpy().astype(np.float32)
                    sim = _cosine(obj_feat, det_feat)
                    if sim < feat_sim_min:
                        continue
                if sim > best_sim:
                    best_sim = sim
                    best_di = di

            if best_di is None:
                continue

            mask = pack_masks[best_di]
            if mask is None:
                continue

            # Pre-filter cloud to points within rough range of camera origin
            # to avoid projecting 15M points per face when most are far away.
            cam_origin = -T_cw[:3, :3].T @ T_cw[:3, 3]  # world pos of camera
            dists = np.linalg.norm(pts - cam_origin.astype(np.float32), axis=1)
            near_mask = dists < 30.0
            pts_near = pts[near_mask]
            near_idx = np.where(near_mask)[0]

            if pts_near.shape[0] == 0:
                continue

            results = project_face_label(
                pts_near, T_cw, K, depth, [mask],
                tau_abs=tau_abs, tau_rel=tau_rel,
            )

            if 0 in results and results[0].size > 0:
                # Map local indices back to global cloud indices
                global_idx = near_idx[results[0]]
                obj_pts_chunks[obj_idx].append(pts[global_idx])

        n_processed += 1

    log.info("Processed %d active objects for Stella geometry", n_processed)

    # Merge chunks per object
    result: List[Optional[np.ndarray]] = []
    n_covered = 0
    for chunks in obj_pts_chunks:
        if not chunks:
            result.append(None)
        else:
            merged = np.concatenate(chunks, axis=0)
            if merged.shape[0] < min_points_per_object:
                result.append(None)
            else:
                result.append(merged)
                n_covered += 1

    log.info(
        "Stella geometry: %d / %d active objects have ≥%d support points",
        n_covered,
        n_processed,
        min_points_per_object,
    )
    return result
