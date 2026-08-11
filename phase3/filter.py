"""Phase 3 — face-aware detection filtering.

Calls FARM's filtering functions in the correct sequence, with two 360-cubemap
adaptations:

1. Border filter: `min_kept_num_pixels=1000` instead of FARM's 4000.
   Cube-face seams mean real large objects can legitimately clip the face edge.
   A detection is only killed if it is both border-touching AND small.

2. Distance filter: uses the `poses_world` list already in the Phase 2 pack
   (one 4×4 pose per face, indexed by `batch_ids`).

All other filters are called with FARM defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence

import torch

# ---------------------------------------------------------------------------
# FARM path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_COMMON = _HERE.parent.parent / "common"
if _COMMON.is_dir() and str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))
try:
    from paths import ensure_sys_path
    ensure_sys_path(_HERE)
except ImportError:
    # fallback: repo/farm_src/src
    import sys as _sys
    _cand = _HERE.parent.parent / "farm_src" / "src"
    if _cand.is_dir() and str(_cand) not in _sys.path:
        _sys.path.insert(0, str(_cand))

from scene_graph.map_update.filtering import (  # noqa: E402
    filter_detections_by_distance,
    filter_detections_by_num_pixels,
    filter_detections_duplicates_iou,
    filter_detections_touching_image_border,
    normalize_seg_outputs,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def filter_pack(
    pack: dict,
    *,
    device: str | torch.device = "cuda",
    min_num_pixels: int = 50,
    min_distance_m: float = 0.3,
    max_distance_m: float = 80.0,
    border_margin_px: int = 5,
    border_max_area_fraction: float = 0.05,
    # Raised vs FARM default (4000) — cube seams legitimately clip large objects
    border_min_kept_pixels: int = 1000,
    iou_dedup_thresh: float = 0.6,
    load_rgb_for_border: bool = True,
) -> dict:
    """Filter one Phase 2 detection pack, returning the same schema.

    Steps (FARM-faithful order):
      1. normalize_seg_outputs — dtype/device coercion
      2. filter_detections_by_num_pixels — drop tiny masks
      3. filter_detections_touching_image_border — drop small border slivers
      4. filter_detections_by_distance — drop out-of-range world means
      5. filter_detections_duplicates_iou — drop per-face duplicates

    Parameters
    ----------
    pack : dict
        Raw Phase 2 pack loaded with torch.load.
    device : str or torch.device
        Target device; tensors are moved here before filtering.
    load_rgb_for_border : bool
        If True, load face RGB images from ``face_meta`` paths for the border
        filter.  If files are missing the border filter is skipped gracefully.
    """
    dev = torch.device(device)

    # Step 1 — normalise
    pack = normalize_seg_outputs(pack, fallback_device=dev)

    # Move key tensors to device
    for key in ("means", "cov6", "features", "scores", "class_ids",
                "batch_ids", "num_pixels", "boxes_xyxy",
                "det_points_flat", "det_points_offsets"):
        if key in pack and isinstance(pack[key], torch.Tensor):
            pack[key] = pack[key].to(dev)

    if pack.get("means") is None or pack["means"].numel() == 0:
        return pack

    # Step 2 — pixel count
    pack = filter_detections_by_num_pixels(pack, min_num_pixels=min_num_pixels)
    if pack["means"].numel() == 0:
        return pack

    # Step 3 — border filter (needs color tensors)
    colors: List[torch.Tensor] = []
    if load_rgb_for_border:
        colors = _load_face_colors(pack)
    if colors:
        pack = filter_detections_touching_image_border(
            pack,
            colors,
            margin_px=border_margin_px,
            max_area_fraction=border_max_area_fraction,
            min_kept_num_pixels=border_min_kept_pixels,
        )
    if pack["means"].numel() == 0:
        return pack

    # Step 4 — distance from camera (uses poses from pack)
    poses_world = pack.get("poses_world") or []
    if poses_world:
        poses_world_t = [
            p.to(dev, dtype=torch.float32) if isinstance(p, torch.Tensor)
            else torch.tensor(p, device=dev, dtype=torch.float32)
            for p in poses_world
        ]
        pack = filter_detections_by_distance(
            pack, poses_world_t,
            min_distance_m=min_distance_m,
            max_distance_m=max_distance_m,
        )
    if pack["means"].numel() == 0:
        return pack

    # Step 5 — IoU dedup (per face via batch_ids inside FARM)
    pack = filter_detections_duplicates_iou(pack, min_iou=iou_dedup_thresh)

    return pack


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _load_face_colors(pack: dict) -> List[torch.Tensor]:
    """Load face RGB tensors from face_meta paths; skip missing files silently."""
    import numpy as np
    from PIL import Image

    face_meta = pack.get("face_meta") or []
    colors: List[torch.Tensor] = []
    for meta in face_meta:
        path = Path(meta.get("rgb", ""))
        if path.exists():
            try:
                img = Image.open(path).convert("RGB")
                t = torch.from_numpy(np.array(img, dtype=np.uint8))
                colors.append(t)
            except Exception:
                colors.append(torch.zeros((504, 504, 3), dtype=torch.uint8))
        else:
            colors.append(torch.zeros((504, 504, 3), dtype=torch.uint8))
    return colors
