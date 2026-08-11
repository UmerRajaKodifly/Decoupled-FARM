"""Phase 3 — scene-state update: fuse detections, update covisibility, apply cannot-links.

Wraps FARM's `update_scene_graph_state` (from object_update.py) and
`update_covisibility_from_visible_indices` (from covisibility.py).

No captions are enqueued here — captions_d is always a list of empty strings.
`det_points_flat` / `det_points_offsets` (already in Phase 2 packs) are forwarded
so FARM's voxel update path is fully active.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

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

from scene_graph.map_update.cannot_link import (  # noqa: E402
    add_same_frame_cannot_links_from_detection_assignments,
)
from scene_graph.map_update.covisibility import (  # noqa: E402
    update_covisibility_from_visible_indices,
)
from scene_graph.map_update.object_update import update_scene_graph_state  # noqa: E402

from label_vote import apply_label_voting_after_fuse  # noqa: E402


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fuse_detections(
    pack: dict,
    det_idx: torch.Tensor,
    obj_idx: torch.Tensor,
    scene_state: dict,
    detection_image_ids: List[Optional[int]],
    *,
    allow_new_objects: bool = True,
    max_merge_distance_m: Optional[float] = 1.0,
    label_min_score: float = 0.25,
    label_margin_ratio: float = 1.15,
    label_use_pixel_weight: bool = True,
) -> dict:
    """Fuse detections into *scene_state* in-place and update covisibility.

    Steps (FARM-faithful):
      1. Call `update_scene_graph_state` — Gaussian/voxel merge or new-object
         append; obj_winner_idx encodes union-find merges of existing objects.
      2. Collect visible object indices from the update result + det→obj map.
      3. Call `update_covisibility_from_visible_indices` for all active objects
         seen this keyframe.
      4. Call `add_same_frame_cannot_links_from_detection_assignments`
         (post-update, so new object indices are known).
      5. Weighted multi-frame class voting — updates ``class_ids`` from
         ``class_vote_mass`` (score × sqrt(pixels) votes + margin).

    Parameters
    ----------
    pack : dict
        Filtered detection pack (Phase 2 schema).
    det_idx : torch.Tensor  shape (M,)
        Signed detection→winner index from `resolve`.
    obj_idx : torch.Tensor  shape (K,)
        Object indices from `resolve`.
    scene_state : dict
        Global scene state; mutated in-place.
    detection_image_ids : list
        Per-detection global image IDs built in `associate.resolve`.
    allow_new_objects : bool
        Whether unmatched detections create new objects.
    max_merge_distance_m : float or None
        Kill a proposed merge if Gaussian centres are further than this.
    label_min_score : float
        Min YOLOE conf for a detection to vote on the object label.
    label_margin_ratio : float
        Require top vote mass this much larger than runner-up to flip labels.
    label_use_pixel_weight : bool
        Weight votes by sqrt(num_pixels / 500) in addition to score.

    Returns
    -------
    update_info : dict
        Keys include ``new_object_indices``, ``n_new``, ``n_merged``.
    """
    means_d = pack.get("means", torch.empty(0, 3))
    cov6_d = pack.get("cov6", torch.empty(0, 6))
    feat_d = pack.get("features", torch.empty(0, 0))
    num_detections = int(means_d.shape[0]) if means_d.numel() > 0 else 0
    captions_d: List[str] = [""] * num_detections

    det_points_flat = pack.get("det_points_flat")
    det_points_offsets = pack.get("det_points_offsets")

    update_info = update_scene_graph_state(
        scene_state,
        means_d,
        cov6_d,
        feat_d,
        captions_d,
        det_idx,
        obj_idx,
        rgb_observations=None,
        detection_image_ids=detection_image_ids,
        allow_new_objects=allow_new_objects,
        det_points_flat=(
            det_points_flat
            if isinstance(det_points_flat, torch.Tensor)
            else None
        ),
        det_points_offsets=(
            det_points_offsets
            if isinstance(det_points_offsets, torch.Tensor)
            else None
        ),
        class_ids_d=pack.get("class_ids"),
        max_merge_distance_m=max_merge_distance_m,
    )
    update_info = update_info or {}

    # ---- collect visible object indices --------------------------------
    # Any existing object matched by a detection counts as visible.
    # Any brand-new object also counts as visible (it was observed this kf).
    visible_set: set[int] = set()

    if det_idx.numel() > 0 and obj_idx.numel() > 0:
        det_idx_cpu = det_idx.detach().to("cpu", dtype=torch.long)
        obj_idx_cpu = obj_idx.detach().to("cpu", dtype=torch.long)
        for d_i, winner_pos in enumerate(det_idx_cpu.tolist()):
            winner_pos = int(winner_pos)
            if 0 <= winner_pos < obj_idx_cpu.numel():
                visible_set.add(int(obj_idx_cpu[winner_pos].item()))

    new_obj_indices: List[int] = [
        int(x) for x in (update_info.get("new_object_indices") or [])
    ]
    visible_set.update(new_obj_indices)

    # ---- covisibility update ------------------------------------------
    if visible_set:
        n_current_objects = (
            int(scene_state["means"].shape[0])
            if isinstance(scene_state.get("means"), torch.Tensor)
            else None
        )
        update_covisibility_from_visible_indices(
            scene_state,
            visible_set,
            num_objects=n_current_objects,
        )

    # ---- post-update cannot-links ------------------------------------
    # Build det_to_obj map (det → final object index or None for unresolved)
    det_to_obj: List[Optional[int]] = [None] * num_detections
    if det_idx.numel() > 0 and obj_idx.numel() > 0:
        det_idx_cpu_l = det_idx.detach().to("cpu", dtype=torch.long).tolist()
        obj_idx_cpu_l = obj_idx.detach().to("cpu", dtype=torch.long).tolist()
        for d_i, wp in enumerate(det_idx_cpu_l):
            wp = int(wp)
            if 0 <= wp < len(obj_idx_cpu_l):
                det_to_obj[d_i] = int(obj_idx_cpu_l[wp])

    for d_i, wp_raw in enumerate(
        det_idx.detach().to("cpu", dtype=torch.long).tolist()
    ):
        if int(wp_raw) < 0 and d_i < len(new_obj_indices):
            pass  # already encoded; new-obj map handled below

    # Overwrite the new-detection slots with their freshly assigned indices
    new_det_indices = [
        int(i)
        for i, v in enumerate(
            det_idx.detach().to("cpu", dtype=torch.long).tolist()
        )
        if int(v) < 0
    ]
    for slot, new_obj_i in zip(new_det_indices, new_obj_indices):
        if 0 <= slot < len(det_to_obj):
            det_to_obj[slot] = int(new_obj_i)

    n_added = add_same_frame_cannot_links_from_detection_assignments(
        scene_state,
        detection_image_ids,
        det_to_obj,
    )
    update_info["same_frame_cannot_links_added"] = int(n_added)
    update_info["n_visible"] = len(visible_set)

    # ---- multi-frame class label voting (overrides FARM first-wins) ----
    vote_stats = apply_label_voting_after_fuse(
        scene_state,
        pack,
        det_to_obj,
        update_info,
        min_score=label_min_score,
        margin_ratio=label_margin_ratio,
        use_pixel_weight=label_use_pixel_weight,
    )
    update_info.update({f"label_vote_{k}": v for k, v in vote_stats.items()})

    return update_info
