"""Weighted multi-frame class-label voting for Phase 3 scene objects.

FARM's default update keeps the first non--1 YOLOE class forever. For open-
vocabulary construction detections that flip labels by viewpoint, we instead
accumulate score-weighted votes and take a (margin-stabilized) argmax.

State fields added on scene_state
---------------------------------
class_vote_mass : list[dict[int, float]]
    class_vote_mass[obj_i][class_id] = accumulated vote weight
class_vote_count : list[int]
    number of contributing detections per object (for diagnostics)

Not used for association — only for the stored ``class_ids`` label after fuse.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import torch


def ensure_vote_buffers(scene_state: dict, n_objects: int) -> None:
    """Grow (or create) per-object vote buffers to length *n_objects*."""
    mass = scene_state.get("class_vote_mass")
    if not isinstance(mass, list):
        mass = []
    counts = scene_state.get("class_vote_count")
    if not isinstance(counts, list):
        counts = []
    while len(mass) < n_objects:
        mass.append({})
    while len(counts) < n_objects:
        counts.append(0)
    scene_state["class_vote_mass"] = mass
    scene_state["class_vote_count"] = counts


def fold_merged_object_votes(
    scene_state: dict,
    merged_objects: Sequence[Dict[str, Any]],
) -> int:
    """Fold loser vote mass into winner after union-find object–object merges.

    Returns number of fold operations performed.
    """
    if not merged_objects:
        return 0
    mass: List[Dict[int, float]] = scene_state.get("class_vote_mass") or []
    counts: List[int] = scene_state.get("class_vote_count") or []
    n_fold = 0
    for info in merged_objects:
        try:
            loser = int(info["loser_idx"])
            winner = int(info["winner_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        if loser < 0 or winner < 0:
            continue
        ensure_vote_buffers(scene_state, max(loser, winner) + 1)
        mass = scene_state["class_vote_mass"]
        counts = scene_state["class_vote_count"]
        for cid, w in (mass[loser] or {}).items():
            mass[winner][int(cid)] = float(mass[winner].get(int(cid), 0.0)) + float(w)
        counts[winner] = int(counts[winner]) + int(counts[loser] or 0)
        mass[loser] = {}
        counts[loser] = 0
        n_fold += 1
    return n_fold


def _detection_weight(
    score: float,
    num_pixels: Optional[float],
    *,
    use_pixel_weight: bool,
    pixel_ref: float,
) -> float:
    s = max(0.0, float(score))
    if not use_pixel_weight or num_pixels is None:
        return s
    px = max(1.0, float(num_pixels))
    # sqrt dampens huge masks; 1.0 weight at pixel_ref pixels
    return s * math.sqrt(px / max(pixel_ref, 1.0))


def apply_class_votes(
    scene_state: dict,
    det_to_obj: List[Optional[int]],
    class_ids: Optional[torch.Tensor],
    scores: Optional[torch.Tensor],
    num_pixels: Optional[torch.Tensor] = None,
    *,
    min_score: float = 0.25,
    margin_ratio: float = 1.15,
    use_pixel_weight: bool = True,
    pixel_ref: float = 500.0,
    min_votes_to_override: int = 2,
) -> Dict[str, int]:
    """Add this batch's detection labels into vote masses and refresh class_ids.

    Parameters
    ----------
    det_to_obj
        Length M list mapping detection index → final object index (or None).
    class_ids, scores, num_pixels
        Per-detection tensors from the filtered pack.
    min_score
        Ignore detections below this YOLOE confidence.
    margin_ratio
        Only switch away from current leading class if new top mass
        >= margin_ratio * second mass (reduces label thrash).
    min_votes_to_override
        With only one vote so far, keep that class. After this many votes,
        margin rule applies for switching.

    Returns
    -------
    stats dict with n_votes_applied, n_labels_changed.
    """
    stats = {"n_votes_applied": 0, "n_labels_changed": 0, "n_objects_touched": 0}
    if not det_to_obj:
        return stats
    if class_ids is None or not isinstance(class_ids, torch.Tensor) or class_ids.numel() == 0:
        return stats

    class_ids_cpu = class_ids.detach().to("cpu", dtype=torch.long).view(-1)
    scores_cpu = None
    if isinstance(scores, torch.Tensor) and scores.numel() >= class_ids_cpu.numel():
        scores_cpu = scores.detach().to("cpu", dtype=torch.float32).view(-1)
    pixels_cpu = None
    if isinstance(num_pixels, torch.Tensor) and num_pixels.numel() >= class_ids_cpu.numel():
        pixels_cpu = num_pixels.detach().to("cpu", dtype=torch.float32).view(-1)

    means = scene_state.get("means")
    n_objects = int(means.shape[0]) if isinstance(means, torch.Tensor) else 0
    if n_objects == 0:
        return stats
    ensure_vote_buffers(scene_state, n_objects)
    mass: List[Dict[int, float]] = scene_state["class_vote_mass"]
    counts: List[int] = scene_state["class_vote_count"]
    class_state = scene_state.get("class_ids")
    if not isinstance(class_state, torch.Tensor) or class_state.numel() < n_objects:
        class_state = torch.full((n_objects,), -1, dtype=torch.long)
        scene_state["class_ids"] = class_state
    # ensure on CPU for simple indexing (caller may have cuda tensors)
    if class_state.device.type != "cpu":
        class_state = class_state.detach().cpu()
        scene_state["class_ids"] = class_state
    class_state = class_state.long()

    touched: set[int] = set()
    m = min(len(det_to_obj), int(class_ids_cpu.numel()))
    for d_i in range(m):
        obj_i = det_to_obj[d_i]
        if obj_i is None:
            continue
        obj_i = int(obj_i)
        if obj_i < 0 or obj_i >= n_objects:
            continue
        cid = int(class_ids_cpu[d_i].item())
        if cid < 0:
            continue
        sc = float(scores_cpu[d_i].item()) if scores_cpu is not None else 1.0
        if sc < float(min_score):
            continue
        px = float(pixels_cpu[d_i].item()) if pixels_cpu is not None else None
        w = _detection_weight(
            sc, px, use_pixel_weight=use_pixel_weight, pixel_ref=pixel_ref
        )
        if w <= 0.0:
            continue
        bucket = mass[obj_i]
        bucket[cid] = float(bucket.get(cid, 0.0)) + float(w)
        counts[obj_i] = int(counts[obj_i]) + 1
        touched.add(obj_i)
        stats["n_votes_applied"] += 1

    stats["n_objects_touched"] = len(touched)

    # Refresh class_ids from vote mass
    for obj_i in touched:
        bucket = mass[obj_i]
        if not bucket:
            continue
        # sort by mass desc
        ranked = sorted(bucket.items(), key=lambda kv: kv[1], reverse=True)
        top_c, top_m = int(ranked[0][0]), float(ranked[0][1])
        second_m = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        n_votes = int(counts[obj_i])
        try:
            cur = int(class_state[obj_i].item())
        except Exception:
            cur = -1

        new_c = top_c
        if cur >= 0 and cur != top_c:
            # Require enough evidence + margin before flipping an established label
            if n_votes < int(min_votes_to_override):
                new_c = cur
            elif second_m > 0.0 and top_m < float(margin_ratio) * second_m:
                # Top is not clearly ahead — keep current if current is still competitive
                cur_mass = float(bucket.get(cur, 0.0))
                if cur_mass > 0.0 and top_m < float(margin_ratio) * cur_mass:
                    new_c = cur
                elif top_m < float(margin_ratio) * second_m:
                    new_c = cur

        if new_c != cur:
            class_state[obj_i] = int(new_c)
            stats["n_labels_changed"] += 1

    # Keep class_ids on the same device as other scene tensors when possible
    device_ref = means.device if isinstance(means, torch.Tensor) else torch.device("cpu")
    if class_state.device != device_ref:
        scene_state["class_ids"] = class_state.to(device=device_ref)
    else:
        scene_state["class_ids"] = class_state

    return stats


def apply_label_voting_after_fuse(
    scene_state: dict,
    pack: dict,
    det_to_obj: List[Optional[int]],
    update_info: dict,
    *,
    min_score: float = 0.25,
    margin_ratio: float = 1.15,
    use_pixel_weight: bool = True,
) -> Dict[str, int]:
    """Convenience: fold object merges, then vote this pack's labels."""
    n_obj = (
        int(scene_state["means"].shape[0])
        if isinstance(scene_state.get("means"), torch.Tensor)
        else 0
    )
    ensure_vote_buffers(scene_state, n_obj)
    n_fold = fold_merged_object_votes(
        scene_state, update_info.get("merged_objects") or []
    )
    stats = apply_class_votes(
        scene_state,
        det_to_obj,
        pack.get("class_ids"),
        pack.get("scores"),
        pack.get("num_pixels"),
        min_score=min_score,
        margin_ratio=margin_ratio,
        use_pixel_weight=use_pixel_weight,
    )
    stats["n_merge_folds"] = int(n_fold)
    return stats
