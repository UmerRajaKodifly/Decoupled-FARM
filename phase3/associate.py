"""Phase 3 — neighbor lookup and correspondence resolution.

Thin wrappers around FARM's `find_neighbors_for_detections` and
`resolve_correspondence`, with a single 360-specific adaptation:

    detection_image_ids[i] = kf_index * 4 + face_index

so that cannot-link and one-to-one constraints are enforced **per face**, not
per keyframe.  Two faces from the same keyframe are allowed to independently
match the same world object.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

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

from scene_graph.pipeline.steps import (  # noqa: E402
    compute_detection_image_ids,
    find_neighbors_for_detections,
    resolve_correspondence,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_neighbors(
    pack: dict,
    scene_state: dict,
    *,
    feature_sim_thresh: float = 0.5,
    hellinger_thresh: float = 0.8,
) -> Tuple[list, list]:
    """Return (neighbors, k_neighbors) for all detections in *pack*.

    Delegates to FARM's `find_neighbors_for_detections`, which handles
    device alignment between the detection tensors and the scene state.

    Parameters
    ----------
    pack : dict
        Filtered Phase 2/3 detection pack (same schema as seg_outputs).
    scene_state : dict
        Current global SceneState (may be empty on first keyframe).
    feature_sim_thresh : float
        Minimum cosine similarity for a feature match.
    hellinger_thresh : float
        Maximum Hellinger distance for a Gaussian match.

    Returns
    -------
    neighbors : list of lists
        ``neighbors[d]`` is the list of (object_index, score) pairs that
        detection *d* is considered a neighbour of.
    k_neighbors : list
        Raw kNN distances per detection (for diagnostics).
    """
    return find_neighbors_for_detections(
        pack,
        scene_state,
        feature_sim_thresh=feature_sim_thresh,
        hellinger_thresh=hellinger_thresh,
    )


def resolve(
    pack: dict,
    neighbors: list,
    scene_state: dict,
    kf_index: int,
    *,
    same_image_one_to_one: bool = True,
    assignment_mode: str = "union_all",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resolve detection→object correspondence using union-find.

    Builds per-face global image IDs:
        ``global_id = kf_index * 4 + face_index``

    These IDs are passed to FARM's `resolve_correspondence` so that:
      - cannot-link constraints are enforced per face
      - one-to-one enforcement prevents two detections on the same face from
        matching the same world object
      - two *different* faces of the same keyframe may independently match the
        same world object (seams / partial views)

    Parameters
    ----------
    pack : dict
        Filtered detection pack; must contain ``batch_ids``.
    neighbors : list
        Output of :func:`find_neighbors`.
    scene_state : dict
        Current SceneState; used for cannot-link lookup and preliminary
        cannot-link updates (via FARM internals).
    kf_index : int
        0-based keyframe index used to build globally unique image IDs.
    same_image_one_to_one : bool
        Enforce one-to-one assignment within each face.
    assignment_mode : str
        Union-find mode passed to FARM; ``"union_all"`` is the default.

    Returns
    -------
    det_idx : torch.Tensor  shape (M,)
        Signed index for each detection:
        - ``>= 0`` → matched to existing object at ``obj_idx[det_idx[i]]``
        - ``< 0``  → new object
    obj_idx : torch.Tensor  shape (K,)
        Object indices referenced by *det_idx*.
    """
    num_detections = int(pack["means"].shape[0]) if pack.get("means") is not None else 0
    object_count = (
        int(scene_state["means"].shape[0])
        if isinstance(scene_state.get("means"), torch.Tensor)
        else 0
    )

    # Build batch_image_ids: one entry per face in this pack
    #   face_index 0..3  →  global_id = kf_index * 4 + face_index
    num_faces = len(pack.get("face_meta") or pack.get("poses_world") or [])
    batch_image_ids = [kf_index * 4 + face_i for face_i in range(max(num_faces, 1))]

    detection_image_ids: List[Optional[int]] = compute_detection_image_ids(
        pack, batch_image_ids, num_detections
    )

    det_idx, obj_idx = resolve_correspondence(
        neighbors,
        object_count,
        scene_state=scene_state,
        detection_image_ids=detection_image_ids,
        seg_outputs=pack,
        same_image_one_to_one=same_image_one_to_one,
        assignment_mode=assignment_mode,
    )
    return det_idx, obj_idx, detection_image_ids
