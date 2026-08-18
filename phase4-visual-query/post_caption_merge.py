"""Phase 4d — post-caption identity merge (FARM-style, batch mode).

Proposes merges for captioned objects using:
  - Hellinger spatial proximity
  - cannot-link filter
  - category compatibility
  - caption embedding cosine similarity
  - visual similarity (DINO ``features`` by default; SigLIP2 if present)

Applies merges via ``update_scene_graph_state`` and merges caption histories.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch

_REPO = Path(__file__).resolve().parent.parent
_FARM_SRC = _REPO / "farm_src" / "src"
if str(_FARM_SRC) not in sys.path:
    sys.path.insert(0, str(_FARM_SRC))

from scene_graph.map_update.cannot_link import cannot_link_index_pairs  # noqa: E402
from scene_graph.map_update.get_neighbors import (  # noqa: E402
    DEFAULT_HELLINGER_MATCH_FLOOR,
    _cov6_to_matrix,
    _hellinger_distance,
)
from scene_graph.map_update.object_update import update_scene_graph_state  # noqa: E402

from scene_io import ensure_caption_fields, is_active, object_count  # noqa: E402

log = logging.getLogger("phase4d.merge")

_UNKNOWN_CATEGORIES = {"", "unknown", "object", "item", "thing", "part", "other"}
_CATEGORY_ALIASES = {
    "couch": "sofa",
    "settee": "sofa",
    "trash bin": "trash can",
    "garbage can": "trash can",
    "waste bin": "trash can",
    "rubbish bin": "trash can",
    "television": "tv",
    "fire detector": "fire alarm",
    "smoke alarm": "fire alarm",
    "smoke detector": "fire alarm",
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _canonical_category(value: object) -> str:
    text = str(value or "").strip().lower()
    text = " ".join(text.replace("_", " ").replace("-", " ").split())
    if text.endswith("s") and len(text) > 3:
        text = text[:-1]
    return _CATEGORY_ALIASES.get(text, text)


@dataclass
class MergeConfig:
    hellinger_thresh: float = 0.65
    caption_thresh: float = 0.92
    visual_thresh: float = 0.90
    require_visual: bool = True
    require_category_compat: bool = True
    max_merge_distance_m: float = 1.0
    spatial_prefilter_m: float = 2.5

    @classmethod
    def from_env(cls) -> "MergeConfig":
        return cls(
            hellinger_thresh=_env_float("CAPTION_MERGE_HELLINGER_THRESH", 0.65),
            caption_thresh=_env_float("CAPTION_MERGE_CAPTION_THRESH", 0.92),
            visual_thresh=_env_float(
                "CAPTION_MERGE_VISUAL_THRESH",
                _env_float("CAPTION_MERGE_SIGLIP2_THRESH", 0.90),
            ),
            require_visual=_env_bool("CAPTION_MERGE_REQUIRE_VISUAL", True),
            require_category_compat=_env_bool("CAPTION_MERGE_REQUIRE_CATEGORY_COMPAT", True),
            max_merge_distance_m=_env_float("CAPTION_MERGE_MAX_DISTANCE_M", 1.0),
            spatial_prefilter_m=_env_float("CAPTION_MERGE_SPATIAL_PREFILTER_M", 2.5),
        )


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union_min_wins(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def _categories_for_index(scene_state: dict, idx: int) -> Set[str]:
    categories_state: List[str] = scene_state.get("object_category", []) or []
    category_candidates_state: List[List[str]] = scene_state.get("object_category_candidates", []) or []
    out: Set[str] = set()
    if 0 <= idx < len(categories_state):
        out.add(_canonical_category(categories_state[idx]))
    if 0 <= idx < len(category_candidates_state):
        rows = category_candidates_state[idx]
        if isinstance(rows, (list, tuple)):
            out.update(_canonical_category(x) for x in rows)
    return {x for x in out if x not in _UNKNOWN_CATEGORIES}


def _supercategory_for_index(scene_state: dict, idx: int) -> str:
    supercategories_state: List[str] = scene_state.get("object_supercategory", []) or []
    if 0 <= idx < len(supercategories_state):
        return _canonical_category(supercategories_state[idx])
    return ""


def category_compatible(scene_state: dict, idx_a: int, idx_b: int, *, require: bool) -> bool:
    if not require:
        return True
    cats_a = _categories_for_index(scene_state, idx_a)
    cats_b = _categories_for_index(scene_state, idx_b)
    if cats_a and cats_b:
        return bool(cats_a & cats_b)
    super_a = _supercategory_for_index(scene_state, idx_a)
    super_b = _supercategory_for_index(scene_state, idx_b)
    if (
        super_a
        and super_b
        and super_a not in _UNKNOWN_CATEGORIES
        and super_b not in _UNKNOWN_CATEGORIES
    ):
        return super_a == super_b
    return True


def _normalize_rows(rows: List[List[float]], device: torch.device) -> Optional[torch.Tensor]:
    if not rows:
        return None
    t = torch.as_tensor(rows, device=device, dtype=torch.float32)
    return torch.nn.functional.normalize(t, p=2, dim=1, eps=1e-12)


def _caption_db_rows(
    scene_state: dict,
    idx: int,
    *,
    caption_dim: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    rows: List[List[float]] = []
    hist = scene_state.get("object_caption_embedding_history", []) or []
    if isinstance(hist, list) and idx < len(hist) and isinstance(hist[idx], list):
        for vec in hist[idx]:
            if isinstance(vec, (list, tuple)) and len(vec) == caption_dim:
                rows.append(list(vec))
    if not rows:
        emb = (scene_state.get("object_caption_embedding", []) or [])
        if idx < len(emb) and isinstance(emb[idx], (list, tuple)) and len(emb[idx]) == caption_dim:
            rows.append(list(emb[idx]))
    return _normalize_rows(rows, device)


def _visual_db_rows(
    scene_state: dict,
    idx: int,
    *,
    visual_dim: int,
    device: torch.device,
    features: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    rows: List[List[float]] = []
    for hist_key, scalar_key in (
        ("object_siglip2_embedding_history", "object_siglip2_embedding"),
        ("object_dino_embedding_history", "object_dino_embedding"),
    ):
        hist = scene_state.get(hist_key, []) or []
        if isinstance(hist, list) and idx < len(hist) and isinstance(hist[idx], list):
            for vec in hist[idx]:
                if isinstance(vec, (list, tuple)) and len(vec) == visual_dim:
                    rows.append(list(vec))
        scalars = scene_state.get(scalar_key, []) or []
        if not rows and idx < len(scalars) and isinstance(scalars[idx], (list, tuple)):
            if len(scalars[idx]) == visual_dim:
                rows.append(list(scalars[idx]))
        if rows:
            break
    if not rows and features is not None and 0 <= idx < features.shape[0]:
        vec = features[idx]
        if vec.numel() == visual_dim:
            rows.append(vec.detach().cpu().tolist())
    return _normalize_rows(rows, device)


def _candidate_indices(scene_state: dict) -> List[int]:
    ensure_caption_fields(scene_state)
    n = object_count(scene_state)
    out: List[int] = []
    for i in range(n):
        if not is_active(scene_state, i):
            continue
        if str(scene_state["object_caption_decision"][i] or "") != "keep":
            continue
        emb = scene_state["object_caption_embedding"][i]
        if not isinstance(emb, list) or len(emb) < 8:
            continue
        out.append(i)
    return out


def _observation_area(obs: object) -> float:
    if not isinstance(obs, dict):
        return 0.0
    for key in ("bbox_area", "area"):
        if key in obs:
            with contextlib.suppress(Exception):
                val = float(obs[key])
                if val > 0:
                    return val
    bbox = obs.get("bbox") or obs.get("box")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        with contextlib.suppress(Exception):
            x0, y0, x1, y1 = (float(bbox[i]) for i in range(4))
            return max(0.0, (x1 - x0) * (y1 - y0))
    return 0.0


def _max_caption_bbox_area(scene_state: dict, idx: int) -> float:
    rgb_rows = scene_state.get("rgb_observations", []) or []
    if idx >= len(rgb_rows) or not isinstance(rgb_rows[idx], list):
        return 0.0
    best = 0.0
    for obs in rgb_rows[idx]:
        best = max(best, _observation_area(obs))
    return best


def _choose_group_caption(scene_state: dict, indices: List[int]) -> str:
    best_caption = ""
    best_area = -1.0
    captions = scene_state.get("object_caption", []) or []
    for idx in indices:
        area = _max_caption_bbox_area(scene_state, idx)
        cap = str(captions[idx] or "").strip() if idx < len(captions) else ""
        if area > best_area and cap:
            best_area = area
            best_caption = cap
    if best_caption:
        return best_caption
    parts = [
        str(captions[idx] or "").strip()
        for idx in indices
        if idx < len(captions) and str(captions[idx] or "").strip()
    ]
    return max(parts, key=len, default="")


def _append_history_value(row: object, scalar_val: object) -> List[object]:
    out = list(row) if isinstance(row, list) else []
    if isinstance(scalar_val, str):
        if scalar_val and (not out or out[0] != scalar_val):
            out.insert(0, scalar_val)
    elif isinstance(scalar_val, (list, tuple)) and len(scalar_val) > 0:
        if not out or out[0] != list(scalar_val):
            out.insert(0, list(scalar_val))
    return out


def _merge_object_histories(scene_state: dict, *, winner_idx: int, loser_indices: List[int]) -> None:
    n = object_count(scene_state)
    ensure_caption_fields(scene_state)

    scalar_to_history = (
        ("object_caption", "object_caption_history"),
        ("object_caption_embedding", "object_caption_embedding_history"),
        ("object_siglip2_embedding", "object_siglip2_embedding_history"),
    )
    for scalar_key, history_key in scalar_to_history:
        scalars = scene_state.get(scalar_key, []) or []
        history = scene_state.get(history_key, []) or []
        while len(scalars) < n:
            scalars.append("" if scalar_key == "object_caption" else [])
        while len(history) < n:
            history.append([])
        for idx in [winner_idx, *loser_indices]:
            if idx < 0 or idx >= n:
                continue
            scalar_val = scalars[idx] if idx < len(scalars) else None
            row = history[idx] if idx < len(history) else []
            history[idx] = _append_history_value(row, scalar_val)
        scene_state[scalar_key] = scalars
        scene_state[history_key] = history

    for key in (
        "object_caption_history",
        "object_caption_embedding_history",
        "object_siglip2_embedding_history",
    ):
        rows = scene_state.get(key, []) or []
        while len(rows) < n:
            rows.append([])
        winner_row = list(rows[winner_idx]) if winner_idx < len(rows) and isinstance(rows[winner_idx], list) else []
        for idx in loser_indices:
            if idx < 0 or idx >= len(rows):
                continue
            loser_row = rows[idx] if isinstance(rows[idx], list) else []
            winner_row.extend(loser_row)
            rows[idx] = []
        rows[winner_idx] = winner_row
        scene_state[key] = rows

    _prune_representatives(scene_state, winner_idx)


def _prune_representatives(scene_state: dict, obj_idx: int, *, max_keep: int = 2) -> None:
    rgb_rows_all = scene_state.get("rgb_observations", []) or []
    rgb_row = (
        rgb_rows_all[obj_idx] if obj_idx < len(rgb_rows_all) and isinstance(rgb_rows_all[obj_idx], list) else []
    )
    if not rgb_row:
        return
    areas = [_observation_area(obs) for obs in rgb_row]
    primary_idx = max(range(len(rgb_row)), key=lambda i: (areas[i], -i)) if rgb_row else 0
    keep = [primary_idx]
    if max_keep > 1:
        for idx in range(len(rgb_row)):
            if idx != primary_idx:
                keep.append(idx)
                break
    keep = keep[:max_keep]
    for key in ("rgb_observations", "object_caption_history", "object_caption_embedding_history"):
        rows = scene_state.get(key, []) or []
        if obj_idx >= len(rows) or not isinstance(rows[obj_idx], list):
            continue
        row = rows[obj_idx]
        rows[obj_idx] = [row[i] for i in keep if 0 <= i < len(row)]
        scene_state[key] = rows


def _hellinger_neighbors(
    obj_idx: int,
    neighbor_indices: torch.Tensor,
    *,
    means: torch.Tensor,
    cov: torch.Tensor,
    hellinger_thresh: float,
) -> torch.Tensor:
    if neighbor_indices.numel() == 0:
        return neighbor_indices
    mu_d = means[obj_idx].unsqueeze(0).expand(neighbor_indices.shape[0], -1)
    cov_d = cov[obj_idx].unsqueeze(0).expand(neighbor_indices.shape[0], -1, -1)
    mu_n = means[neighbor_indices]
    cov_n = cov[neighbor_indices]
    with torch.amp.autocast("cuda", enabled=False):
        h2 = _hellinger_distance(
            mu_d.float(),
            cov_d.float(),
            mu_n.float(),
            cov_n.float(),
            DEFAULT_HELLINGER_MATCH_FLOOR,
        )
    keep = h2 < float(hellinger_thresh)
    return neighbor_indices[keep]


def propose_merge_pairs(
    scene_state: dict,
    config: MergeConfig,
    *,
    max_candidates: int = 0,
) -> List[Tuple[int, int]]:
    means = scene_state.get("means")
    cov6 = scene_state.get("cov6")
    features = scene_state.get("features")
    active_flags = scene_state.get("active")
    if not isinstance(means, torch.Tensor) or not isinstance(cov6, torch.Tensor):
        return []
    n = int(means.shape[0])
    if cov6.shape[0] != n:
        return []

    candidates = _candidate_indices(scene_state)
    if max_candidates > 0:
        candidates = candidates[:max_candidates]
    if len(candidates) < 2:
        return []

    device = means.device
    active_mask = None
    if isinstance(active_flags, torch.Tensor) and active_flags.numel() >= n:
        active_mask = active_flags.to(device=device, dtype=torch.bool)

    cov = _cov6_to_matrix(cov6)
    finite = torch.isfinite(cov).all(dim=(1, 2)) & torch.isfinite(means).all(dim=1)
    posdiag = (torch.diagonal(cov, dim1=-2, dim2=-1) > 0).all(dim=1)
    valid = finite & posdiag
    if active_mask is not None:
        valid = valid & active_mask

    valid_indices = torch.nonzero(valid, as_tuple=False).view(-1)
    if valid_indices.numel() == 0:
        return []

    caption_dim = 0
    for idx in candidates:
        emb = scene_state["object_caption_embedding"][idx]
        if isinstance(emb, list) and len(emb) > caption_dim:
            caption_dim = len(emb)
    visual_dim = int(features.shape[1]) if isinstance(features, torch.Tensor) and features.ndim == 2 else 0

    caption_db: Dict[int, torch.Tensor] = {}
    for idx in valid_indices.tolist():
        rows = _caption_db_rows(scene_state, idx, caption_dim=caption_dim, device=device)
        if rows is not None:
            caption_db[idx] = rows

    visual_db: Dict[int, torch.Tensor] = {}
    feat_norm = None
    if visual_dim > 0 and isinstance(features, torch.Tensor) and features.shape[0] == n:
        feat_norm = torch.nn.functional.normalize(
            features.to(device=device, dtype=torch.float32),
            p=2,
            dim=1,
            eps=1e-12,
        )
        for idx in valid_indices.tolist():
            rows = _visual_db_rows(scene_state, idx, visual_dim=visual_dim, device=device, features=feat_norm)
            if rows is not None:
                visual_db[idx] = rows

    cand_tensor = torch.as_tensor(candidates, device=device, dtype=torch.long)
    db_means = means[valid_indices]
    dist = torch.cdist(means[cand_tensor].float(), db_means.float())
    radius = float(config.spatial_prefilter_m)

    pairs: List[Tuple[int, int]] = []
    blocked_pairs = cannot_link_index_pairs(scene_state, n_objects=n)

    def _cannot_link(a: int, b: int) -> bool:
        if a == b:
            return False
        lo, hi = (a, b) if a < b else (b, a)
        return (lo, hi) in blocked_pairs

    for det_local, obj_idx in enumerate(candidates):
        near_local = torch.nonzero(dist[det_local] <= radius, as_tuple=False).view(-1)
        if near_local.numel() == 0:
            continue
        neighbor_indices = valid_indices[near_local]
        neighbor_indices = neighbor_indices[neighbor_indices != obj_idx]
        if neighbor_indices.numel() == 0:
            continue
        neighbor_indices = _hellinger_neighbors(
            obj_idx,
            neighbor_indices,
            means=means,
            cov=cov,
            hellinger_thresh=config.hellinger_thresh,
        )
        if neighbor_indices.numel() == 0:
            continue

        det_caption: Optional[torch.Tensor] = None
        emb = scene_state["object_caption_embedding"][obj_idx]
        if isinstance(emb, (list, tuple)) and len(emb) == caption_dim and caption_dim > 0:
            det_caption = torch.nn.functional.normalize(
                torch.as_tensor(emb, device=device, dtype=torch.float32).unsqueeze(0),
                p=2,
                dim=1,
                eps=1e-12,
            )

        det_visual: Optional[torch.Tensor] = None
        if visual_dim > 0 and isinstance(features, torch.Tensor) and obj_idx < features.shape[0]:
            det_visual = torch.nn.functional.normalize(
                features[obj_idx].to(device=device, dtype=torch.float32).unsqueeze(0),
                p=2,
                dim=1,
                eps=1e-12,
            )

        for neighbor_idx in neighbor_indices.tolist():
            if neighbor_idx == obj_idx or neighbor_idx < 0 or neighbor_idx >= n:
                continue
            if _cannot_link(int(obj_idx), int(neighbor_idx)):
                continue
            if not category_compatible(scene_state, obj_idx, neighbor_idx, require=config.require_category_compat):
                continue

            lang_pass = False
            visual_pass = False
            has_caption = False
            has_visual = False

            if det_caption is not None:
                db_caption = caption_db.get(int(neighbor_idx))
                if db_caption is not None and db_caption.numel() > 0:
                    has_caption = True
                    sim = torch.sum(db_caption * det_caption, dim=1)
                    if sim.numel() > 0 and bool((sim >= config.caption_thresh).any().item()):
                        lang_pass = True

            if det_visual is not None:
                db_visual = visual_db.get(int(neighbor_idx))
                if db_visual is not None and db_visual.numel() > 0:
                    has_visual = True
                    sim = torch.sum(db_visual * det_visual, dim=1)
                    if sim.numel() > 0 and bool((sim >= config.visual_thresh).any().item()):
                        visual_pass = True

            if config.require_visual:
                merge_pass = lang_pass and visual_pass
            elif has_caption and has_visual:
                merge_pass = lang_pass and visual_pass
            else:
                merge_pass = lang_pass or visual_pass

            if merge_pass:
                a, b = int(obj_idx), int(neighbor_idx)
                pairs.append((a, b) if a < b else (b, a))

    return sorted(set(pairs))


def _groups_from_pairs(pairs: List[Tuple[int, int]], n: int) -> Dict[int, List[int]]:
    uf = _UnionFind(n)
    for a, b in pairs:
        uf.union_min_wins(a, b)
    grouped: Dict[int, Set[int]] = {}
    for a, b in pairs:
        for idx in (a, b):
            root = uf.find(idx)
            grouped.setdefault(root, set()).add(idx)
    return {root: sorted(members) for root, members in grouped.items() if len(members) > 1}


def apply_post_caption_merges(
    scene_state: dict,
    config: Optional[MergeConfig] = None,
    *,
    dry_run: bool = False,
    max_candidates: int = 0,
) -> dict:
    """Propose and apply caption-driven identity merges. Returns summary stats."""
    config = config or MergeConfig.from_env()
    n = object_count(scene_state)
    if n == 0:
        return {"n_proposed_pairs": 0, "n_groups": 0, "n_merged_objects": 0, "dry_run": dry_run}

    pairs = propose_merge_pairs(scene_state, config, max_candidates=max_candidates)
    groups = _groups_from_pairs(pairs, n)
    if not groups:
        log.info("No post-caption merge groups proposed")
        return {
            "n_proposed_pairs": len(pairs),
            "n_groups": 0,
            "n_merged_objects": 0,
            "dry_run": dry_run,
            "config": config.__dict__,
        }

    merged_object_count = sum(len(m) - 1 for m in groups.values())
    log.info(
        "Proposed %d merge pairs → %d groups (%d objects absorbed)",
        len(pairs),
        len(groups),
        merged_object_count,
    )

    if dry_run:
        return {
            "n_proposed_pairs": len(pairs),
            "n_groups": len(groups),
            "n_merged_objects": merged_object_count,
            "groups": groups,
            "dry_run": True,
            "config": config.__dict__,
        }

    means = scene_state["means"]
    features = scene_state.get("features")
    cov6 = scene_state["cov6"]
    if features is None:
        raise ValueError("scene_state missing features tensor — required for merge")

    is_locked = scene_state.get("is_locked") or []
    device = means.device
    obj_winner_idx = torch.arange(n, dtype=torch.long, device=device)
    merge_ops: List[Tuple[int, List[int]]] = []

    for _root, members in sorted(groups.items(), key=lambda kv: kv[0]):
        winner_idx = min(members)
        for idx in members:
            if idx < len(is_locked) and bool(is_locked[idx]):
                winner_idx = -1
                break
        if winner_idx < 0:
            continue
        losers = [idx for idx in members if idx != winner_idx]
        if not losers:
            continue
        for idx in losers:
            obj_winner_idx[idx] = winner_idx
        merge_ops.append((winner_idx, losers))

    if not merge_ops:
        return {
            "n_proposed_pairs": len(pairs),
            "n_groups": len(groups),
            "n_merged_objects": 0,
            "dry_run": False,
            "config": config.__dict__,
        }

    zero_long = torch.zeros((0,), dtype=torch.long, device=device)
    zero_feat = features.new_zeros((0, features.shape[1]))
    zero_mu = means.new_zeros((0, 3))
    zero_cov6 = cov6.new_zeros((0, 6))

    update_scene_graph_state(
        scene_state,
        zero_mu,
        zero_cov6,
        zero_feat,
        [],
        zero_long,
        obj_winner_idx,
        max_merge_distance_m=float(config.max_merge_distance_m),
    )

    captions = scene_state.get("object_caption", []) or []
    categories = scene_state.get("object_category", []) or []
    supercategories = scene_state.get("object_supercategory", []) or []
    attributes = scene_state.get("object_key_attributes", []) or []
    decisions = scene_state.get("object_caption_decision", []) or []

    applied = 0
    for winner_idx, losers in merge_ops:
        _merge_object_histories(scene_state, winner_idx=winner_idx, loser_indices=losers)
        group = [winner_idx, *losers]
        chosen = _choose_group_caption(scene_state, group)
        if chosen and winner_idx < len(captions):
            captions[winner_idx] = chosen
        applied += len(losers)
        log.debug("Merged %s → winner %d", losers, winner_idx)

    scene_state["object_caption"] = captions
    scene_state["object_category"] = categories
    scene_state["object_supercategory"] = supercategories
    scene_state["object_key_attributes"] = attributes
    scene_state["object_caption_decision"] = decisions

    active = scene_state.get("active")
    n_active = int(active.sum().item()) if isinstance(active, torch.Tensor) else n
    return {
        "n_proposed_pairs": len(pairs),
        "n_groups": len(merge_ops),
        "n_merged_objects": applied,
        "n_active_after": n_active,
        "dry_run": False,
        "config": config.__dict__,
    }
