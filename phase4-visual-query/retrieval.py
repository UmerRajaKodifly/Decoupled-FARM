"""Geometric + semantic scoring for visual query."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from query_parser import Predicate, QueryGraph
from scene_io import is_active, object_count


@dataclass
class ScoredHit:
    object_index: int
    score: float
    semantic_score: float = 0.0
    predicate_score: float = 1.0
    label: str = ""
    caption: str = ""
    category: str = ""
    mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    reasons: List[str] = field(default_factory=list)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _object_mean(scene_state: dict, idx: int) -> np.ndarray:
    means = scene_state["means"]
    if isinstance(means, torch.Tensor) and 0 <= idx < means.shape[0]:
        return means[idx].detach().cpu().numpy().astype(np.float64)
    return np.zeros(3, dtype=np.float64)


def _object_label(scene_state: dict, idx: int, vocab: List[str]) -> str:
    cids = scene_state.get("class_ids")
    if isinstance(cids, torch.Tensor) and 0 <= idx < cids.shape[0]:
        cid = int(cids[idx].item())
        if 0 <= cid < len(vocab):
            return vocab[cid]
    cat = scene_state.get("object_category") or []
    if idx < len(cat) and cat[idx]:
        return str(cat[idx])
    return f"obj{idx}"


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _semantic_scores(
    scene_state: dict,
    query_vec: np.ndarray,
    *,
    vocab: List[str],
    class_mismatch_floor: float = 0.3,
    target_class: Optional[str] = None,
) -> Dict[int, float]:
    n = object_count(scene_state)
    embs = scene_state.get("object_caption_embedding") or []
    decisions = scene_state.get("object_caption_decision") or []
    scores: Dict[int, float] = {}

    tc = (target_class or "").strip().lower()
    for i in range(n):
        if not is_active(scene_state, i):
            continue
        if i < len(decisions) and str(decisions[i]) == "drop":
            continue
        vec = embs[i] if i < len(embs) else []
        if not isinstance(vec, list) or len(vec) < 8:
            continue
        s = _cosine(query_vec, np.asarray(vec, dtype=np.float64))
        if tc:
            label = _object_label(scene_state, i, vocab).lower()
            cat = str((scene_state.get("object_category") or [""] * n)[i] or "").lower()
            if tc not in label and tc not in cat and label != tc:
                s *= class_mismatch_floor
        scores[i] = s
    return scores


def _resolve_anchor_index(
    scene_state: dict,
    anchor_phrase: str,
    vocab: List[str],
    anchor_vec: Optional[np.ndarray],
) -> Optional[int]:
    """Best object index matching anchor phrase by embedding or label."""
    if not anchor_phrase or anchor_phrase.lower() in {"target", "$target"}:
        return None
    phrase = anchor_phrase.strip().lower()
    n = object_count(scene_state)
    # Label substring match first
    for i in range(n):
        if not is_active(scene_state, i):
            continue
        lab = _object_label(scene_state, i, vocab).lower()
        cat = str((scene_state.get("object_category") or [""] * n)[i] or "").lower()
        if phrase in lab or phrase in cat or lab in phrase:
            return i
    if anchor_vec is not None:
        sem = _semantic_scores(scene_state, anchor_vec, vocab=vocab, target_class=None)
        if sem:
            return max(sem.items(), key=lambda kv: kv[1])[0]
    return None


def _predicate_factor(
    scene_state: dict,
    obj_idx: int,
    preds: List[Predicate],
    *,
    vocab: List[str],
    anchor_indices: Dict[str, int],
    near_thresh_m: float = 3.0,
) -> Tuple[float, List[str]]:
    if not preds:
        return 1.0, []
    pos = _object_mean(scene_state, obj_idx)
    factors: List[float] = []
    reasons: List[str] = []

    for p in preds:
        name = p.name
        if name in {"IsCategory", "HasAttribute"}:
            factors.append(1.0)
            continue
        if name == "Closest":
            # handled globally after sorting by distance to query anchor
            factors.append(1.0)
            continue
        if name == "Farthest":
            factors.append(1.0)
            continue

        anchor_phrase = p.args[1] if len(p.args) > 1 else (p.args[0] if p.args else "")
        anchor_idx = anchor_indices.get(anchor_phrase)
        if anchor_idx is None:
            factors.append(0.85)  # soft penalty — anchor not resolved
            reasons.append(f"{name}:anchor_unresolved")
            continue
        anchor_pos = _object_mean(scene_state, anchor_idx)
        d = _dist(pos, anchor_pos)
        dz = float(pos[2] - anchor_pos[2])

        if name in {"Near", "NextTo"}:
            score = max(0.0, 1.0 - d / max(near_thresh_m, 0.5))
            factors.append(score)
            reasons.append(f"Near:d={d:.2f}m")
        elif name == "Above":
            score = 1.0 if dz > 0.3 and d < near_thresh_m * 2 else max(0.0, 0.5 - abs(dz) * 0.1)
            factors.append(score)
            reasons.append(f"Above:dz={dz:.2f}m")
        elif name == "Below":
            score = 1.0 if dz < -0.3 and d < near_thresh_m * 2 else max(0.0, 0.5 - abs(dz) * 0.1)
            factors.append(score)
            reasons.append(f"Below:dz={dz:.2f}m")
        elif name == "On":
            horiz = math.sqrt((pos[0] - anchor_pos[0]) ** 2 + (pos[1] - anchor_pos[1]) ** 2)
            score = 1.0 if abs(dz) < 0.5 and horiz < 1.5 else max(0.0, 1.0 - horiz / 3.0)
            factors.append(score)
            reasons.append(f"On:horiz={horiz:.2f}m dz={dz:.2f}m")
        else:
            factors.append(1.0)

    if not factors:
        return 1.0, reasons
    # geometric mean
    prod = 1.0
    for f in factors:
        prod *= max(f, 1e-6)
    return prod ** (1.0 / len(factors)), reasons


def retrieve(
    scene_state: dict,
    qg: QueryGraph,
    query_vec: np.ndarray,
    *,
    vocab: List[str],
    top_k: int = 20,
    near_thresh_m: float = 3.0,
) -> List[ScoredHit]:
    sem = _semantic_scores(
        scene_state,
        query_vec,
        vocab=vocab,
        target_class=qg.target_class,
    )
    if not sem:
        return []

    # Resolve anchor objects for predicates
    anchor_indices: Dict[str, int] = {}
    for p in qg.predicates:
        for arg in p.args:
            if arg.lower() in {"target", "$target"}:
                continue
            if arg not in anchor_indices:
                # embed anchor phrase on the fly using mean of matching objects — caller may pass precomputed
                anchor_indices[arg] = _resolve_anchor_index(scene_state, arg, vocab, query_vec)  # type: ignore

    hits: List[ScoredHit] = []
    captions = scene_state.get("object_caption") or []
    categories = scene_state.get("object_category") or []
    n = object_count(scene_state)

    for idx, sem_score in sem.items():
        pred_score, reasons = _predicate_factor(
            scene_state, idx, qg.predicates, vocab=vocab, anchor_indices=anchor_indices, near_thresh_m=near_thresh_m
        )
        combined = sem_score * pred_score
        m = _object_mean(scene_state, idx)
        hits.append(
            ScoredHit(
                object_index=idx,
                score=combined,
                semantic_score=sem_score,
                predicate_score=pred_score,
                label=_object_label(scene_state, idx, vocab),
                caption=str(captions[idx] if idx < len(captions) else ""),
                category=str(categories[idx] if idx < len(categories) else ""),
                mean=(float(m[0]), float(m[1]), float(m[2])),
                reasons=reasons,
            )
        )

    # Closest: re-rank by distance to anchor or scene centroid
    if any(p.name == "Closest" for p in qg.predicates):
        anchor_pos = None
        for p in qg.predicates:
            if p.name in {"Near", "NextTo", "On"} and len(p.args) > 1:
                ai = anchor_indices.get(p.args[1])
                if ai is not None:
                    anchor_pos = _object_mean(scene_state, ai)
                    break
        if anchor_pos is None:
            active_means = np.stack([_object_mean(scene_state, i) for i in sem.keys()])
            anchor_pos = active_means.mean(axis=0)
        for h in hits:
            d = _dist(np.array(h.mean), anchor_pos)
            h.score = h.semantic_score * max(0.05, 1.0 / (1.0 + d))
            h.reasons.append(f"Closest:d={d:.2f}m")

    hits.sort(key=lambda x: x.score, reverse=True)
    return hits[:top_k]
