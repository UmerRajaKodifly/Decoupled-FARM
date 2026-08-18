"""Lightweight query API for the 3D viewer (uses query_index.json)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_P4 = _REPO / "phase4-visual-query"
if str(_P4) not in sys.path:
    sys.path.insert(0, str(_P4))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _load_index(data_dir: Path) -> dict:
    path = data_dir / "query_index.json"
    if not path.is_file():
        return {"version": 1, "n_objects": 0, "objects": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _keyword_score(query: str, obj: dict) -> float:
    q = query.lower()
    text = " ".join(
        [
            str(obj.get("caption") or ""),
            str(obj.get("category") or ""),
            str(obj.get("label") or ""),
            " ".join(obj.get("attributes") or []),
        ]
    ).lower()
    if not text.strip():
        return 0.0
    tokens = [t for t in q.split() if len(t) > 2]
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in text)
    return hits / len(tokens)


def _embed_query(text: str) -> Optional[np.ndarray]:
    try:
        from gemini_client import GeminiClient

        client = GeminiClient()
        vecs = client.embed_texts([text])
        if vecs and vecs[0]:
            return np.asarray(vecs[0], dtype=np.float64)
    except Exception as exc:
        import logging

        logging.getLogger("query_api").warning("Query embedding failed, falling back to keywords: %s", exc)
    return None


def _parse_query_simple(query: str) -> dict:
    try:
        from query_parser import parse_query

        qg = parse_query(query)
        return {
            "target_description": qg.target_description,
            "target_class": qg.target_class,
            "predicates": [{"name": p.name, "args": p.args} for p in qg.predicates],
            "reasoning": qg.reasoning,
        }
    except Exception as exc:
        import logging

        logging.getLogger("query_api").warning("Query parse failed, using raw string: %s", exc)
        return {"target_description": query, "target_class": None, "predicates": [], "reasoning": ""}


def _predicate_score(obj: dict, preds: List[dict], anchors: Dict[str, dict]) -> float:
    if not preds:
        return 1.0
    pos = np.asarray(obj.get("mean") or [0, 0, 0], dtype=np.float64)
    factors: List[float] = []
    for p in preds:
        name = p.get("name") or ""
        args = p.get("args") or []
        if name in {"IsCategory", "HasAttribute", "Closest", "Farthest"}:
            factors.append(1.0)
            continue
        anchor_key = args[1] if len(args) > 1 else (args[0] if args else "")
        anchor = anchors.get(str(anchor_key))
        if anchor is None:
            factors.append(0.85)
            continue
        ap = np.asarray(anchor.get("mean") or [0, 0, 0], dtype=np.float64)
        d = float(np.linalg.norm(pos - ap))
        dz = float(pos[2] - ap[2])
        if name in {"Near", "NextTo"}:
            factors.append(max(0.0, 1.0 - d / 3.0))
        elif name == "Above":
            factors.append(1.0 if dz > 0.3 and d < 6 else 0.4)
        elif name == "Below":
            factors.append(1.0 if dz < -0.3 and d < 6 else 0.4)
        else:
            factors.append(1.0)
    prod = 1.0
    for f in factors:
        prod *= max(f, 1e-6)
    return prod ** (1.0 / max(len(factors), 1))


def search(
    data_dir: Path,
    query: str,
    *,
    top_k: int = 15,
) -> dict:
    index = _load_index(data_dir)
    objects: List[dict] = list(index.get("objects") or [])
    if not objects:
        return {
            "query": query,
            "error": "query_index.json missing or empty — run Track B pipeline",
            "results": [],
        }

    parsed = _parse_query_simple(query)
    target = str(parsed.get("target_description") or query)
    q_vec = _embed_query(target)

    # Resolve anchors by keyword
    anchors: Dict[str, dict] = {}
    for p in parsed.get("predicates") or []:
        for arg in p.get("args") or []:
            if str(arg).lower() in {"target", "$target"}:
                continue
            key = str(arg)
            if key in anchors:
                continue
            best = max(objects, key=lambda o: _keyword_score(key, o), default=None)
            if best and _keyword_score(key, best) > 0:
                anchors[key] = best

    hits: List[dict] = []
    for obj in objects:
        if q_vec is not None and obj.get("embedding"):
            sem = _cosine(q_vec, np.asarray(obj["embedding"], dtype=np.float64))
        else:
            sem = _keyword_score(query, obj)
        pred = _predicate_score(obj, parsed.get("predicates") or [], anchors)
        score = sem * pred
        hits.append(
            {
                "object_index": obj.get("id"),
                "score": round(score, 4),
                "semantic_score": round(sem, 4),
                "predicate_score": round(pred, 4),
                "label": obj.get("label"),
                "caption": obj.get("caption"),
                "category": obj.get("category"),
                "mean": obj.get("mean"),
            }
        )

    if any(p.get("name") == "Closest" for p in parsed.get("predicates") or []):
        anchor_pos = None
        for p in parsed.get("predicates") or []:
            args = p.get("args") or []
            if len(args) > 1 and args[1] in anchors:
                anchor_pos = np.asarray(anchors[args[1]]["mean"], dtype=np.float64)
                break
        if anchor_pos is None:
            means = np.asarray([o.get("mean") or [0, 0, 0] for o in objects], dtype=np.float64)
            anchor_pos = means.mean(axis=0)
        for h in hits:
            d = float(np.linalg.norm(np.asarray(h["mean"], dtype=np.float64) - anchor_pos))
            h["score"] = round(h["semantic_score"] * max(0.05, 1.0 / (1.0 + d)), 4)

    hits.sort(key=lambda x: x["score"], reverse=True)

    return {
        "query": query,
        "parsed": parsed,
        "n_indexed": len(objects),
        "results": hits[:top_k],
    }
