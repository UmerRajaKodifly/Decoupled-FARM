"""Parse natural-language queries into QueryGraph JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from gemini_client import GeminiClient
from prompts import QUERY_PARSER_SYSTEM, build_query_user_prompt


@dataclass
class Predicate:
    name: str
    args: List[str]
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryGraph:
    target_description: str
    predicates: List[Predicate]
    reasoning: str = ""
    target_class: Optional[str] = None


def _safe_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start : end + 1])
                return obj if isinstance(obj, dict) else {}
            except json.JSONDecodeError:
                pass
    return {}


def parse_query(query: str, *, client: Optional[GeminiClient] = None, mock: bool = False) -> QueryGraph:
    gem = client or GeminiClient(mock=mock)
    raw = gem.parse_json_text(system=QUERY_PARSER_SYSTEM, user=build_query_user_prompt(query))
    obj = _safe_json(raw)

    target = str(obj.get("target_description") or query).strip()
    target_class = obj.get("target_class")
    if isinstance(target_class, str):
        target_class = target_class.strip() or None
    else:
        target_class = None

    preds: List[Predicate] = []
    for item in obj.get("predicates") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        args_raw = item.get("args") or []
        args = [str(a) for a in args_raw] if isinstance(args_raw, list) else []
        preds.append(Predicate(name=name, args=args, kwargs=dict(item.get("kwargs") or {})))

    # Fixup closest/nearest wording
    q_lower = query.lower()
    if any(w in q_lower for w in ("closest", "nearest")):
        if not any(p.name == "Closest" for p in preds):
            preds.append(Predicate(name="Closest", args=["target"]))
    if "farthest" in q_lower and not any(p.name == "Farthest" for p in preds):
        preds.append(Predicate(name="Farthest", args=["target"]))

    return QueryGraph(
        target_description=target,
        target_class=target_class,
        predicates=preds,
        reasoning=str(obj.get("reasoning") or ""),
    )
