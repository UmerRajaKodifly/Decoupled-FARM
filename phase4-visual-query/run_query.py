#!/usr/bin/env python3
"""CLI — natural-language visual query over captioned scene state."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from build_query_index import load_vocab  # noqa: E402
from caption import load_scene  # noqa: E402
from gemini_client import GeminiClient  # noqa: E402
from query_parser import parse_query  # noqa: E402
from retrieval import retrieve  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("run_query")


def main() -> int:
    repo = _HERE.parent
    p = argparse.ArgumentParser(description="Visual query retrieval")
    p.add_argument("query", type=str, help="Natural language query")
    p.add_argument("--scene-state", type=Path, default=None)
    p.add_argument("--vocab-file", type=Path, default=repo / "vocab" / "construction_vocab.txt")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    if args.scene_state is None:
        for cand in [
            repo / "outputs" / "latest" / "phase4" / "scene_state_merged.pt",
            repo / "outputs" / "latest" / "phase4" / "scene_state_enriched.pt",
        ]:
            if cand.is_file():
                args.scene_state = cand
                break
        if args.scene_state is None:
            args.scene_state = repo / "outputs" / "latest" / "phase4" / "scene_state_merged.pt"

    if not args.scene_state.is_file():
        log.error("Missing scene state: %s", args.scene_state)
        return 2

    ss = load_scene(args.scene_state)
    vocab = load_vocab(args.vocab_file)
    client = GeminiClient()

    qg = parse_query(args.query, client=client)
    q_vecs = client.embed_texts([qg.target_description])
    q_vec = np.asarray(q_vecs[0], dtype=np.float64)

    hits = retrieve(ss, qg, q_vec, vocab=vocab, top_k=args.top_k)

    payload = {
        "query": args.query,
        "parsed": {
            "target_description": qg.target_description,
            "target_class": qg.target_class,
            "predicates": [{"name": pr.name, "args": pr.args} for pr in qg.predicates],
            "reasoning": qg.reasoning,
        },
        "results": [
            {
                "object_index": h.object_index,
                "score": round(h.score, 4),
                "semantic_score": round(h.semantic_score, 4),
                "predicate_score": round(h.predicate_score, 4),
                "label": h.label,
                "caption": h.caption,
                "category": h.category,
                "mean": list(h.mean),
                "reasons": h.reasons,
            }
            for h in hits
        ],
    }

    print(json.dumps(payload, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
