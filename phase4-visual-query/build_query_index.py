"""Export lightweight query index for the 3D viewer API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch

from scene_io import ensure_caption_fields, is_active, object_count


def build_query_index(
    scene_state: dict,
    vocab: List[str],
    *,
    embed_model: str = "text-embedding-004",
) -> dict:
    ensure_caption_fields(scene_state)
    n = object_count(scene_state)
    objects = []
    for i in range(n):
        if not is_active(scene_state, i):
            continue
        decision = str(scene_state["object_caption_decision"][i] or "")
        if decision == "drop":
            continue
        emb = scene_state["object_caption_embedding"][i]
        if not isinstance(emb, list) or len(emb) < 8:
            continue
        cid = -1
        cids = scene_state.get("class_ids")
        if isinstance(cids, torch.Tensor) and i < cids.shape[0]:
            cid = int(cids[i].item())
        label = vocab[cid] if 0 <= cid < len(vocab) else f"obj{i}"
        mean = scene_state["means"][i].tolist()
        objects.append(
            {
                "id": i,
                "label": label,
                "category": str(scene_state["object_category"][i] or ""),
                "supercategory": str(scene_state["object_supercategory"][i] or ""),
                "caption": str(scene_state["object_caption"][i] or ""),
                "attributes": list(scene_state["object_key_attributes"][i] or []),
                "mean": [round(float(x), 4) for x in mean],
                "embedding": [float(x) for x in emb],
            }
        )
    return {
        "version": 1,
        "embed_model": embed_model,
        "n_objects": len(objects),
        "objects": objects,
    }


def write_query_index(path: Path, index: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")


def load_vocab(path: Path) -> List[str]:
    if not path.is_file():
        return []
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
