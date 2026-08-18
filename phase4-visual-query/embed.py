"""Phase 4c — caption text embeddings."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

from caption import save_scene
from gemini_client import GeminiClient
from scene_io import REAL_EMBED_MIN_DIM, ensure_caption_fields, is_active, object_count

log = logging.getLogger("phase4c.embed")


def embed_captions(
    scene_state: dict,
    *,
    client: Optional[GeminiClient] = None,
    batch_size: int = 16,
    only_kept: bool = True,
    skip_existing: bool = True,
    cache_dir=None,
    checkpoint_path: Optional[Path] = None,
) -> dict:
    ensure_caption_fields(scene_state)
    n = object_count(scene_state)
    gem = client or GeminiClient(cache_dir=cache_dir)

    indices: List[int] = []
    texts: List[str] = []
    for i in range(n):
        if not is_active(scene_state, i):
            continue
        decision = str(scene_state["object_caption_decision"][i] or "")
        if only_kept and decision != "keep":
            continue
        text = str(scene_state["object_caption"][i] or "").strip()
        if not text:
            continue
        if skip_existing:
            existing = scene_state["object_caption_embedding"][i]
            if isinstance(existing, list) and len(existing) >= REAL_EMBED_MIN_DIM:
                continue
        indices.append(i)
        texts.append(text)

    log.info("Embedding %d caption texts …", len(texts))
    t0 = time.time()
    embedded = 0

    def _checkpoint(reason: str) -> None:
        if checkpoint_path is None or embedded <= 0:
            return
        save_scene(Path(checkpoint_path), scene_state)
        log.info("Embedding checkpoint (%s) %s n=%d", reason, checkpoint_path, embedded)

    try:
        for start in range(0, len(texts), batch_size):
            chunk_idx = indices[start : start + batch_size]
            chunk_txt = texts[start : start + batch_size]
            vecs = gem.embed_texts(chunk_txt)
            for i, vec in zip(chunk_idx, vecs):
                if not isinstance(vec, list) or len(vec) < REAL_EMBED_MIN_DIM:
                    raise RuntimeError(f"object {i}: embedding missing or too short")
                scene_state["object_caption_embedding"][i] = vec
                embedded += 1
            _checkpoint("batch")
    finally:
        _checkpoint("exit")
    log.info(
        "Embedded %d objects in %.1fs api_calls=%s cache_hits=%s",
        embedded,
        time.time() - t0,
        getattr(gem, "n_api_calls", "?"),
        getattr(gem, "n_cache_hits", "?"),
    )
    return {"n_embedded": embedded, "n_candidates": len(texts)}
