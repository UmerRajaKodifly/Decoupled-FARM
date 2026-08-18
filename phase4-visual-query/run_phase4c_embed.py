#!/usr/bin/env python3
"""Phase 4c — embed caption text vectors."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from caption import load_scene, save_scene  # noqa: E402
from embed import embed_captions  # noqa: E402
from gemini_client import GeminiClient  # noqa: E402
from scene_io import overlay_embeddings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("run_phase4c")


def _default_scene(repo: Path) -> Path:
    for cand in [
        repo / "outputs" / "latest" / "phase4" / "scene_state_captioned.pt",
        repo / "outputs" / "latest" / "phase4" / "scene_state_with_crops.pt",
    ]:
        if cand.is_file():
            return cand
    return repo / "outputs" / "latest" / "phase4" / "scene_state_captioned.pt"


def main() -> int:
    repo = _HERE.parent
    p = argparse.ArgumentParser(description="Phase 4c — caption embeddings")
    p.add_argument("--scene-state", type=Path, default=_default_scene(repo))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--embed-model", type=str, default="text-embedding-004")
    args = p.parse_args()

    if not args.scene_state.is_file():
        log.error("Missing scene state: %s", args.scene_state)
        return 2

    out = args.output_dir or args.scene_state.parent
    cache = args.cache_dir or (out / "gemini_cache")
    ss = load_scene(args.scene_state)
    client = GeminiClient(cache_dir=cache, embed_model=args.embed_model)

    out_pt = out / "scene_state_enriched.pt"
    if out_pt.is_file() and not args.no_skip_existing:
        n_emb = overlay_embeddings(ss, load_scene(out_pt))
        log.info("Resumed %d embeddings from %s", n_emb, out_pt)

    stats = embed_captions(
        ss,
        client=client,
        batch_size=args.batch_size,
        skip_existing=not args.no_skip_existing,
        cache_dir=cache,
        checkpoint_path=out_pt,
    )

    save_scene(out_pt, ss)
    stats["output"] = str(out_pt)
    (out / "phase4c_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_pt)
    if stats.get("n_candidates", 0) > 0 and stats.get("n_embedded", 0) == 0:
        log.error("Phase 4c embedded 0 of %d candidates", stats["n_candidates"])
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
