#!/usr/bin/env python3
"""Phase 4d — post-caption identity merge."""

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
from post_caption_merge import MergeConfig, apply_post_caption_merges  # noqa: E402
from scene_io import caption_summary  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("run_phase4d")


def _default_scene(repo: Path) -> Path:
    for cand in [
        repo / "outputs" / "latest" / "phase4" / "scene_state_enriched.pt",
        repo / "outputs" / "latest" / "phase4" / "scene_state_captioned.pt",
    ]:
        if cand.is_file():
            return cand
    return repo / "outputs" / "latest" / "phase4" / "scene_state_enriched.pt"


def main() -> int:
    repo = _HERE.parent
    p = argparse.ArgumentParser(description="Phase 4d — post-caption identity merge")
    p.add_argument("--scene-state", type=Path, default=_default_scene(repo))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--output-name", type=str, default="scene_state_merged.pt")
    p.add_argument("--dry-run", action="store_true", help="Propose merges only; do not write scene state")
    p.add_argument("--hellinger-thresh", type=float, default=None)
    p.add_argument("--caption-thresh", type=float, default=None)
    p.add_argument("--visual-thresh", type=float, default=None)
    p.add_argument("--no-require-visual", action="store_true")
    p.add_argument("--max-candidates", type=int, default=0, help="Limit merge candidates (0 = all)")
    args = p.parse_args()

    if not args.scene_state.is_file():
        log.error("Missing scene state: %s", args.scene_state)
        return 2

    out = args.output_dir or args.scene_state.parent
    ss = load_scene(args.scene_state)

    config = MergeConfig.from_env()
    if args.hellinger_thresh is not None:
        config.hellinger_thresh = args.hellinger_thresh
    if args.caption_thresh is not None:
        config.caption_thresh = args.caption_thresh
    if args.visual_thresh is not None:
        config.visual_thresh = args.visual_thresh
    if args.no_require_visual:
        config.require_visual = False

    before = caption_summary(ss)
    stats = apply_post_caption_merges(
        ss,
        config,
        dry_run=args.dry_run,
        max_candidates=args.max_candidates,
    )
    after = caption_summary(ss) if not args.dry_run else before
    stats["caption_summary_before"] = before
    stats["caption_summary_after"] = after
    stats["input"] = str(args.scene_state)

    summary_path = out / "phase4d_summary.json"
    if args.dry_run:
        log.info("Dry run: %s", json.dumps({k: stats[k] for k in stats if k != "groups"}, indent=2))
    else:
        out_pt = out / args.output_name
        save_scene(out_pt, ss)
        stats["output"] = str(out_pt)
        summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        log.info("Wrote %s (merged %d objects)", out_pt, stats.get("n_merged_objects", 0))
        log.info("Active captions: keep %d → %d", before.get("n_keep", 0), after.get("n_keep", 0))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
