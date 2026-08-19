#!/usr/bin/env python3
"""Phase 4b — structured captioning on full perspective faces + bbox (HK vLLM)."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from caption import load_scene, run_captioning, save_caption_manifest, save_scene  # noqa: E402
from vlm_client import DEFAULT_VL_MODEL, VlmClient  # noqa: E402
from scene_io import caption_summary  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("run_phase4b")


def _default_scene(repo: Path) -> Path:
    for cand in [
        repo / "outputs" / "latest" / "phase4" / "scene_state_with_crops.pt",
        repo / "outputs" / "phase4" / "scene_state_with_crops.pt",
    ]:
        if cand.is_file():
            return cand
    runs = sorted((repo / "outputs" / "runs").glob("run_*/phase4/scene_state_with_crops.pt"))
    if runs:
        return runs[-1]
    return repo / "outputs" / "latest" / "phase4" / "scene_state_with_crops.pt"


def main() -> int:
    repo = _HERE.parent
    p = argparse.ArgumentParser(description="Phase 4b — VLM captioning (HK vLLM)")
    p.add_argument("--scene-state", type=Path, default=_default_scene(repo))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--vocab-file", type=Path, default=repo / "vocab" / "construction_vocab.txt")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--max-objects", type=int, default=0)
    p.add_argument("--include-inactive", action="store_true")
    p.add_argument("--no-skip-existing", action="store_true")
    p.add_argument("--crops-dir", type=Path, default=None, help="Phase 4a crops/ fallback if a face is missing")
    p.add_argument("--faces-dir", type=Path, default=None, help="Phase 1.5 faces/ (default: run_dir/phase1.5/faces)")
    p.add_argument("--caption-model", type=str, default=DEFAULT_VL_MODEL)
    p.add_argument("--fail-fast", action="store_true", help="Abort the batch on the first caption API error")
    p.add_argument("--use-crops", action="store_true", help="Send padded crops instead of full faces")
    p.add_argument("--checkpoint-every", type=int, default=25, help="Save scene_state_captioned.pt every N new captions")
    args = p.parse_args()

    import os
    if os.environ.get("MAX_OBJECTS"):
        with contextlib.suppress(ValueError):
            args.max_objects = int(os.environ["MAX_OBJECTS"])
    if os.environ.get("CHECKPOINT_EVERY"):
        with contextlib.suppress(ValueError):
            args.checkpoint_every = int(os.environ["CHECKPOINT_EVERY"])

    if not args.scene_state.is_file():
        log.error("Missing scene state: %s (run Phase 4a first)", args.scene_state)
        return 2

    out = args.output_dir or args.scene_state.parent
    out.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir or (out / "vlm_cache")

    log.info("Loading %s", args.scene_state)
    ss = load_scene(args.scene_state)
    crops_dir = args.crops_dir or (args.scene_state.parent / "crops")
    faces_dir = args.faces_dir
    if faces_dir is None:
        run_dir = args.scene_state.parent.parent
        cand = run_dir / "phase1.5" / "faces"
        if cand.is_dir():
            faces_dir = cand
    if faces_dir is None or not faces_dir.is_dir():
        log.warning("Faces dir not found — will fall back to crops if present")
    client = VlmClient(cache_dir=cache, vl_model=args.caption_model)

    out_pt = out / "scene_state_captioned.pt"
    results = run_captioning(
        ss,
        vocab_file=args.vocab_file,
        client=client,
        max_objects=args.max_objects,
        only_active=not args.include_inactive,
        skip_existing=not args.no_skip_existing,
        crops_dir=crops_dir,
        faces_dir=faces_dir,
        scene_state_path=args.scene_state,
        fail_fast=args.fail_fast,
        use_full_face=not args.use_crops,
        checkpoint_path=out_pt,
        checkpoint_every=args.checkpoint_every,
    )

    save_scene(out_pt, ss)
    save_caption_manifest(out / "caption_results.json", results)

    summary = caption_summary(ss)
    summary["n_jobs"] = len(results)
    summary["n_ok"] = sum(1 for r in results if r.ok)
    summary["n_api_fail"] = sum(
        1 for r in results if (not r.ok) and r.error not in {"no_face_or_crop", "face_missing_bbox"}
    )
    (out / "phase4b_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote %s  summary=%s", out_pt, summary)
    if summary["n_ok"] == 0 and summary["n_api_fail"] > 0:
        log.error("Phase 4b captioned 0 objects (%d API failures) — refusing to continue", summary["n_api_fail"])
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
