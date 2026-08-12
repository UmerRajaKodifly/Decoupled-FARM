#!/usr/bin/env python3
"""Phase 4a — Best-view crop extraction (no VLM / captions yet).

Loads Phase 3.5 ``scene_state_stella.pt`` (or Phase 3 ``scene_state.pt``)
+ Phase 2 packs, picks the best supporting detection per active object,
crops the RGB face, and writes:

  output/crops/obj_XXXXXX_oYYYY.jpg
  output/best_views.json
  output/scene_state_with_crops.pt   (enriched scene state)
  output/phase4a_summary.json

Defaults are tighter than the original pipeline/ version because Stella means
are accurate — we no longer need a 4 m centre-distance window.

Usage (host / conda)
--------------------
    conda activate farm-phase2
    cd repo/
    python phase4-caption-best-view/run_phase4_crops.py \\
        --scene-state outputs/phase3.5/scene_state_stella.pt \\
        --det-dir     outputs/phase2 \\
        --output-dir  outputs/phase4

Usage (inside Docker)
--------------------
    python /workspace/phase4-caption-best-view/run_phase4_crops.py \\
        --scene-state /phase3.5/scene_state_stella.pt \\
        --det-dir     /phase2 \\
        --output-dir  /phase4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from best_view import (
    apply_best_views_to_scene_state,
    results_to_jsonable,
    select_best_views,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase4a")


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent.parent  # repo root inside Docker

    # Prefer Stella state; fall back to Phase 3
    def _default_state() -> Path:
        s = here / "outputs" / "phase3.5" / "scene_state_stella.pt"
        if s.is_file():
            return s
        return here / "outputs" / "phase3" / "scene_state.pt"

    p = argparse.ArgumentParser(description="Phase 4a — best-view RGB crops")
    p.add_argument(
        "--scene-state",
        type=Path,
        default=_default_state(),
        help="scene_state_stella.pt (Phase 3.5) or scene_state.pt (Phase 3)",
    )
    p.add_argument(
        "--det-dir",
        type=Path,
        default=here / "outputs" / "phase2",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=here / "outputs" / "phase4",
    )
    p.add_argument("--include-inactive", action="store_true",
                   help="Also crop inactive/pruned objects")
    # Tighter defaults vs pipeline/ (Stella means are accurate now)
    p.add_argument("--feat-sim-min", type=float, default=0.50,
                   help="Min cosine similarity vs fused object feature (default 0.50, tighter than original 0.35)")
    p.add_argument("--max-center-dist", type=float, default=1.5,
                   help="Max centre distance (m) object↔detection (default 1.5 m, was 4.0)")
    p.add_argument("--pad-frac", type=float, default=0.10,
                   help="Fractional padding around detection box")
    p.add_argument("--max-objects", type=int, default=0,
                   help="Debug: only first N objects (0=all)")
    p.add_argument(
        "--write-scene-state",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-write-scene-state",
        action="store_false",
        dest="write_scene_state",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.scene_state.is_file():
        # Try relative to script location
        alt = _HERE.parent / "outputs" / "phase3.5" / "scene_state_stella.pt"
        if alt.is_file():
            args.scene_state = alt
        else:
            log.error("Missing scene state: %s", args.scene_state)
            return 2
    if not args.det_dir.is_dir():
        log.error("Missing det dir: %s", args.det_dir)
        return 2

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    crops_dir = out / "crops"

    log.info("Loading %s", args.scene_state)
    scene_state = torch.load(args.scene_state, map_location="cpu", weights_only=False)
    n = int(scene_state["means"].shape[0])
    n_act = (
        int(scene_state["active"].sum().item())
        if isinstance(scene_state.get("active"), torch.Tensor)
        else n
    )
    log.info("Objects: total=%d active=%d  (feat_sim_min=%.2f max_center_dist=%.1fm)",
             n, n_act, args.feat_sim_min, args.max_center_dist)

    t0 = time.time()
    results = select_best_views(
        scene_state,
        args.det_dir,
        crops_dir,
        only_active=not args.include_inactive,
        feat_sim_min=args.feat_sim_min,
        max_center_dist_m=args.max_center_dist,
        pad_frac=args.pad_frac,
        max_objects=args.max_objects,
    )
    elapsed = time.time() - t0

    n_ok = sum(1 for r in results if r.ok)
    n_fail = len(results) - n_ok
    log.info("Crops: ok=%d fail=%d in %.1fs", n_ok, n_fail, elapsed)

    manifest_path = out / "best_views.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(results_to_jsonable(results), fh, indent=2)
    log.info("Wrote %s", manifest_path)

    if args.write_scene_state:
        apply_best_views_to_scene_state(scene_state, results)
        out_pt = out / "scene_state_with_crops.pt"
        torch.save(scene_state, out_pt)
        log.info("Wrote enriched state → %s", out_pt)

    summary = {
        "n_objects_total": n,
        "n_objects_active": n_act,
        "n_attempted": len(results),
        "n_crops_ok": n_ok,
        "n_crops_fail": n_fail,
        "elapsed_s": round(elapsed, 2),
        "det_dir": str(args.det_dir),
        "scene_state_in": str(args.scene_state),
        "crops_dir": str(crops_dir),
        "feat_sim_min": args.feat_sim_min,
        "max_center_dist_m": args.max_center_dist,
        "crop_pct": round(100.0 * n_ok / max(n_act, 1), 1),
    }
    summary_path = out / "phase4a_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Summary → %s", summary_path)
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
