#!/usr/bin/env python3
"""Phase 3.5 — Stella PCD geometry refinement.

Replaces inflated per-frame Gaussian geometry in ``scene_state.pt`` with
tight bounding geometry derived from Stella's dense point cloud.

Steps
-----
1. Load Stella dense points from ``out.db`` (dense_points table).
2. Voxel-downsample to ~1–3 M points (default 5 cm grid).
3. For each active fused object: project downsampled cloud into each
   observed face; apply YOLOE mask + DA3 depth occlusion filter; accumulate
   Stella inlier points.
4. Replace object ``means`` / ``cov6`` with median + empirical cov (clamped).
5. Write ``outputs/phase3.5/scene_state_stella.pt`` + summary JSON.

Usage (host / conda)
-----
    conda activate farm-phase2
    cd repo/
    python phase3.5-stella-geometry/run_phase35.py \\
        --phase1-dir  outputs/phase1 \\
        --phase15-dir outputs/phase1.5 \\
        --det-dir     outputs/phase2 \\
        --scene-state outputs/phase3/scene_state.pt \\
        --output-dir  outputs/phase3.5

Usage (inside Docker farm container)
-----
    python /workspace/phase3.5-stella-geometry/run_phase35.py \\
        --phase1-dir  /phase1 \\
        --phase15-dir /phase1.5 \\
        --det-dir     /phase2 \\
        --scene-state /phase3/scene_state.pt \\
        --output-dir  /phase3.5
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
for _p in [str(_HERE), str(_HERE.parent / "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from load_stella_cloud import load_dense_cloud, voxel_downsample
from assign_to_objects import build_object_point_arrays
from rewrite_geometry import rewrite_geometry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase35")


def _sorted_pack_paths(det_dir: Path) -> List[Path]:
    paths = sorted(det_dir.glob("detections_kf*.pt"))
    if not paths:
        raise FileNotFoundError(f"No detections_kf*.pt in {det_dir}")
    return paths


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3.5 — Stella PCD geometry refinement"
    )
    p.add_argument("--phase1-dir", type=Path,
                   default=Path("outputs/phase1"),
                   help="Phase 1 output dir (contains out.db)")
    p.add_argument("--phase15-dir", type=Path,
                   default=Path("outputs/phase1.5"),
                   help="Phase 1.5 output dir (depth/ faces/ frames_json/)")
    p.add_argument("--det-dir", type=Path,
                   default=Path("outputs/phase2"),
                   help="Phase 2 detections_kf*.pt directory")
    p.add_argument("--scene-state", type=Path,
                   default=Path("outputs/phase3/scene_state.pt"),
                   help="Phase 3 fused scene_state.pt")
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/phase3.5"),
                   help="Where to write scene_state_stella.pt")
    p.add_argument("--voxel-size", type=float, default=0.05,
                   help="Voxel grid size for cloud downsampling (metres)")
    p.add_argument("--tau-abs", type=float, default=0.15,
                   help="Absolute depth occlusion tolerance (metres)")
    p.add_argument("--tau-rel", type=float, default=0.05,
                   help="Relative depth tolerance (fraction of camera Z)")
    p.add_argument("--feat-sim-min", type=float, default=0.30,
                   help="Min feature cosine to accept a detection for Stella labeling")
    p.add_argument("--max-center-dist", type=float, default=6.0,
                   help="Max centre distance (m) for detection↔object matching")
    p.add_argument("--min-pts", type=int, default=5,
                   help="Min Stella inliers required to update an object")
    p.add_argument("--max-sigma", type=float, default=1.5,
                   help="Max σ per axis in clamped empirical covariance (metres)")
    p.add_argument("--device", default="cpu",
                   help="Not used for cloud work; here for consistency")
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    for attr, label in [
        ("phase1_dir", "Phase 1 dir"),
        ("det_dir", "Phase 2 det dir"),
        ("scene_state", "scene_state.pt"),
    ]:
        path = getattr(args, attr)
        if not path.exists():
            log.error("Missing %s: %s", label, path)
            return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    db_path = args.phase1_dir / "out.db"
    if not db_path.is_file():
        log.error("Stella map DB not found: %s", db_path)
        return 2

    t_start = time.time()

    # ------------------------------------------------------------------
    # 1. Load + downsample Stella dense cloud
    # ------------------------------------------------------------------
    log.info("Loading Stella dense cloud from %s …", db_path)
    pts, colors = load_dense_cloud(db_path)
    pts_ds, colors_ds = voxel_downsample(pts, colors, voxel_size=args.voxel_size)
    del pts, colors  # free memory

    # ------------------------------------------------------------------
    # 2. Load Phase 3 scene_state
    # ------------------------------------------------------------------
    log.info("Loading scene_state: %s", args.scene_state)
    scene_state = torch.load(args.scene_state, map_location="cpu", weights_only=False)
    means = scene_state.get("means")
    n_total = int(means.shape[0]) if isinstance(means, torch.Tensor) else 0
    active = scene_state.get("active")
    n_active = int(active.sum().item()) if isinstance(active, torch.Tensor) else n_total
    log.info("Phase 3 objects: total=%d active=%d", n_total, n_active)

    # ------------------------------------------------------------------
    # 3. Load Phase 2 pack paths
    # ------------------------------------------------------------------
    pack_paths = _sorted_pack_paths(args.det_dir)
    log.info("Phase 2 packs: %d files", len(pack_paths))

    # ------------------------------------------------------------------
    # 4. Assign Stella points to objects
    # ------------------------------------------------------------------
    log.info("Assigning Stella points to fused objects …")
    t_assign = time.time()
    object_point_arrays = build_object_point_arrays(
        pts_ds,
        scene_state,
        pack_paths,
        tau_abs=args.tau_abs,
        tau_rel=args.tau_rel,
        feat_sim_min=args.feat_sim_min,
        max_center_dist_m=args.max_center_dist,
        min_points_per_object=args.min_pts,
    )
    t_assign_done = time.time()
    n_covered = sum(1 for p in object_point_arrays if p is not None)
    log.info(
        "Point assignment done in %.1fs: %d / %d active objects covered",
        t_assign_done - t_assign, n_covered, n_active,
    )

    # ------------------------------------------------------------------
    # 5. Rewrite geometry
    # ------------------------------------------------------------------
    log.info("Rewriting geometry …")
    ss_stella = rewrite_geometry(
        scene_state,
        object_point_arrays,
        min_pts=args.min_pts,
        max_sigma=args.max_sigma,
    )

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    out_pt = args.output_dir / "scene_state_stella.pt"
    torch.save(ss_stella, out_pt)
    log.info("Wrote %s", out_pt)

    # ------------------------------------------------------------------
    # 7. Summary JSON
    # ------------------------------------------------------------------
    n_pts = ss_stella.get("stella_n_pts")
    n_pts_arr = n_pts.numpy() if isinstance(n_pts, torch.Tensor) else np.zeros(n_total)
    n_updated = int((n_pts_arr > 0).sum())

    summary = {
        "n_objects_total": n_total,
        "n_objects_active": n_active,
        "n_objects_stella_updated": n_updated,
        "n_objects_unchanged": n_active - n_updated,
        "stella_coverage_pct": round(100.0 * n_updated / max(n_active, 1), 1),
        "median_pts_per_updated": int(np.median(n_pts_arr[n_pts_arr > 0])) if n_updated > 0 else 0,
        "cloud_pts_after_downsample": int(pts_ds.shape[0]),
        "voxel_size_m": args.voxel_size,
        "tau_abs_m": args.tau_abs,
        "tau_rel": args.tau_rel,
        "feat_sim_min": args.feat_sim_min,
        "max_sigma_m": args.max_sigma,
        "elapsed_s": round(time.time() - t_start, 1),
        "output_pt": str(out_pt),
    }
    summary_path = args.output_dir / "phase35_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary → %s", summary_path)
    log.info(
        "Phase 3.5 done in %.1fs  — %d / %d active objects updated with Stella geometry",
        time.time() - t_start, n_updated, n_active,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
