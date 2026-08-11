#!/usr/bin/env python3
"""Phase 3 — Associate / Fuse / Map

Main entry point. Reads Phase 2 detection packs in keyframe order, runs the
filter → associate → update pipeline, and writes `scene_state.pt`.

Usage
-----
conda run -n farm-phase2 python run_phase3.py \\
    --det-dir ../phase2-detect-segment-embed/output \\
    --output-dir ./output \\
    --device cuda

Optional arguments control thresholds and checkpointing interval.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

import torch

# ---------------------------------------------------------------------------
# FARM path setup — must happen before FARM imports elsewhere
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_COMMON = _HERE.parent.parent / "common"
if _COMMON.is_dir() and str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))
try:
    from paths import ensure_sys_path
    ensure_sys_path(_HERE)
except ImportError:
    # fallback: repo/farm_src/src
    import sys as _sys
    _cand = _HERE.parent.parent / "farm_src" / "src"
    if _cand.is_dir() and str(_cand) not in _sys.path:
        _sys.path.insert(0, str(_cand))

from scene_graph.map_update.models import initialize_scene_graph_state  # noqa: E402

from associate import find_neighbors, resolve  # noqa: E402
from filter import filter_pack                  # noqa: E402
from update import fuse_detections              # noqa: E402

try:
    from pipeline_logger import (  # noqa: E402
        log_elapsed,
        log_stage_banner,
        setup_logger,
        tqdm_kwargs,
    )
    log = setup_logger("phase3", stage="phase3")
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("phase3")

    def log_stage_banner(logger, title: str) -> None:
        logger.info("=" * 40)
        logger.info(title)

    def log_elapsed(logger, t0: float, label: str) -> float:
        el = time.time() - t0
        logger.info("%s finished in %.1fs", label, el)
        return el

    def tqdm_kwargs(desc: str = "", **extra):
        return {"desc": desc, **extra}

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **_kw):  # type: ignore
        return x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sorted_pack_paths(det_dir: Path) -> List[Path]:
    paths = sorted(det_dir.glob("detections_kf*.pt"))
    if not paths:
        raise FileNotFoundError(
            f"No detections_kf*.pt files found in {det_dir}"
        )
    return paths


def _count_objects(scene_state: dict) -> int:
    means = scene_state.get("means")
    return int(means.shape[0]) if isinstance(means, torch.Tensor) else 0


def _count_active(scene_state: dict) -> int:
    active = scene_state.get("active")
    if not isinstance(active, torch.Tensor):
        return 0
    return int(active.sum().item())


def _checkpoint(scene_state: dict, output_dir: Path, kf_index: int) -> None:
    path = output_dir / f"scene_state_ckpt_kf{kf_index:06d}.pt"
    torch.save(scene_state, path)
    log.info("Checkpoint saved → %s", path.name)


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

def run_phase3(
    det_dir: Path,
    output_dir: Path,
    device: str = "cuda",
    feature_dim: int = 384,
    feature_sim_thresh: float = 0.5,
    hellinger_thresh: float = 0.8,
    max_merge_distance_m: float = 1.0,
    checkpoint_every: int = 50,
    min_num_pixels: int = 50,
    min_distance_m: float = 0.3,
    max_distance_m: float = 80.0,
    label_min_score: float = 0.25,
    label_margin_ratio: float = 1.15,
    label_use_pixel_weight: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_stage_banner(log, "Phase 3 — Associate / Fuse / Map")

    pack_paths = _sorted_pack_paths(det_dir)
    log.info("Found %d keyframe packs in %s", len(pack_paths), det_dir)
    log.info(
        "Label voting: min_score=%.2f margin=%.2f pixel_weight=%s",
        label_min_score, label_margin_ratio, label_use_pixel_weight,
    )

    scene_state = initialize_scene_graph_state(
        feature_dim=feature_dim,
        device=device,
    )

    stats: List[dict] = []
    t_start = time.time()

    for kf_index, pack_path in enumerate(tqdm(pack_paths, **tqdm_kwargs(desc="Phase 3 fuse"))):
        t_kf = time.time()

        # --- load ---
        pack = torch.load(pack_path, map_location=device, weights_only=False)

        n_raw = int(pack["means"].shape[0]) if pack.get("means") is not None else 0
        if n_raw == 0:
            stats.append({"kf": kf_index, "n_raw": 0, "n_filtered": 0,
                           "n_new": 0, "n_merged": 0, "n_objects": 0})
            continue

        # --- filter ---
        pack = filter_pack(
            pack,
            device=device,
            min_num_pixels=min_num_pixels,
            min_distance_m=min_distance_m,
            max_distance_m=max_distance_m,
        )
        n_filtered = int(pack["means"].shape[0]) if pack.get("means") is not None and pack["means"].numel() > 0 else 0

        if n_filtered == 0:
            stats.append({"kf": kf_index, "n_raw": n_raw, "n_filtered": 0,
                           "n_new": 0, "n_merged": 0,
                           "n_objects": _count_objects(scene_state)})
            continue

        # --- find neighbors ---
        neighbors, _k_neighbors = find_neighbors(
            pack,
            scene_state,
            feature_sim_thresh=feature_sim_thresh,
            hellinger_thresh=hellinger_thresh,
        )

        # --- resolve correspondence ---
        det_idx, obj_idx, detection_image_ids = resolve(
            pack, neighbors, scene_state, kf_index
        )

        # --- fuse into scene state ---
        update_info = fuse_detections(
            pack,
            det_idx,
            obj_idx,
            scene_state,
            detection_image_ids,
            max_merge_distance_m=max_merge_distance_m,
            label_min_score=label_min_score,
            label_margin_ratio=label_margin_ratio,
            label_use_pixel_weight=label_use_pixel_weight,
        )

        n_new = len(update_info.get("new_object_indices") or [])
        n_merged = n_filtered - n_new
        n_obj = _count_objects(scene_state)
        n_active = _count_active(scene_state)

        stats.append({
            "kf": kf_index,
            "n_raw": n_raw,
            "n_filtered": n_filtered,
            "n_new": n_new,
            "n_merged": n_merged,
            "n_objects": n_obj,
            "n_active": n_active,
            "label_votes": int(update_info.get("label_vote_n_votes_applied") or 0),
            "label_changes": int(update_info.get("label_vote_n_labels_changed") or 0),
        })

        elapsed_kf = time.time() - t_kf
        if kf_index % 10 == 0:
            log.info(
                "kf %04d/%04d | raw=%d filt=%d new=%d merged=%d "
                "| total_obj=%d active=%d | votes=%d flips=%d | %.2fs",
                kf_index, len(pack_paths) - 1,
                n_raw, n_filtered, n_new, n_merged,
                n_obj, n_active,
                int(update_info.get("label_vote_n_votes_applied") or 0),
                int(update_info.get("label_vote_n_labels_changed") or 0),
                elapsed_kf,
            )

        # --- checkpoint ---
        if checkpoint_every > 0 and (kf_index + 1) % checkpoint_every == 0:
            _checkpoint(scene_state, output_dir, kf_index)

    # --- final save ---
    out_path = output_dir / "scene_state.pt"
    torch.save(scene_state, out_path)
    total_time = time.time() - t_start
    n_final = _count_objects(scene_state)
    n_active_final = _count_active(scene_state)
    log.info(
        "Done. %d keyframes | %d total objects | %d active | %.1fs total",
        len(pack_paths), n_final, n_active_final, total_time,
    )
    log_elapsed(log, t_start, "Phase 3")

    # --- save per-kf stats ---
    stats_path = output_dir / "run_stats.json"
    with open(stats_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    log.info("Per-keyframe stats → %s", stats_path)

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 — Associate / Fuse / Map")
    p.add_argument(
        "--det-dir",
        type=Path,
        default=Path(__file__).parent.parent
        / "phase2-detect-segment-embed"
        / "output",
        help="Directory containing Phase 2 detections_kf*.pt files",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "output",
        help="Output directory for scene_state.pt and checkpoints",
    )
    p.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    p.add_argument("--feature-dim", type=int, default=384,
                   help="DINOv3 feature dimension (default: 384)")
    p.add_argument("--feature-sim-thresh", type=float, default=0.5,
                   help="Cosine similarity threshold for neighbour matching")
    p.add_argument("--hellinger-thresh", type=float, default=0.8,
                   help="Hellinger distance threshold for Gaussian matching")
    p.add_argument("--max-merge-dist", type=float, default=1.0,
                   help="Maximum world-space distance (m) to allow a merge")
    p.add_argument("--checkpoint-every", type=int, default=50,
                   help="Save an intermediate checkpoint every N keyframes (0=off)")
    p.add_argument("--min-pixels", type=int, default=50,
                   help="Minimum mask pixels to keep a detection")
    p.add_argument("--min-dist", type=float, default=0.3,
                   help="Minimum detection distance from camera (m)")
    p.add_argument("--max-dist", type=float, default=80.0,
                   help="Maximum detection distance from camera (m)")
    p.add_argument("--label-min-score", type=float, default=0.25,
                   help="Min detection score to contribute a class vote")
    p.add_argument("--label-margin", type=float, default=1.15,
                   help="Top vote mass must exceed runner-up * margin to flip label")
    p.add_argument("--label-no-pixel-weight", action="store_true",
                   help="Vote with score only (disable sqrt(num_pixels) weights)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = run_phase3(
        det_dir=args.det_dir,
        output_dir=args.output_dir,
        device=args.device,
        feature_dim=args.feature_dim,
        feature_sim_thresh=args.feature_sim_thresh,
        hellinger_thresh=args.hellinger_thresh,
        max_merge_distance_m=args.max_merge_dist,
        checkpoint_every=args.checkpoint_every,
        min_num_pixels=args.min_pixels,
        min_distance_m=args.min_dist,
        max_distance_m=args.max_dist,
        label_min_score=args.label_min_score,
        label_margin_ratio=args.label_margin,
        label_use_pixel_weight=not args.label_no_pixel_weight,
    )
    print(f"scene_state.pt written → {out}")
