#!/usr/bin/env python3
"""Build viewer data directory for the Three.js 3D viewer.

Reads scene_state_stella.pt + Stella dense cloud + Phase 4 crops and writes:
  <output-dir>/
    metadata.json      scene-level info, object list (id/label/color/bbox)
    bg_cloud.bin       background Stella pts: N × 16 bytes (xyz f32 + rgb u8 + pad u8)
    objects.json       per-active-object detail with embedded pts_b64
    crops/             symlink (or copy) of Phase 4 crops

Usage
-----
    conda activate farm-phase2
    cd /home/kodifly/Desktop/farm-git/repo

    # Auto-resolve latest run
    python 3d-viewer/build_viewer_data.py

    # Explicit paths
    python 3d-viewer/build_viewer_data.py \\
        --stella-state  outputs/runs/run_XXXX/phase3.5/scene_state_stella.pt \\
        --db-path       outputs/runs/run_XXXX/phase1/out.db \\
        --crops-dir     outputs/runs/run_XXXX/phase4/crops \\
        --output-dir    outputs/runs/run_XXXX/validation/3d-viewer \\
        --voxel-size    0.05
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import struct
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_PHASE35 = _REPO / "phase3.5-stella-geometry"
for _p in [str(_PHASE35), str(_REPO / "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from load_stella_cloud import load_dense_cloud, voxel_downsample


# ---------------------------------------------------------------------------
# Color palette (golden-ratio HSV — same as export_labeled_cloud.py)
# ---------------------------------------------------------------------------

def _object_colors(n: int) -> np.ndarray:
    import colorsys
    colors = np.empty((n, 3), dtype=np.uint8)
    h = 0.0
    golden = 0.618033988749895
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(h % 1.0, 0.85, 0.95)
        colors[i] = [int(r * 255), int(g * 255), int(b * 255)]
        h += golden
    return colors


def _color_hex(rgb: np.ndarray) -> str:
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# Binary bg_cloud.bin format:
#   header: 8 bytes  "BGCLOUD\n"
#   per point: 16 bytes  [x f32][y f32][z f32][r u8][g u8][b u8][pad u8]
# ---------------------------------------------------------------------------

_BG_MAGIC = b"BGCLOUD\n"  # 8 bytes


def _write_bg_cloud(path: Path, pts: np.ndarray, colors: np.ndarray) -> None:
    """Write background cloud as packed binary."""
    n = pts.shape[0]
    # Build structured array: xyz float32 + rgb uint8 + pad
    dt = np.dtype([
        ("x", np.float32), ("y", np.float32), ("z", np.float32),
        ("r", np.uint8), ("g", np.uint8), ("b", np.uint8), ("pad", np.uint8),
    ])
    arr = np.empty(n, dtype=dt)
    arr["x"] = pts[:, 0]
    arr["y"] = pts[:, 1]
    arr["z"] = pts[:, 2]
    arr["r"] = colors[:, 0]
    arr["g"] = colors[:, 1]
    arr["b"] = colors[:, 2]
    arr["pad"] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(_BG_MAGIC)
        f.write(struct.pack("<I", n))  # 4 bytes: count
        f.write(b"\x00" * 4)           # 4 bytes: reserved
        f.write(arr.tobytes())
    mb = path.stat().st_size / 1e6
    print(f"  [bg_cloud] {n:,} pts → {path.name} ({mb:.1f} MB)")


def _pts_to_b64(pts: np.ndarray) -> str:
    """Encode (N,3) float32 pts as base64 for embedding in JSON."""
    raw = pts.astype(np.float32).tobytes()
    return base64.b64encode(raw).decode("ascii")


# ---------------------------------------------------------------------------
# Background masking: remove points inside any active object's 3σ AABB
# ---------------------------------------------------------------------------

def _build_background(
    cloud_pts: np.ndarray,
    means: np.ndarray,
    cov6: Optional[np.ndarray],
    *,
    sigma_mult: float = 2.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (bg_pts, bg_colors) — cloud pts NOT inside any object's AABB."""
    n = cloud_pts.shape[0]
    keep = np.ones(n, dtype=bool)

    if cov6 is not None:
        sigma3 = sigma_mult * np.sqrt(np.maximum(
            np.stack([cov6[:, 0], cov6[:, 3], cov6[:, 5]], axis=1), 1e-6
        ))  # (K, 3)
        lo = means - sigma3  # (K, 3)
        hi = means + sigma3  # (K, 3)

        # Vectorised: mark all pts inside any AABB as foreground
        # Process in chunks to avoid OOM with large clouds
        chunk = 100_000
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            p = cloud_pts[start:end, np.newaxis, :]   # (C, 1, 3)
            in_box = np.all((p >= lo[np.newaxis]) & (p <= hi[np.newaxis]), axis=2)  # (C, K)
            keep[start:end] = ~np.any(in_box, axis=1)

    grey = np.full((keep.sum(), 3), 60, dtype=np.uint8)
    return cloud_pts[keep], grey


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _load_vocab(path: Optional[Path]) -> List[str]:
    if path and path.is_file():
        return [l.strip() for l in path.read_text().splitlines()
                if l.strip() and not l.lstrip().startswith("#")]
    return []


def _auto_run_dir() -> Path:
    runs_dir = _REPO / "outputs" / "runs"
    if runs_dir.is_dir():
        runs = sorted(p for p in runs_dir.glob("run_*/") if p.is_dir())
        if runs:
            return runs[-1]
    return _REPO / "outputs"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Three.js viewer data from pipeline outputs")
    p.add_argument("--stella-state", type=Path, default=None)
    p.add_argument("--phase3-state", type=Path, default=None, help="Fallback if no stella state")
    p.add_argument("--db-path", type=Path, default=None, help="Stella out.db")
    p.add_argument("--crops-dir", type=Path, default=None, help="Phase 4 crops/ directory")
    p.add_argument("--vocab-file", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write viewer data (default: latest run validation/3d-viewer)")
    p.add_argument("--voxel-size", type=float, default=0.05,
                   help="Voxel downsample size for background cloud (0 = no downsampling)")
    p.add_argument("--max-bg-pts", type=int, default=1_500_000,
                   help="Hard cap on background cloud points for the viewer")
    p.add_argument("--no-bg-mask", action="store_true",
                   help="Skip background masking (keep all cloud pts grey)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    run_dir = _auto_run_dir()

    # Auto-resolve inputs
    if args.stella_state is None:
        for cand in [
            run_dir / "phase3.5" / "scene_state_stella.pt",
            run_dir / "phase4" / "scene_state_with_crops.pt",
            run_dir / "phase3" / "scene_state.pt",
        ]:
            if cand.is_file():
                args.stella_state = cand
                print(f"[build] Auto-selected scene state: {cand}")
                break

    if args.stella_state is None or not args.stella_state.is_file():
        print(f"[build] ERROR: no scene state found")
        return 2

    if args.db_path is None:
        args.db_path = run_dir / "phase1" / "out.db"

    if args.crops_dir is None:
        for cand in [
            run_dir / "phase4" / "crops",
            _REPO.parent / "pipeline" / "phase4-caption-best-view" / "output" / "crops",
        ]:
            if cand.is_dir():
                args.crops_dir = cand
                print(f"[build] Auto-selected crops dir: {cand}")
                break

    if args.vocab_file is None:
        for cand in [
            _REPO / "vocab" / "construction_vocab.txt",
            _REPO / "phase2-detect-segment-embed" / "vocab" / "construction_vocab.txt",
        ]:
            if cand.is_file():
                args.vocab_file = cand
                break

    if args.output_dir is None:
        args.output_dir = run_dir / "validation" / "3d-viewer"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load scene state
    # ------------------------------------------------------------------
    print(f"[build] Loading scene state: {args.stella_state}")
    ss = torch.load(args.stella_state, map_location="cpu", weights_only=False)

    vocab = _load_vocab(args.vocab_file)
    print(f"[build] Vocab: {len(vocab)} classes")

    means_t = ss["means"]           # (N, 3)
    cov6_t  = ss.get("cov6")
    active_t = ss.get("active")
    class_ids_t = ss.get("class_ids")
    object_stella_pts: list = ss.get("object_stella_pts") or []
    vote_mass: list = ss.get("class_vote_mass") or []

    n_total = int(means_t.shape[0])
    act_mask = (
        active_t.numpy().astype(bool)
        if isinstance(active_t, torch.Tensor)
        else np.ones(n_total, dtype=bool)
    )
    active_indices = np.where(act_mask)[0]
    n_active = len(active_indices)
    print(f"[build] Objects: total={n_total}  active={n_active}")

    means_np  = means_t.numpy().astype(np.float32)
    cov6_np   = cov6_t.numpy().astype(np.float32) if isinstance(cov6_t, torch.Tensor) else None
    cids_np   = class_ids_t.numpy() if isinstance(class_ids_t, torch.Tensor) else None

    means_active = means_np[act_mask]
    cov6_active  = cov6_np[act_mask] if cov6_np is not None else None

    # Assign colors
    colors_active = _object_colors(n_active)

    # Best-view crop paths from scene state
    crop_paths: list = ss.get("best_view_crop_path") or [""] * n_total
    while len(crop_paths) < n_total:
        crop_paths.append("")

    # ------------------------------------------------------------------
    # Load Stella dense cloud for background
    # ------------------------------------------------------------------
    cloud_pts = np.zeros((0, 3), dtype=np.float32)
    if args.db_path.is_file():
        print(f"[build] Loading Stella dense cloud from {args.db_path} …")
        raw_pts, _ = load_dense_cloud(args.db_path)
        if args.voxel_size > 0:
            cloud_pts, _ = voxel_downsample(raw_pts, None, voxel_size=args.voxel_size)
            del raw_pts
        else:
            cloud_pts = raw_pts
        print(f"[build] Dense cloud: {cloud_pts.shape[0]:,} pts (voxel_size={args.voxel_size})")
    else:
        print(f"[build] WARN: out.db not found — no background cloud")

    # Build background (subtract object regions)
    if cloud_pts.shape[0] > 0:
        if args.no_bg_mask:
            bg_pts = cloud_pts
            bg_col = np.full((bg_pts.shape[0], 3), 60, dtype=np.uint8)
        else:
            print("[build] Masking object regions from background …")
            bg_pts, bg_col = _build_background(cloud_pts, means_active, cov6_active)
            del cloud_pts
            print(f"[build] Background pts after masking: {bg_pts.shape[0]:,}")

        # Hard cap
        if bg_pts.shape[0] > args.max_bg_pts:
            rng = np.random.default_rng(42)
            idx = rng.choice(bg_pts.shape[0], size=args.max_bg_pts, replace=False)
            bg_pts = bg_pts[idx]
            bg_col = bg_col[idx]
            print(f"[build] Capped to {args.max_bg_pts:,} background pts")

        _write_bg_cloud(args.output_dir / "bg_cloud.bin", bg_pts, bg_col)
        n_bg = int(bg_pts.shape[0])
        scene_bbox_lo = bg_pts.min(axis=0).tolist() if bg_pts.shape[0] else [0, 0, 0]
        scene_bbox_hi = bg_pts.max(axis=0).tolist() if bg_pts.shape[0] else [1, 1, 1]
        del bg_pts, bg_col
    else:
        n_bg = 0
        scene_bbox_lo = means_active.min(axis=0).tolist() if n_active else [0, 0, 0]
        scene_bbox_hi = means_active.max(axis=0).tolist() if n_active else [1, 1, 1]

    # ------------------------------------------------------------------
    # Build per-object data
    # ------------------------------------------------------------------
    print(f"[build] Building per-object data for {n_active} objects …")
    objects_list = []
    n_with_pts = 0

    for ki, ai in enumerate(active_indices):
        cid = int(cids_np[ai]) if cids_np is not None else -1
        label = vocab[cid] if 0 <= cid < len(vocab) else f"obj{ai}"
        color = colors_active[ki]
        color_hex = _color_hex(color)
        mean = means_np[ai].tolist()

        # Bounding box from cov6 (3σ)
        if cov6_np is not None:
            sig3 = 3.0 * np.sqrt(np.maximum(
                [cov6_np[ai, 0], cov6_np[ai, 3], cov6_np[ai, 5]], 1e-6
            ))
            bbox_min = (means_np[ai] - sig3).tolist()
            bbox_max = (means_np[ai] + sig3).tolist()
        else:
            bbox_min = (means_np[ai] - 0.5).tolist()
            bbox_max = (means_np[ai] + 0.5).tolist()

        # Stella pts
        pts_b64 = ""
        n_pts = 0
        if ai < len(object_stella_pts) and object_stella_pts[ai] is not None:
            pts = object_stella_pts[ai]
            if isinstance(pts, torch.Tensor):
                pts = pts.numpy()
            pts = np.asarray(pts, dtype=np.float32)
            if pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] > 0:
                pts_b64 = _pts_to_b64(pts)
                n_pts = pts.shape[0]
                n_with_pts += 1
                # Refine bbox from actual pts
                bbox_min = pts.min(axis=0).tolist()
                bbox_max = pts.max(axis=0).tolist()

        # Crop path — resolve to relative path from output_dir
        crop_rel = ""
        if ai < len(crop_paths) and crop_paths[ai]:
            cp = Path(str(crop_paths[ai]))
            if cp.is_file():
                crop_rel = f"crops/{cp.name}"
            elif args.crops_dir and (args.crops_dir / cp.name).is_file():
                crop_rel = f"crops/{cp.name}"

        # Also scan crops_dir by index / object_id
        if not crop_rel and args.crops_dir and args.crops_dir.is_dir():
            for cand_name in [
                f"obj_{ai:06d}_o{ai:04d}.jpg",
                f"obj_{ai:06d}_o{int(ai):04d}.jpg",
            ]:
                if (args.crops_dir / cand_name).is_file():
                    crop_rel = f"crops/{cand_name}"
                    break
            if not crop_rel:
                # Find any crop for this object index
                for p in sorted(args.crops_dir.glob(f"obj_{ai:06d}_*.jpg")):
                    crop_rel = f"crops/{p.name}"
                    break

        # Vote summary (top-3)
        vote_summary = {}
        if vote_mass and ai < len(vote_mass) and isinstance(vote_mass[ai], dict):
            ranked = sorted(vote_mass[ai].items(), key=lambda kv: float(kv[1]), reverse=True)[:3]
            vote_summary = {
                (vocab[int(c)] if 0 <= int(c) < len(vocab) else f"cls{c}"): round(float(w), 3)
                for c, w in ranked
            }

        objects_list.append({
            "id": int(ai),
            "index": ki,
            "label": label,
            "class_id": cid,
            "color": color_hex,
            "mean": [round(v, 4) for v in mean],
            "bbox_min": [round(v, 4) for v in bbox_min],
            "bbox_max": [round(v, 4) for v in bbox_max],
            "n_pts": n_pts,
            "pts_b64": pts_b64,
            "crop_rel": crop_rel,
            "vote_summary": vote_summary,
        })

    print(f"[build] Objects with Stella pts: {n_with_pts}/{n_active}")

    # ------------------------------------------------------------------
    # Copy / symlink crops dir
    # ------------------------------------------------------------------
    crops_out_dir = args.output_dir / "crops"
    if args.crops_dir and args.crops_dir.is_dir():
        if crops_out_dir.is_symlink():
            crops_out_dir.unlink()
        if crops_out_dir.is_dir():
            shutil.rmtree(crops_out_dir)
        try:
            os.symlink(args.crops_dir.resolve(), crops_out_dir)
            print(f"[build] Crops symlink: {crops_out_dir} → {args.crops_dir.resolve()}")
        except OSError:
            shutil.copytree(str(args.crops_dir), str(crops_out_dir))
            print(f"[build] Crops copied to {crops_out_dir}")

    # ------------------------------------------------------------------
    # Write metadata.json
    # ------------------------------------------------------------------
    meta = {
        "n_objects_total": n_total,
        "n_objects_active": n_active,
        "n_objects_with_pts": n_with_pts,
        "n_bg_pts": n_bg,
        "scene_bbox_lo": [round(v, 3) for v in scene_bbox_lo],
        "scene_bbox_hi": [round(v, 3) for v in scene_bbox_hi],
        "voxel_size": args.voxel_size,
        "stella_state": str(args.stella_state),
        "vocab_size": len(vocab),
        "has_bg_cloud": n_bg > 0,
        "bg_cloud_file": "bg_cloud.bin" if n_bg > 0 else None,
        "objects_file": "objects.json",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"[build] metadata.json written")

    # ------------------------------------------------------------------
    # Write objects.json  (strip pts_b64 from metadata, keep in objects)
    # ------------------------------------------------------------------
    (args.output_dir / "objects.json").write_text(
        json.dumps(objects_list, separators=(",", ":")), encoding="utf-8"
    )
    sz_mb = (args.output_dir / "objects.json").stat().st_size / 1e6
    print(f"[build] objects.json written ({sz_mb:.1f} MB)")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print(f"\n[build] All viewer data → {args.output_dir}/")
    print(f"  metadata.json   scene info")
    print(f"  objects.json    {n_active} objects + embedded pts + crop paths ({sz_mb:.1f} MB)")
    if n_bg:
        bg_mb = (args.output_dir / "bg_cloud.bin").stat().st_size / 1e6
        print(f"  bg_cloud.bin    {n_bg:,} background pts ({bg_mb:.1f} MB)")
    print(f"\nTo launch the viewer:")
    print(f"  python 3d-viewer/serve.py --data-dir {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
