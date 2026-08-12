#!/usr/bin/env python3
"""Export a colored PLY and top-down PNG that show spatial object segmentation quality.

What this produces
------------------
labeled_cloud.ply
    Dense Stella cloud with every point colored:
      • unique vivid hue per object (for assigned object points)
      • mid-grey (120,120,120) for background / unassigned points
    Open in CloudCompare or MeshLab to inspect spatial separation.

topdown.png
    2D top-down view (world X-Z plane):
      • grey dots  = downsampled Stella background cloud
      • colored filled circles = object footprints (one color per object)
      • black ellipses = ±2σ extent of each object's fused Gaussian
    Good for checking: are objects at the right position? do they overlap?

object_isolation/<obj_id>.ply  (if --isolate-objects)
    One PLY per active object: just the Stella points assigned to it.
    Load several in CloudCompare at once to check spatial boundaries.

Usage
-----
    conda activate farm-phase2
    cd /home/kodifly/Desktop/farm-git/repo

    # Full export for latest run
    python phase3.5-stella-geometry/export_labeled_cloud.py

    # Specific run
    python phase3.5-stella-geometry/export_labeled_cloud.py \\
        --stella-state  outputs/runs/run_XXXX/phase3.5/scene_state_stella.pt \\
        --phase3-state  outputs/runs/run_XXXX/phase3/scene_state.pt \\
        --db-path       outputs/runs/run_XXXX/phase1/out.db \\
        --output-dir    outputs/runs/run_XXXX/validation/stella_cloud \\
        --voxel-size    0.05

    # Also write per-object PLY files
    python phase3.5-stella-geometry/export_labeled_cloud.py --isolate-objects
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
for _p in [str(_HERE), str(_HERE.parent / "common")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from load_stella_cloud import load_dense_cloud, voxel_downsample


# ---------------------------------------------------------------------------
# Color palette — maximally distinct hues (golden ratio spacing)
# ---------------------------------------------------------------------------

def _object_colors(n: int) -> np.ndarray:
    """Return (n, 3) uint8 vivid colors, maximally spread in hue."""
    import colorsys
    colors = np.empty((n, 3), dtype=np.uint8)
    golden = 0.618033988749895
    h = 0.0
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(h % 1.0, 0.85, 0.95)
        colors[i] = [int(r * 255), int(g * 255), int(b * 255)]
        h += golden
    return colors


_GREY = np.array([110, 110, 110], dtype=np.uint8)
_BG   = np.array([60,  60,  60],  dtype=np.uint8)


# ---------------------------------------------------------------------------
# PLY writer (ASCII — readable by CloudCompare / MeshLab / Viser)
# ---------------------------------------------------------------------------

def _write_ply(path: Path, pts: np.ndarray, colors: np.ndarray) -> None:
    """Write (N,3) float32 pts + (N,3) uint8 colors as ASCII PLY."""
    n = pts.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = pts[i]
            r, g, b = colors[i]
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")


def _write_ply_binary(path: Path, pts: np.ndarray, colors: np.ndarray) -> None:
    """Write binary PLY — much faster and smaller than ASCII."""
    n = pts.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    dt = np.dtype([
        ("x", np.float32), ("y", np.float32), ("z", np.float32),
        ("red", np.uint8), ("green", np.uint8), ("blue", np.uint8),
    ])
    data = np.empty(n, dtype=dt)
    data["x"] = pts[:, 0]
    data["y"] = pts[:, 1]
    data["z"] = pts[:, 2]
    data["red"]   = colors[:, 0]
    data["green"] = colors[:, 1]
    data["blue"]  = colors[:, 2]
    header = (
        b"ply\nformat binary_little_endian 1.0\n"
        + f"element vertex {n}\n".encode()
        + b"property float x\nproperty float y\nproperty float z\n"
        b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        b"end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header)
        f.write(data.tobytes())


# ---------------------------------------------------------------------------
# Top-down 2D map
# ---------------------------------------------------------------------------

def _topdown_png(
    out_path: Path,
    bg_pts: np.ndarray,           # (N,3) world XYZ — background cloud
    obj_pts_list: List[Optional[np.ndarray]],  # per active object
    means_active: np.ndarray,     # (K,3) active object means
    cov6_active: Optional[np.ndarray],         # (K,6) or None
    labels: Optional[List[str]],  # per active object
    colors_active: np.ndarray,    # (K,3) uint8
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[export] matplotlib not available — skipping topdown PNG")
        return

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect("equal")
    ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#111111")

    # Background cloud (X-Z plane, Y is up)
    if bg_pts.shape[0] > 0:
        # Subsample background for speed
        rng = np.random.default_rng(0)
        max_bg = 200_000
        idx = rng.choice(bg_pts.shape[0], size=min(max_bg, bg_pts.shape[0]), replace=False)
        bx, bz = bg_pts[idx, 0], bg_pts[idx, 2]
        ax.scatter(bx, bz, s=0.3, c="#444444", linewidths=0, alpha=0.5, rasterized=True)

    # Object Stella points
    for pts, col in zip(obj_pts_list, colors_active):
        if pts is None or pts.shape[0] == 0:
            continue
        c = tuple(int(v) / 255.0 for v in col)
        ax.scatter(pts[:, 0], pts[:, 2], s=2.0, c=[c], linewidths=0,
                   alpha=0.8, rasterized=True)

    # Object means + 2σ ellipses
    for ki in range(len(means_active)):
        mx, mz = float(means_active[ki, 0]), float(means_active[ki, 2])
        col = tuple(int(v) / 255.0 for v in colors_active[ki])
        ax.plot(mx, mz, "o", color=col, markersize=4, markeredgecolor="white",
                markeredgewidth=0.5, zorder=5)
        # 2σ ellipse in XZ plane
        if cov6_active is not None:
            sx = float(np.sqrt(max(float(cov6_active[ki, 0]), 1e-6)))
            sz = float(np.sqrt(max(float(cov6_active[ki, 5]), 1e-6)))
            from matplotlib.patches import Ellipse
            ell = Ellipse(
                (mx, mz), width=4 * sx, height=4 * sz,
                edgecolor=col, facecolor="none", linewidth=0.8, alpha=0.6, zorder=4,
            )
            ax.add_patch(ell)
        # Label (class name only, small font)
        if labels and ki < len(labels):
            ax.text(mx, mz, labels[ki], fontsize=4, color="white",
                    ha="center", va="bottom", alpha=0.8, zorder=6)

    ax.set_title(
        f"Top-down object map  ({len(means_active)} active objects, XZ plane)\n"
        "Grey = Stella background  ·  Colour = object Stella pts  ·  Circles = ±2σ Gaussian",
        color="white", fontsize=10,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")
    ax.set_xlabel("X (m)", color="white")
    ax.set_ylabel("Z (m)", color="white")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[export] Top-down map → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _auto_run_dir() -> Path:
    """Find latest run under outputs/runs/."""
    runs = sorted(
        (p for p in (Path("outputs/runs")).glob("run_*/") if p.is_dir()),
        key=lambda p: p.name,
    )
    return runs[-1] if runs else Path("outputs")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export labeled Stella cloud + top-down spatial validation map"
    )
    p.add_argument("--stella-state", type=Path, default=None,
                   help="scene_state_stella.pt (defaults to latest run)")
    p.add_argument("--phase3-state", type=Path, default=None,
                   help="original scene_state.pt for Gaussian comparison (optional)")
    p.add_argument("--db-path", type=Path, default=None,
                   help="Stella out.db for loading background dense cloud")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where to write PLY + PNG (defaults to validation/stella_cloud/ of latest run)")
    p.add_argument("--voxel-size", type=float, default=0.05,
                   help="Voxel size for background cloud downsampling (m)")
    p.add_argument("--isolate-objects", action="store_true",
                   help="Also write per-object PLY files for detailed inspection")
    p.add_argument("--max-bg-pts", type=int, default=500_000,
                   help="Max background cloud points in PLY (performance)")
    p.add_argument("--vocab-file", type=Path, default=None)
    return p.parse_args()


def _load_vocab(path: Optional[Path]) -> List[str]:
    if path and path.is_file():
        return [l.strip() for l in path.read_text().splitlines()
                if l.strip() and not l.lstrip().startswith("#")]
    return []


def main() -> int:
    args = _parse_args()

    # Auto-resolve paths from latest run
    run_dir = _auto_run_dir()

    if args.stella_state is None:
        args.stella_state = run_dir / "phase3.5" / "scene_state_stella.pt"
    if not args.stella_state.is_file():
        # Fall back to Phase 3 scene state without stella pts
        fallback = run_dir / "phase3" / "scene_state.pt"
        if fallback.is_file():
            print(f"[export] No stella state — using Phase 3 state: {fallback}")
            args.stella_state = fallback
        else:
            print(f"[export] ERROR: stella state not found: {args.stella_state}")
            return 2

    if args.db_path is None:
        args.db_path = run_dir / "phase1" / "out.db"

    if args.output_dir is None:
        args.output_dir = run_dir / "validation" / "stella_cloud"

    if args.vocab_file is None:
        for cand in [
            Path("vocab/construction_vocab.txt"),
            _HERE.parent / "vocab" / "construction_vocab.txt",
        ]:
            if cand.is_file():
                args.vocab_file = cand
                break

    vocab = _load_vocab(args.vocab_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load scene state
    # ------------------------------------------------------------------
    print(f"[export] Loading {args.stella_state} …")
    ss = torch.load(args.stella_state, map_location="cpu", weights_only=False)

    means_t = ss["means"]
    cov6_t  = ss.get("cov6")
    active_t = ss.get("active")
    class_ids_t = ss.get("class_ids")
    object_stella_pts: List[Optional[np.ndarray]] = ss.get("object_stella_pts") or []

    n = int(means_t.shape[0])
    act_mask = (
        active_t.numpy().astype(bool)
        if isinstance(active_t, torch.Tensor)
        else np.ones(n, dtype=bool)
    )
    active_indices = np.where(act_mask)[0]
    n_active = len(active_indices)
    print(f"[export] Objects: total={n}  active={n_active}")

    means_np = means_t.numpy().astype(np.float32)
    cov6_np  = cov6_t.numpy().astype(np.float32) if isinstance(cov6_t, torch.Tensor) else None
    class_ids_np = (
        class_ids_t.numpy() if isinstance(class_ids_t, torch.Tensor) else None
    )

    means_active = means_np[act_mask]
    cov6_active  = cov6_np[act_mask] if cov6_np is not None else None

    # Class labels for active objects
    labels = []
    for ai in active_indices:
        cid = int(class_ids_np[ai]) if class_ids_np is not None else -1
        name = vocab[cid] if 0 <= cid < len(vocab) else f"obj{ai}"
        labels.append(name)

    # Per-active-object stella pts
    active_stella_pts: List[Optional[np.ndarray]] = [
        object_stella_pts[ai] if ai < len(object_stella_pts) else None
        for ai in active_indices
    ]

    # Assign colors by active index
    colors_active = _object_colors(n_active)

    # ------------------------------------------------------------------
    # Load background Stella cloud (for PLY + topdown bg)
    # ------------------------------------------------------------------
    bg_pts: np.ndarray = np.zeros((0, 3), dtype=np.float32)
    if args.db_path.is_file():
        print(f"[export] Loading Stella dense cloud from {args.db_path} …")
        raw_pts, _ = load_dense_cloud(args.db_path)
        bg_pts, _ = voxel_downsample(raw_pts, None, voxel_size=args.voxel_size)
        del raw_pts
        print(f"[export] Background cloud: {bg_pts.shape[0]} pts after voxel downsample")
    else:
        print(f"[export] WARN: out.db not found at {args.db_path} — no background cloud")

    # ------------------------------------------------------------------
    # Assemble colored PLY
    # ------------------------------------------------------------------
    all_pts_list: List[np.ndarray] = []
    all_col_list: List[np.ndarray] = []

    # Object points (colored)
    n_obj_pts = 0
    for ki, pts in enumerate(active_stella_pts):
        if pts is None or pts.shape[0] == 0:
            continue
        c = colors_active[ki]
        col = np.tile(c, (pts.shape[0], 1))
        all_pts_list.append(pts.astype(np.float32))
        all_col_list.append(col)
        n_obj_pts += pts.shape[0]

    # Background cloud (grey), subsampled to max_bg_pts
    if bg_pts.shape[0] > 0:
        rng = np.random.default_rng(42)
        if bg_pts.shape[0] > args.max_bg_pts:
            idx = rng.choice(bg_pts.shape[0], size=args.max_bg_pts, replace=False)
            bg_sub = bg_pts[idx]
        else:
            bg_sub = bg_pts
        bg_col = np.tile(_BG, (bg_sub.shape[0], 1))
        all_pts_list.append(bg_sub)
        all_col_list.append(bg_col)
        print(f"[export] PLY: {n_obj_pts} object pts + {bg_sub.shape[0]} background pts")
    else:
        print(f"[export] PLY: {n_obj_pts} object pts (no background cloud)")

    if all_pts_list:
        all_pts = np.concatenate(all_pts_list, axis=0)
        all_col = np.concatenate(all_col_list, axis=0)
        ply_path = args.output_dir / "labeled_cloud.ply"
        print(f"[export] Writing binary PLY ({all_pts.shape[0]} pts) → {ply_path} …")
        _write_ply_binary(ply_path, all_pts, all_col)
        print(f"[export] Done. Open in CloudCompare: cloudcompare {ply_path}")
    else:
        print("[export] No points to export.")

    # ------------------------------------------------------------------
    # Per-object isolation PLYs
    # ------------------------------------------------------------------
    if args.isolate_objects:
        iso_dir = args.output_dir / "object_isolation"
        iso_dir.mkdir(parents=True, exist_ok=True)
        n_written = 0
        for ki, (ai, pts) in enumerate(zip(active_indices, active_stella_pts)):
            if pts is None or pts.shape[0] < 3:
                continue
            cid = int(class_ids_np[ai]) if class_ids_np is not None else -1
            name = vocab[cid] if 0 <= cid < len(vocab) else "unknown"
            col = np.tile(colors_active[ki], (pts.shape[0], 1))
            out = iso_dir / f"obj_{ai:06d}_{name.replace(' ', '_')}.ply"
            _write_ply_binary(out, pts.astype(np.float32), col)
            n_written += 1
        print(f"[export] Wrote {n_written} per-object PLY files → {iso_dir}")

    # ------------------------------------------------------------------
    # Top-down map
    # ------------------------------------------------------------------
    _topdown_png(
        args.output_dir / "topdown.png",
        bg_pts=bg_pts,
        obj_pts_list=active_stella_pts,
        means_active=means_active,
        cov6_active=cov6_active,
        labels=labels,
        colors_active=colors_active,
    )

    # ------------------------------------------------------------------
    # Spatial overlap report
    # ------------------------------------------------------------------
    _write_overlap_report(
        args.output_dir / "overlap_report.json",
        means_active, cov6_active, labels, active_indices,
    )

    print(f"\n[export] All outputs → {args.output_dir}/")
    print(f"  labeled_cloud.ply   — open in CloudCompare or MeshLab")
    print(f"  topdown.png         — 2D spatial layout of all objects")
    print(f"  overlap_report.json — pairs of objects with overlapping Gaussians")
    if args.isolate_objects:
        print(f"  object_isolation/   — one PLY per object")
    return 0


def _write_overlap_report(
    out_path: Path,
    means: np.ndarray,    # (K, 3)
    cov6: Optional[np.ndarray],  # (K, 6)
    labels: List[str],
    indices: np.ndarray,
) -> None:
    """Flag pairs of active objects whose 3σ boxes overlap — signals under-separation."""
    if cov6 is None:
        return
    K = means.shape[0]
    # Half-extents at 3σ
    sigma3 = 3.0 * np.sqrt(np.maximum(
        np.stack([cov6[:, 0], cov6[:, 3], cov6[:, 5]], axis=1), 1e-6
    ))
    lo = means - sigma3   # (K,3)
    hi = means + sigma3

    overlaps = []
    for i in range(K):
        for j in range(i + 1, K):
            # AABB overlap in all 3 axes
            if (lo[i] < hi[j]).all() and (lo[j] < hi[i]).all():
                dist = float(np.linalg.norm(means[i] - means[j]))
                overlaps.append({
                    "obj_a": int(indices[i]),
                    "label_a": labels[i] if i < len(labels) else "?",
                    "obj_b": int(indices[j]),
                    "label_b": labels[j] if j < len(labels) else "?",
                    "centre_dist_m": round(dist, 3),
                })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"n_overlapping_pairs": len(overlaps), "pairs": overlaps[:200]},
        indent=2
    ))
    print(f"[export] Overlap report: {len(overlaps)} overlapping object pairs → {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
