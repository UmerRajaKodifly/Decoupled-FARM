#!/usr/bin/env python3
"""Validate Phase 2 detection packs before moving to Phase 3.

What this checks
----------------
Phase 2 must produce *per-keyframe* proposals that are:
  1. Spatially sensible (3D Gaussians in world frame, metric depth)
  2. Semantically plausible (construction-ish labels, non-junk confidences)
  3. Feature-ready for Phase 3 merge (DINOv3 features L2-norm ≈ 1, finite)

It does NOT check cross-keyframe identity — that is Phase 3.

Outputs under <det-dir>/validation/
  metrics.json              aggregate numbers + pass/warn flags
  summary.txt               human-readable rollup
  class_hist.png            detections per class
  score_hist.png            YOLOE score distribution
  det_count_per_kf.png      detections vs keyframe index
  mean_depths.png           ||mean|| / Z distribution
  feature_norm_hist.png     DINOv3 L2 norms (should peak near 1.0)
  world_xyz_scatter.png     top-down XY of all means
  overlays/                 RGB face overlays with masks + labels
  overlays_3d.html          interactive 3D scatter (plotly if available)
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _as_numpy(x) -> np.ndarray:
    if x is None:
        return np.zeros((0,), dtype=np.float32)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _load_pack(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _kf_sort_key(p: Path) -> int:
    m = re.search(r"kf(\d+)", p.stem)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Metrics over one pack
# ---------------------------------------------------------------------------

def _pack_stats(pack: dict) -> dict:
    means = _as_numpy(pack.get("means")).reshape(-1, 3) if pack.get("means") is not None else np.zeros((0, 3))
    cov6 = _as_numpy(pack.get("cov6")).reshape(-1, 6) if pack.get("cov6") is not None else np.zeros((0, 6))
    feats = _as_numpy(pack.get("features"))
    scores = _as_numpy(pack.get("scores")).reshape(-1)
    class_ids = _as_numpy(pack.get("class_ids")).reshape(-1).astype(np.int64)
    labels = pack.get("labels") or pack.get("names") or []
    vocab = pack.get("vocab") or []
    batch_ids = _as_numpy(pack.get("batch_ids")).reshape(-1).astype(np.int64)

    M = means.shape[0]
    if feats is None or feats.size == 0:
        feats = np.zeros((M, 0), dtype=np.float32)
    elif feats.ndim == 1:
        feats = feats.reshape(M, -1) if M else feats.reshape(0, -1)

    # Resolve labels from class ids if labels empty
    if len(labels) != M and len(vocab) > 0 and class_ids.size == M:
        labels = [vocab[int(i)] if 0 <= int(i) < len(vocab) else f"id{int(i)}" for i in class_ids]
    elif len(labels) != M:
        labels = [f"id{int(i)}" for i in class_ids] if class_ids.size == M else [f"det{i}" for i in range(M)]

    feat_norms = np.linalg.norm(feats, axis=1) if feats.shape[-1] > 0 and M else np.zeros((M,))
    # Eigen-ish size of Gaussian: sqrt of diag of cov (xx,yy,zz indices 0,3,5)
    if M and cov6.shape[0] == M:
        extents = np.sqrt(np.clip(cov6[:, [0, 3, 5]], 1e-12, None))  # (M,3) std-like
        max_extent = extents.max(axis=1)
    else:
        max_extent = np.zeros((M,))

    finite_means = np.isfinite(means).all(axis=1) if M else np.array([], dtype=bool)
    radii = np.linalg.norm(means, axis=1) if M else np.zeros((0,))

    return {
        "M": M,
        "means": means,
        "scores": scores,
        "labels": list(labels),
        "class_ids": class_ids,
        "batch_ids": batch_ids,
        "feat_norms": feat_norms,
        "max_extent": max_extent,
        "radii": radii,
        "n_nonfinite_means": int((~finite_means).sum()) if M else 0,
        "n_zero_extent": int((max_extent < 1e-4).sum()) if M else 0,
        "n_feat_bad_norm": int(((feat_norms < 0.5) | (feat_norms > 1.5)).sum()) if M and feats.shape[-1] else 0,
        "face_meta": pack.get("face_meta") or [],
        "kf_id": pack.get("kf_id", path_stem_safe(pack)),
        "masks": pack.get("masks"),
        "vocab": vocab,
    }


def path_stem_safe(pack: dict) -> str:
    return str(pack.get("kf_id") or "unknown")


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

_COLORS = [
    (255, 64, 64), (64, 180, 255), (80, 220, 120), (255, 180, 40),
    (200, 100, 255), (255, 100, 180), (40, 220, 220), (180, 220, 40),
]


def _draw_overlays_for_pack(stats: dict, out_dir: Path, max_faces: int = 4) -> List[Path]:
    """Draw mask + label overlays for each face that has RGB on disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    face_meta = stats["face_meta"]
    masks = stats["masks"]
    batch_ids = stats["batch_ids"]
    labels = stats["labels"]
    scores = stats["scores"]
    means = stats["means"]

    if not face_meta:
        return saved

    # Group detections by face index
    by_face: Dict[int, List[int]] = defaultdict(list)
    for di, b in enumerate(batch_ids.tolist() if batch_ids.size else []):
        by_face[int(b)].append(di)

    for face_idx, meta in enumerate(face_meta[:max_faces]):
        rgb_path = Path(meta.get("rgb", ""))
        if not rgb_path.exists():
            continue
        img = Image.open(rgb_path).convert("RGB")
        draw = ImageDraw.Draw(img, "RGBA")
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        det_ids = by_face.get(face_idx, [])
        for j, di in enumerate(det_ids):
            color = _COLORS[j % len(_COLORS)]
            # Masks may be list aligned to batch, or flat list of size M
            mask_t = None
            if isinstance(masks, (list, tuple)):
                if len(masks) == stats["M"] and di < len(masks):
                    mask_t = masks[di]
                elif face_idx < len(masks) and len(det_ids) == 1:
                    mask_t = masks[face_idx]
                elif di < len(masks):
                    mask_t = masks[di]
            if mask_t is not None:
                m = _as_numpy(mask_t).astype(bool)
                if m.ndim == 2 and m.shape[0] == img.height and m.shape[1] == img.width:
                    # Semi-transparent overlay
                    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                    o_pix = overlay.load()
                    ys, xs = np.where(m)
                    for y, x in zip(ys[:: max(1, len(ys)//8000)], xs[:: max(1, len(xs)//8000)]):
                        o_pix[int(x), int(y)] = (*color, 90)
                    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
                    draw = ImageDraw.Draw(img, "RGBA")
                    # Bbox from mask
                    if ys.size:
                        x0, x1 = int(xs.min()), int(xs.max())
                        y0, y1 = int(ys.min()), int(ys.max())
                        draw.rectangle([x0, y0, x1, y1], outline=(*color, 255), width=2)

            lab = labels[di] if di < len(labels) else f"#{di}"
            sc = float(scores[di]) if di < len(scores) else 0.0
            z = float(means[di, 2]) if di < len(means) else float("nan")
            r = float(np.linalg.norm(means[di])) if di < len(means) else float("nan")
            text = f"{lab} {sc:.2f} |r|={r:.1f}m"
            # Rough text position
            draw.text((8, 8 + 14 * j), text, fill=(*color, 255), font=font)

        out_path = out_dir / f"{stats['kf_id']}_face{face_idx}.jpg"
        img.convert("RGB").save(out_path, quality=90)
        saved.append(out_path)

    return saved


# ---------------------------------------------------------------------------
# Aggregate + plots
# ---------------------------------------------------------------------------

def _try_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _save_hist(path: Path, data: np.ndarray, title: str, xlabel: str, bins: int = 40):
    plt = _try_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 4))
    if data.size:
        ax.hist(data, bins=bins, color="#3a7bd5", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_bar(path: Path, labels: List[str], counts: List[int], title: str):
    plt = _try_matplotlib()
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.35), 4))
    ax.bar(range(len(labels)), counts, color="#3a7bd5")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_line(path: Path, ys: List[int], title: str, ylabel: str):
    plt = _try_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(ys, color="#3a7bd5", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("keyframe order")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_xy(path: Path, means: np.ndarray, title: str):
    plt = _try_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 6))
    if means.size:
        ax.scatter(means[:, 0], means[:, 1], s=8, c=means[:, 2], cmap="viridis", alpha=0.7)
        cbar = fig.colorbar(ax.collections[0], ax=ax)
        cbar.set_label("Z (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X world (m)")
    ax.set_ylabel("Y world (m)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_plotly_3d(path: Path, means: np.ndarray, labels: List[str], scores: np.ndarray):
    try:
        import plotly.express as px
        import pandas as pd
    except ImportError:
        return False
    if means.size == 0:
        return False
    df = pd.DataFrame({
        "x": means[:, 0], "y": means[:, 1], "z": means[:, 2],
        "label": labels, "score": scores,
    })
    fig = px.scatter_3d(df, x="x", y="y", z="z", color="label", size="score",
                        size_max=12, opacity=0.75, title="Phase 2 world Gaussians (means)")
    fig.write_html(str(path))
    return True


# ---------------------------------------------------------------------------
# Expectation checks
# ---------------------------------------------------------------------------

def _evaluate(agg: dict) -> Tuple[List[str], List[str], List[str]]:
    """Return (pass_msgs, warn_msgs, fail_msgs)."""
    ok, warn, fail = [], [], []
    n_kf = agg["n_keyframes"]
    n_det = agg["n_detections"]
    empty = agg["n_empty_kf"]
    empty_rate = empty / max(1, n_kf)

    if n_kf == 0:
        fail.append("No detections_kf*.pt files found.")
        return ok, warn, fail

    ok.append(f"Loaded {n_kf} keyframe packs, {n_det} total detections.")

    if n_det == 0:
        fail.append("Zero total detections — YOLOE found nothing (vocab/conf/image issue).")
    elif empty_rate > 0.7:
        fail.append(f"{empty_rate:.0%} keyframes empty — conf too high or vocab mismatch.")
    elif empty_rate > 0.35:
        warn.append(f"{empty_rate:.0%} empty keyframes — inspect overlays; may still be OK outdoors.")
    else:
        ok.append(f"Empty keyframe rate {empty_rate:.0%} is acceptable.")

    med = agg["median_dets_per_kf"]
    if n_det > 0 and med < 1:
        warn.append(f"Median detections/kf = {med:.1f} is very low.")
    elif med > 80:
        warn.append(f"Median detections/kf = {med:.1f} is very high (noisy / double-dets).")
    else:
        ok.append(f"Median detections/kf = {med:.1f}.")

    if agg["n_nonfinite_means"] > 0:
        fail.append(f"{agg['n_nonfinite_means']} Gaussians have non-finite means (depth/K/pose bug).")
    else:
        ok.append("All means are finite.")

    if n_det and agg["frac_bad_feat_norm"] > 0.2:
        fail.append(
            f"{agg['frac_bad_feat_norm']:.0%} features have ||f|| not near 1 "
            f"(DINOv3 path broken or features zero)."
        )
    elif n_det:
        ok.append(
            f"Feature norms OK (median={agg['median_feat_norm']:.3f}, "
            f"bad={agg['frac_bad_feat_norm']:.1%})."
        )

    if n_det and agg["frac_zero_extent"] > 0.3:
        warn.append(
            f"{agg['frac_zero_extent']:.0%} Gaussians have near-zero extent "
            f"(too few depth points under mask)."
        )

    # Metric range sanity for construction walk
    if n_det and agg["radius_p95"] > 200:
        warn.append(f"P95 ||mean|| = {agg['radius_p95']:.1f} m seems large for site scale / pose unit.")
    elif n_det and agg["radius_p95"] < 0.5:
        warn.append(f"P95 ||mean|| = {agg['radius_p95']:.1f} m seems tiny (depth scale or pose issue?).")
    elif n_det:
        ok.append(f"World radii look site-scale (P50={agg['radius_p50']:.1f} m, P95={agg['radius_p95']:.1f} m).")

    if n_det and agg["score_p50"] < 0.25:
        warn.append(f"Median score {agg['score_p50']:.2f} is low — many weak detections.")
    elif n_det:
        ok.append(f"Score distribution OK (P50={agg['score_p50']:.2f}).")

    if n_det and len(agg["top_classes"]) == 0:
        warn.append("No class labels resolved.")

    return ok, warn, fail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate(
    det_dir: Path,
    out_dir: Optional[Path] = None,
    max_overlay_kfs: int = 12,
    overlay_stride: int = 10,
) -> Path:
    det_dir = det_dir.resolve()
    out_dir = (out_dir or det_dir / "validation").resolve()
    overlay_dir = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    packs = sorted(det_dir.glob("detections_kf*.pt"), key=_kf_sort_key)
    if not packs:
        packs = sorted(det_dir.glob("*.pt"), key=_kf_sort_key)

    all_scores: List[float] = []
    all_norms: List[float] = []
    all_radii: List[float] = []
    all_extents: List[float] = []
    all_means: List[np.ndarray] = []
    all_labels: List[str] = []
    dets_per_kf: List[int] = []
    class_counter: Counter = Counter()
    n_empty = 0
    n_nonfinite = 0
    n_zero_extent = 0
    n_feat_bad = 0
    n_det = 0
    overlay_paths: List[str] = []

    for i, p in enumerate(packs):
        pack = _load_pack(p)
        st = _pack_stats(pack)
        M = st["M"]
        dets_per_kf.append(M)
        n_det += M
        if M == 0:
            n_empty += 1
        n_nonfinite += st["n_nonfinite_means"]
        n_zero_extent += st["n_zero_extent"]
        n_feat_bad += st["n_feat_bad_norm"]
        if M:
            all_scores.extend(st["scores"].tolist())
            all_norms.extend(st["feat_norms"].tolist())
            all_radii.extend(st["radii"].tolist())
            all_extents.extend(st["max_extent"].tolist())
            all_means.append(st["means"])
            all_labels.extend(st["labels"])
            class_counter.update(st["labels"])

        # overlays: always first few + every stride
        if i < max_overlay_kfs or (overlay_stride > 0 and i % overlay_stride == 0):
            saved = _draw_overlays_for_pack(st, overlay_dir)
            overlay_paths.extend(str(x) for x in saved)

    means_all = np.concatenate(all_means, axis=0) if all_means else np.zeros((0, 3))
    scores_np = np.asarray(all_scores, dtype=np.float32)
    norms_np = np.asarray(all_norms, dtype=np.float32)
    radii_np = np.asarray(all_radii, dtype=np.float32)
    extents_np = np.asarray(all_extents, dtype=np.float32)

    def _pct(a, q):
        return float(np.percentile(a, q)) if a.size else float("nan")

    top = class_counter.most_common(30)
    agg = {
        "n_keyframes": len(packs),
        "n_detections": n_det,
        "n_empty_kf": n_empty,
        "median_dets_per_kf": float(np.median(dets_per_kf)) if dets_per_kf else 0.0,
        "mean_dets_per_kf": float(np.mean(dets_per_kf)) if dets_per_kf else 0.0,
        "n_nonfinite_means": n_nonfinite,
        "n_zero_extent": n_zero_extent,
        "frac_zero_extent": n_zero_extent / max(1, n_det),
        "frac_bad_feat_norm": n_feat_bad / max(1, n_det),
        "median_feat_norm": _pct(norms_np, 50),
        "score_p50": _pct(scores_np, 50),
        "score_p10": _pct(scores_np, 10),
        "score_p90": _pct(scores_np, 90),
        "radius_p50": _pct(radii_np, 50),
        "radius_p95": _pct(radii_np, 95),
        "extent_p50": _pct(extents_np, 50),
        "top_classes": top,
        "dets_per_kf": dets_per_kf,
        "overlay_count": len(overlay_paths),
        "det_dir": str(det_dir),
    }

    ok, warn, fail = _evaluate(agg)
    agg["checks"] = {"pass": ok, "warn": warn, "fail": fail}
    agg["status"] = "FAIL" if fail else ("WARN" if warn else "PASS")

    # Plots
    if means_all.size:
        _save_xy(out_dir / "world_xy_scatter.png", means_all, "World-frame detection means (XY, color=Z)")
    if scores_np.size:
        _save_hist(out_dir / "score_hist.png", scores_np, "YOLOE confidence scores", "score")
    if norms_np.size:
        _save_hist(out_dir / "feature_norm_hist.png", norms_np, "DINOv3 feature L2 norms", "||f||")
    if radii_np.size:
        _save_hist(out_dir / "mean_radii_hist.png", radii_np, "||mean|| world distance from origin", "metres")
    if extents_np.size:
        _save_hist(out_dir / "extent_hist.png", extents_np, "Approx Gaussian max std (m)", "metres")
    if dets_per_kf:
        _save_line(out_dir / "det_count_per_kf.png", dets_per_kf, "Detections per keyframe", "count")
    if top:
        _save_bar(out_dir / "class_hist.png", [t[0] for t in top], [t[1] for t in top], "Top classes")

    if means_all.size:
        _save_plotly_3d(
            out_dir / "overlays_3d.html",
            means_all,
            all_labels if len(all_labels) == len(means_all) else [f"d{i}" for i in range(len(means_all))],
            scores_np if scores_np.size == len(means_all) else np.ones(len(means_all)),
        )

    # Strip large arrays before JSON
    metrics_out = {k: v for k, v in agg.items() if k != "dets_per_kf"}
    metrics_out["dets_per_kf_head"] = dets_per_kf[:50]
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    lines = [
        f"Phase 2 validation — {agg['status']}",
        f"det_dir: {det_dir}",
        f"keyframes: {agg['n_keyframes']}  detections: {agg['n_detections']}  empty_kf: {agg['n_empty_kf']}",
        f"median dets/kf: {agg['median_dets_per_kf']:.1f}  score P50: {agg['score_p50']:.3f}",
        f"feat-norm median: {agg['median_feat_norm']:.3f}  radius P50/P95: {agg['radius_p50']:.2f}/{agg['radius_p95']:.2f} m",
        "",
        "PASS:",
        *[f"  + {m}" for m in ok],
        "WARN:",
        *([f"  ! {m}" for m in warn] if warn else ["  (none)"]),
        "FAIL:",
        *([f"  x {m}" for m in fail] if fail else ["  (none)"]),
        "",
        "Top classes:",
        *[f"  {c}: {n}" for c, n in top[:15]],
        "",
        f"Overlays: {overlay_dir} ({len(overlay_paths)} images)",
        f"Open: {out_dir / 'world_xy_scatter.png'}, {out_dir / 'overlays_3d.html'}",
        "",
        "What you should visually verify in overlays/",
        "  - Masks hug real objects (not sky/ground slabs that shouldn't be there)",
        "  - Labels are construction-plausible for the content",
        "  - Same physical object on adjacent faces may appear twice (OK; Phase 3 merges)",
        "  - Thin false positives at conf boundary → raise --conf; misses → lower --conf / expand vocab",
        "What world_xy / 3D should show",
        "  - Points form a walkable site footprint (not a random cloud at origin)",
        "  - Means sit at object-scale metres (not km or mm)",
    ]
    summary = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(summary)
    print(summary)
    return out_dir / "summary.txt"


def main():
    ap = argparse.ArgumentParser(description="Validate Phase 2 detection packs.")
    ap.add_argument(
        "--det-dir", type=Path,
        default=Path(__file__).parent / "output",
        help="Directory containing detections_kf*.pt",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="Validation output dir (default: <det-dir>/validation)",
    )
    ap.add_argument("--max-overlay-kfs", type=int, default=12)
    ap.add_argument("--overlay-stride", type=int, default=10)
    args = ap.parse_args()
    validate(args.det_dir, args.out_dir, args.max_overlay_kfs, args.overlay_stride)


if __name__ == "__main__":
    main()
