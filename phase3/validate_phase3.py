#!/usr/bin/env python3
"""Phase 3 — Validation

Loads `scene_state.pt` and `run_stats.json` from the Phase 3 output directory
and produces a suite of diagnostic plots and a PASS/WARN/FAIL summary.

Output artifacts (all under ``output/validation/``)
----------------------------------------------------
object_count_growth.png   — N total / active objects vs keyframe index
merge_rate.png            — new / merged detections per keyframe
world_xy_scatter.png      — top-down (X,Y) scatter of final object means
overlays_3d.html          — interactive Plotly 3D of object means + labels
class_breakdown.png       — object count per class label
feature_consistency.png   — intra-object feature variance (lower = stable)
metrics.json              — numeric summary
summary.txt               — PASS / WARN / FAIL gate report

Gate to Phase 4
---------------
PASS  — merging occurred; object count well below 2000; scene footprint coherent
WARN  — minor anomalies worth checking but not blocking
FAIL  — a hard constraint is broken (no merges, runaway growth, etc.)

Usage
-----
conda run -n farm-phase2 python validate_phase3.py \\
    --output-dir ./output \\
    [--vocab-file ../phase2-detect-segment-embed/vocab/construction_vocab.txt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Optional soft dependencies — degrade gracefully
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MATPLOTLIB = True
except ImportError:
    _MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

# ---------------------------------------------------------------------------
# FARM path setup
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


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_line(
    path: Path,
    xs: List,
    ys_dict: Dict[str, List],
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    if not _MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, ys in ys_dict.items():
        ax.plot(xs, ys, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_bar(path: Path, labels: List[str], values: List[int], title: str) -> None:
    if not _MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.45), 5))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("Number of objects")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_scatter(
    path: Path,
    xs: np.ndarray,
    ys: np.ndarray,
    colors: Optional[np.ndarray],
    labels_map: Dict[int, str],
    title: str,
) -> None:
    if not _MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    if colors is not None and len(np.unique(colors)) <= 50:
        unique_colors = np.unique(colors)
        cmap = plt.get_cmap("tab20", len(unique_colors))
        for i, c in enumerate(unique_colors):
            mask = colors == c
            label = labels_map.get(int(c), str(c))
            ax.scatter(xs[mask], ys[mask], s=12, alpha=0.7,
                       color=cmap(i), label=label)
        ax.legend(loc="upper right", fontsize=6, markerscale=2,
                  ncol=max(1, len(unique_colors) // 20))
    else:
        ax.scatter(xs, ys, s=12, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_hist(path: Path, values: np.ndarray, title: str, xlabel: str) -> None:
    if not _MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=50, edgecolor="none", alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_plotly_3d(
    path: Path,
    means: np.ndarray,
    class_ids: np.ndarray,
    vocab: List[str],
    n_active_mask: np.ndarray,
) -> None:
    if not _PLOTLY:
        return
    labels = [vocab[int(c)] if 0 <= int(c) < len(vocab) else str(c)
              for c in class_ids]
    colors = np.where(n_active_mask, "rgba(50,200,50,0.8)", "rgba(180,180,180,0.4)")
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=means[:, 0].tolist(),
                y=means[:, 1].tolist(),
                z=means[:, 2].tolist(),
                mode="markers+text",
                marker=dict(
                    size=4,
                    color=class_ids.tolist(),
                    colorscale="Viridis",
                    opacity=0.8,
                    colorbar=dict(title="class id"),
                ),
                text=labels,
                textposition="top center",
                textfont=dict(size=8),
                hovertemplate="<b>%{text}</b><br>x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}",
            )
        ]
    )
    fig.update_layout(
        title="Phase 3 — Object Means (3D)",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.write_html(str(path))


# ---------------------------------------------------------------------------
# Feature consistency
# ---------------------------------------------------------------------------

def _feature_variance_per_class(
    features: np.ndarray,
    class_ids: np.ndarray,
    active: np.ndarray,
) -> Dict[int, float]:
    """Mean intra-class feature variance (L2 distance from class centroid)."""
    result: Dict[int, float] = {}
    if features is None or features.size == 0:
        return result
    active_ids = np.unique(class_ids[active])
    for c in active_ids:
        mask = (class_ids == c) & active
        feats = features[mask]
        if feats.shape[0] < 2:
            continue
        centroid = feats.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(feats - centroid, axis=1)
        result[int(c)] = float(dists.mean())
    return result


# ---------------------------------------------------------------------------
# PASS / WARN / FAIL evaluation
# ---------------------------------------------------------------------------

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def _evaluate(
    n_active: int,
    n_total: int,
    n_total_phase2_dets: int,
    merge_rate_nonzero: bool,
    has_coherent_scatter: bool,
) -> List[tuple[str, str, str]]:
    """Return list of (status, check_name, message) tuples."""
    results: List[tuple[str, str, str]] = []

    # 1. At least some merging happened
    if n_active == 0:
        results.append((FAIL, "nonzero_objects", "No active objects in scene state."))
    elif n_total_phase2_dets > 0 and n_active >= n_total_phase2_dets * 0.95:
        results.append((
            FAIL, "merge_occurred",
            f"Active objects ({n_active}) is ≥95% of Phase 2 detections "
            f"({n_total_phase2_dets}). Merging may not have occurred.",
        ))
    elif n_active >= n_total_phase2_dets * 0.7:
        results.append((
            WARN, "merge_rate_low",
            f"Active objects ({n_active}) is ≥70% of Phase 2 detections — "
            "low merge rate.  Check thresholds.",
        ))
    else:
        results.append((
            PASS, "merge_occurred",
            f"Merging occurred: {n_total_phase2_dets} detections → {n_active} active objects.",
        ))

    # 2. Runaway growth guard
    if n_active > 2000:
        results.append((
            WARN, "object_count",
            f"Active object count ({n_active}) exceeds 2000. "
            "May include over-segmented background.",
        ))
    else:
        results.append((
            PASS, "object_count",
            f"Active object count ({n_active}) is within reasonable bounds.",
        ))

    # 3. Non-zero merges per kf
    if not merge_rate_nonzero:
        results.append((
            WARN, "merge_activity",
            "No merged detections across any keyframe. "
            "Possible thresholds too tight.",
        ))
    else:
        results.append((PASS, "merge_activity", "Merging activity detected across keyframes."))

    # 4. Scatter coherence (crude)
    if not has_coherent_scatter:
        results.append((
            WARN, "scatter_coherence",
            "Object means span >500 m — possible depth scale issue or outliers.",
        ))
    else:
        results.append((PASS, "scatter_coherence", "World scatter footprint is reasonable."))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate(
    output_dir: Path,
    vocab_file: Optional[Path] = None,
) -> None:
    val_dir = output_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load scene state ------------------------------------------------
    state_path = output_dir / "scene_state.pt"
    if not state_path.exists():
        print(f"[ERROR] scene_state.pt not found at {state_path}")
        sys.exit(1)

    print(f"Loading scene state from {state_path} …")
    state = torch.load(state_path, map_location="cpu", weights_only=False)

    means_t = state.get("means")
    cov6_t = state.get("cov6")
    features_t = state.get("features")
    class_ids_t = state.get("class_ids")
    active_t = state.get("active")

    means = means_t.numpy() if isinstance(means_t, torch.Tensor) else np.zeros((0, 3))
    features = features_t.numpy() if isinstance(features_t, torch.Tensor) else np.zeros((0, 1))
    class_ids = (
        class_ids_t.numpy().astype(int) if isinstance(class_ids_t, torch.Tensor)
        else np.zeros(means.shape[0], dtype=int)
    )
    active = (
        active_t.numpy().astype(bool) if isinstance(active_t, torch.Tensor)
        else np.ones(means.shape[0], dtype=bool)
    )

    n_total = int(means.shape[0])
    n_active = int(active.sum())

    # ---- Load vocab -------------------------------------------------------
    vocab: List[str] = []
    if vocab_file and vocab_file.exists():
        vocab = [l.strip() for l in vocab_file.read_text().splitlines() if l.strip()]
    labels_map = {i: v for i, v in enumerate(vocab)}

    # ---- Load run stats ---------------------------------------------------
    stats_path = output_dir / "run_stats.json"
    stats: List[dict] = []
    if stats_path.exists():
        with open(stats_path) as fh:
            stats = json.load(fh)

    n_total_phase2_dets = int(sum(s.get("n_raw", 0) for s in stats))
    kf_indices = [s["kf"] for s in stats]
    n_new_per_kf = [s.get("n_new", 0) for s in stats]
    n_merged_per_kf = [s.get("n_merged", 0) for s in stats]
    n_obj_per_kf = [s.get("n_objects", 0) for s in stats]
    n_active_per_kf = [s.get("n_active", 0) for s in stats]

    merge_rate_nonzero = any(v > 0 for v in n_merged_per_kf)

    # ---- Object count growth ----------------------------------------------
    _save_line(
        val_dir / "object_count_growth.png",
        xs=kf_indices,
        ys_dict={"total": n_obj_per_kf, "active": n_active_per_kf},
        title="Object count vs keyframe",
        xlabel="Keyframe index",
        ylabel="N objects",
    )

    # ---- Merge rate -------------------------------------------------------
    _save_line(
        val_dir / "merge_rate.png",
        xs=kf_indices,
        ys_dict={"new": n_new_per_kf, "merged": n_merged_per_kf},
        title="New vs merged detections per keyframe",
        xlabel="Keyframe index",
        ylabel="Count",
    )

    # ---- World XY scatter -------------------------------------------------
    if n_total > 0:
        scatter_colors = class_ids if class_ids.shape[0] == means.shape[0] else None
        # Use only active objects for the primary scatter
        if active.any():
            _save_scatter(
                val_dir / "world_xy_scatter.png",
                xs=means[active, 0],
                ys=means[active, 1],
                colors=scatter_colors[active] if scatter_colors is not None else None,
                labels_map=labels_map,
                title="Top-down XY scatter of active objects",
            )
        footprint = float(np.ptp(means[active, :2].flatten())) if active.any() else 0.0
        has_coherent_scatter = footprint < 500.0
    else:
        has_coherent_scatter = False
        footprint = 0.0

    # ---- 3D HTML ----------------------------------------------------------
    if n_total > 0:
        _save_plotly_3d(
            val_dir / "overlays_3d.html",
            means=means,
            class_ids=class_ids,
            vocab=vocab,
            n_active_mask=active,
        )

    # ---- Class breakdown --------------------------------------------------
    if n_total > 0 and active.any():
        unique_classes, counts = np.unique(class_ids[active], return_counts=True)
        class_labels = [labels_map.get(int(c), str(c)) for c in unique_classes]
        _save_bar(
            val_dir / "class_breakdown.png",
            labels=class_labels,
            values=counts.tolist(),
            title="Active object count per class",
        )
    else:
        unique_classes, counts = np.array([]), np.array([])

    # ---- Feature consistency ----------------------------------------------
    if features is not None and features.shape[0] > 0:
        var_per_class = _feature_variance_per_class(features, class_ids, active)
        if var_per_class:
            class_labels_feat = [labels_map.get(c, str(c)) for c in var_per_class]
            var_values = list(var_per_class.values())
            _save_bar(
                val_dir / "feature_consistency.png",
                labels=class_labels_feat,
                values=[round(v * 1000) for v in var_values],
                title="Mean intra-class feature variance × 1000 (lower = stable)",
            )

    # ---- Metrics JSON -----------------------------------------------------
    covis_adj = state.get("covisibility_adj")
    n_covis_edges = 0
    if isinstance(covis_adj, torch.Tensor):
        n_covis_edges = int((covis_adj > 0).sum().item())

    class_counts_dict = {
        labels_map.get(int(c), str(c)): int(cnt)
        for c, cnt in zip(unique_classes.tolist(), counts.tolist())
    } if len(unique_classes) > 0 else {}

    metrics = {
        "n_objects": n_total,
        "n_active": n_active,
        "n_inactive": n_total - n_active,
        "n_total_phase2_detections": n_total_phase2_dets,
        "n_covisibility_edges": n_covis_edges,
        "footprint_xy_m": round(footprint, 2),
        "per_class_active_counts": class_counts_dict,
    }
    metrics_path = val_dir / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"metrics.json → {metrics_path}")

    # ---- PASS / WARN / FAIL gate -----------------------------------------
    gate_results = _evaluate(
        n_active=n_active,
        n_total=n_total,
        n_total_phase2_dets=n_total_phase2_dets,
        merge_rate_nonzero=merge_rate_nonzero,
        has_coherent_scatter=has_coherent_scatter,
    )

    lines = [
        "Phase 3 Validation Summary",
        "=" * 50,
        "",
        f"  Total objects   : {n_total}",
        f"  Active objects  : {n_active}",
        f"  Covis edges     : {n_covis_edges}",
        f"  Scene footprint : {footprint:.1f} m (XY range)",
        f"  Phase2 dets     : {n_total_phase2_dets}",
        "",
        "Gate checks:",
    ]
    overall = PASS
    for status, check, msg in gate_results:
        lines.append(f"  [{status:4}]  {check}: {msg}")
        if status == FAIL:
            overall = FAIL
        elif status == WARN and overall == PASS:
            overall = WARN

    lines += ["", f"Overall: {overall}", ""]
    lines.append("Plots written to: " + str(val_dir))

    summary_text = "\n".join(lines)
    summary_path = val_dir / "summary.txt"
    summary_path.write_text(summary_text)
    print(summary_text)
    print(f"summary.txt → {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 — Validate scene_state.pt")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "output",
        help="Phase 3 output directory containing scene_state.pt and run_stats.json",
    )
    p.add_argument(
        "--vocab-file",
        type=Path,
        default=Path(__file__).parent.parent
        / "phase2-detect-segment-embed"
        / "vocab"
        / "construction_vocab.txt",
        help="Path to the construction vocabulary text file",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    validate(output_dir=args.output_dir, vocab_file=args.vocab_file)
