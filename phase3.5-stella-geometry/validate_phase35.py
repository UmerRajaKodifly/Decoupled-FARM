#!/usr/bin/env python3
"""Phase 3.5 Validation — Stella geometry vs. Phase 3 Gaussian geometry.

Compares box-size distributions before and after Stella refinement and
produces pass/warn/fail gates.

Outputs (under --out-dir)
-------------------------
box_comparison.png   side-by-side histograms of 5σ box sides
coverage.png         Stella support points per object
metrics.json         numeric summary
summary.txt          PASS / WARN / FAIL gate report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False


def _cov6_to_sigma(cov6: torch.Tensor) -> np.ndarray:
    """Return (N, 3) σ per axis = sqrt of cov diagonal."""
    diag = torch.stack([cov6[:, 0], cov6[:, 3], cov6[:, 5]], dim=1)
    return torch.sqrt(diag.clamp(min=0)).numpy()


def _box5_sides(cov6: torch.Tensor) -> np.ndarray:
    """5σ box side lengths (N, 3)."""
    return 5.0 * _cov6_to_sigma(cov6)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 3.5 Validation — Stella vs Gaussian geometry"
    )
    p.add_argument("--phase3-state", type=Path,
                   default=Path("outputs/phase3/scene_state.pt"),
                   help="Original Phase 3 scene_state.pt")
    p.add_argument("--stella-state", type=Path,
                   default=Path("outputs/phase3.5/scene_state_stella.pt"),
                   help="Phase 3.5 Stella-updated scene_state.pt")
    p.add_argument("--summary-json", type=Path,
                   default=Path("outputs/phase3.5/phase35_summary.json"),
                   help="Phase 3.5 run summary.json (optional)")
    p.add_argument("--out-dir", type=Path,
                   default=Path("outputs/validation/phase3.5"),
                   help="Where to write plots and summary")
    p.add_argument("--vocab-file", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.stella_state.is_file():
        print(f"[phase3.5 validate] ERROR: stella state not found: {args.stella_state}")
        return 2

    ss_new = torch.load(args.stella_state, map_location="cpu", weights_only=False)
    active_new = ss_new.get("active")
    cov6_new = ss_new.get("cov6")

    act_mask = (
        active_new.numpy().astype(bool)
        if isinstance(active_new, torch.Tensor)
        else None
    )
    n_total = int(cov6_new.shape[0]) if isinstance(cov6_new, torch.Tensor) else 0
    n_active = int(act_mask.sum()) if act_mask is not None else n_total

    box_new = _box5_sides(cov6_new)
    if act_mask is not None:
        box_new_act = box_new[act_mask]
    else:
        box_new_act = box_new

    max_side_new = box_new_act.max(axis=1)

    stella_n_pts = ss_new.get("stella_n_pts")
    n_pts_arr = (
        stella_n_pts.numpy()
        if isinstance(stella_n_pts, torch.Tensor)
        else np.zeros(n_total)
    )
    n_updated = int((n_pts_arr > 0).sum())
    coverage_pct = 100.0 * n_updated / max(n_active, 1)

    # Phase 3 comparison if available
    p3_p95_max: Optional[float] = None
    p3_median_max: Optional[float] = None
    box_old_act: Optional[np.ndarray] = None
    if args.phase3_state.is_file():
        ss_old = torch.load(args.phase3_state, map_location="cpu", weights_only=False)
        cov6_old = ss_old.get("cov6")
        active_old = ss_old.get("active")
        if isinstance(cov6_old, torch.Tensor):
            box_old = _box5_sides(cov6_old)
            if isinstance(active_old, torch.Tensor):
                box_old_act = box_old[active_old.numpy().astype(bool)]
            else:
                box_old_act = box_old
            p3_p95_max = float(np.percentile(box_old_act.max(axis=1), 95))
            p3_median_max = float(np.median(box_old_act.max(axis=1)))

    new_p95_max = float(np.percentile(max_side_new, 95))
    new_median_max = float(np.median(max_side_new))
    n_big_new = int((max_side_new > 5.0).sum())

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if _MPL:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].set_title("Max 5σ box side (active objects)")
        bins = np.linspace(0, min(30.0, float(max_side_new.max()) + 1), 50)
        axes[0].hist(max_side_new, bins=bins, alpha=0.7, label="Stella", color="steelblue")
        if box_old_act is not None:
            old_max = box_old_act.max(axis=1)
            axes[0].hist(old_max, bins=bins, alpha=0.5, label="Gaussian", color="tomato")
        axes[0].axvline(5.0, color="red", linestyle="--", label="5 m threshold")
        axes[0].set_xlabel("Max 5σ side (m)")
        axes[0].set_ylabel("# objects")
        axes[0].legend()

        axes[1].set_title("Stella support pts / object (log scale)")
        pts_vals = n_pts_arr[act_mask] if act_mask is not None else n_pts_arr
        axes[1].hist(pts_vals[pts_vals > 0], bins=40, color="steelblue", alpha=0.7)
        axes[1].set_xlabel("# Stella inlier points")
        axes[1].set_ylabel("# objects")
        axes[1].set_yscale("log")

        plt.tight_layout()
        out_fig = args.out_dir / "box_comparison.png"
        fig.savefig(str(out_fig), dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[phase3.5 validate] Saved {out_fig}")

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------
    gates = []

    if coverage_pct < 30.0:
        gates.append(f"WARN  coverage {coverage_pct:.0f}% < 30% (check tau/mask alignment)")
    elif coverage_pct < 10.0:
        gates.append(f"FAIL  coverage {coverage_pct:.0f}% too low")
    else:
        gates.append(f"PASS  coverage {coverage_pct:.1f}% of active objects updated")

    if new_p95_max < 5.0:
        gates.append(f"PASS  p95 max-side={new_p95_max:.2f}m < 5 m (tight)")
    elif new_p95_max < 10.0:
        gates.append(f"WARN  p95 max-side={new_p95_max:.2f}m — some large boxes remain")
    else:
        gates.append(f"FAIL  p95 max-side={new_p95_max:.2f}m — geometry still bloated")

    if p3_p95_max is not None:
        improvement = 100.0 * (1.0 - new_p95_max / p3_p95_max)
        gates.append(
            f"INFO  p95 improvement vs Gaussian: {improvement:.0f}% "
            f"({p3_p95_max:.1f}m → {new_p95_max:.1f}m)"
        )

    if n_big_new > 30:
        gates.append(f"WARN  {n_big_new} objects still have 5σ > 5 m")
    else:
        gates.append(f"PASS  only {n_big_new} objects with 5σ > 5 m")

    if any(g.startswith("FAIL") for g in gates):
        overall = "FAIL"
    elif any(g.startswith("WARN") for g in gates):
        overall = "WARN"
    else:
        overall = "PASS"

    summary_lines = [
        f"Phase 3.5 Stella Geometry Validation",
        f"  Input Stella state : {args.stella_state}",
        f"  Objects total/active: {n_total}/{n_active}",
        f"  Stella updated: {n_updated} ({coverage_pct:.1f}%)",
        f"  New p95 max-side: {new_p95_max:.2f} m",
        f"  New median max-side: {new_median_max:.2f} m",
        "",
    ] + gates + ["", f"OVERALL: {overall}"]

    txt_path = args.out_dir / "summary.txt"
    txt_path.write_text("\n".join(summary_lines))
    print("\n".join(summary_lines))

    metrics = {
        "n_total": n_total,
        "n_active": n_active,
        "n_stella_updated": n_updated,
        "coverage_pct": round(coverage_pct, 1),
        "stella_p95_max_side_m": round(new_p95_max, 3),
        "stella_median_max_side_m": round(new_median_max, 3),
        "n_big_boxes_gt5m": n_big_new,
        "gaussian_p95_max_side_m": round(p3_p95_max, 3) if p3_p95_max is not None else None,
        "overall": overall,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return 0 if overall != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
