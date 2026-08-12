#!/usr/bin/env python3
"""Phase 4a Validation — crop quality contact sheet + ranking diagnostics.

Outputs (under --out-dir)
-------------------------
crop_grid.html         HTML contact sheet with all crops + object labels
score_sim_hist.png     feature_sim and quality score histograms
fail_list.json         objects with no crop or low sim
metrics.json           numeric summary
summary.txt            PASS / WARN / FAIL
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    _MPL = True
except ImportError:
    _MPL = False

try:
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 4a validation")
    p.add_argument("--phase4-dir", type=Path,
                   default=Path("outputs/phase4"),
                   help="Phase 4a output dir (contains best_views.json + crops/)")
    p.add_argument("--out-dir", type=Path,
                   default=Path("outputs/validation/phase4"),
                   help="Where to write validation artifacts")
    p.add_argument("--vocab-file", type=Path, default=None)
    return p.parse_args()


def _load_vocab(path: Optional[Path]) -> List[str]:
    if path is None or not path.is_file():
        return []
    return [ln.strip() for ln in path.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def main() -> int:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bv_json = args.phase4_dir / "best_views.json"
    summary_json = args.phase4_dir / "phase4a_summary.json"

    if not bv_json.is_file():
        print(f"[phase4 validate] ERROR: {bv_json} not found")
        return 2

    with open(bv_json) as f:
        bv_list = json.load(f)

    summary_data: dict = {}
    if summary_json.is_file():
        with open(summary_json) as f:
            summary_data = json.load(f)

    vocab = _load_vocab(args.vocab_file)

    def _class_name(cid: int) -> str:
        if 0 <= cid < len(vocab):
            return vocab[cid]
        return f"class_{cid}"

    n_total = len(bv_list)
    n_ok = sum(1 for r in bv_list if r.get("ok"))
    n_fail = n_total - n_ok

    sims = [r.get("feature_sim", 0.0) for r in bv_list if r.get("ok")]
    qualities = [r.get("quality", 0.0) for r in bv_list if r.get("ok")]
    dists = [r.get("mean_dist_m", 0.0) for r in bv_list if r.get("ok")]

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if _MPL and sims:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        axes[0].hist(sims, bins=30, color="steelblue", alpha=0.8)
        axes[0].axvline(0.50, color="red", linestyle="--", label="sim=0.50")
        axes[0].set_title("Feature cosine similarity")
        axes[0].set_xlabel("sim")
        axes[0].legend()

        axes[1].hist(qualities, bins=30, color="seagreen", alpha=0.8)
        axes[1].set_title("Crop quality (score × √pixels)")
        axes[1].set_xlabel("quality")

        axes[2].hist(dists, bins=30, color="darkorange", alpha=0.8)
        axes[2].axvline(1.5, color="red", linestyle="--", label="1.5 m gate")
        axes[2].set_title("Object–detection centre dist (m)")
        axes[2].set_xlabel("dist (m)")
        axes[2].legend()

        plt.tight_layout()
        fig.savefig(str(args.out_dir / "score_sim_hist.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # HTML contact sheet
    # ------------------------------------------------------------------
    if _PLOTLY:
        crops_dir = args.phase4_dir / "crops"
        ok_results = [r for r in bv_list if r.get("ok") and r.get("crop_path")]
        # Build a simple HTML grid (not Plotly image — just <img> tags)
        rows_html = []
        per_row = 6
        for i in range(0, len(ok_results), per_row):
            chunk = ok_results[i: i + per_row]
            cells = ""
            for r in chunk:
                crop_p = Path(r["crop_path"])
                label = _class_name(r.get("class_id", -1))
                sim = r.get("feature_sim", 0.0)
                dist = r.get("mean_dist_m", 0.0)
                # Use relative path if within phase4 dir
                try:
                    rel = crop_p.relative_to(args.phase4_dir)
                    src = str(rel)
                except ValueError:
                    src = str(crop_p)
                cells += (
                    f'<td style="text-align:center;padding:4px">'
                    f'<img src="../../../{args.phase4_dir}/{src}" '
                    f'style="max-height:120px;max-width:160px;border:1px solid #ccc">'
                    f'<br><small><b>{label}</b><br>'
                    f'sim={sim:.2f} dist={dist:.1f}m'
                    f'</small></td>'
                )
            rows_html.append(f"<tr>{cells}</tr>")

        html = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            "<title>Phase 4a Crops</title></head><body>"
            f"<h2>Phase 4a Crops — {n_ok} ok / {n_fail} fail</h2>"
            f"<p>feat_sim_min={summary_data.get('feat_sim_min','?')}  "
            f"max_center_dist={summary_data.get('max_center_dist_m','?')}m</p>"
            "<table border='0'>" + "\n".join(rows_html) + "</table>"
            "</body></html>"
        )
        grid_path = args.out_dir / "crop_grid.html"
        grid_path.write_text(html, encoding="utf-8")
        print(f"[phase4 validate] Crop grid → {grid_path}")

    # ------------------------------------------------------------------
    # Fail list
    # ------------------------------------------------------------------
    fail_list = [
        {"object_index": r.get("object_index"), "reason": r.get("reason", ""),
         "class_id": r.get("class_id", -1), "class": _class_name(r.get("class_id", -1))}
        for r in bv_list if not r.get("ok")
    ]
    (args.out_dir / "fail_list.json").write_text(
        json.dumps(fail_list, indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------
    crop_pct = 100.0 * n_ok / max(n_total, 1)
    median_sim = float(np.median(sims)) if sims else 0.0
    median_dist = float(np.median(dists)) if dists else 0.0

    gates = []
    if crop_pct >= 60:
        gates.append(f"PASS  crop coverage {crop_pct:.0f}% (≥60% active objects)")
    elif crop_pct >= 30:
        gates.append(f"WARN  crop coverage {crop_pct:.0f}% — many objects missing crops")
    else:
        gates.append(f"FAIL  crop coverage {crop_pct:.0f}% — most objects without crops")

    if median_sim >= 0.50:
        gates.append(f"PASS  median feature_sim={median_sim:.2f} (≥0.50)")
    else:
        gates.append(f"WARN  median feature_sim={median_sim:.2f} (<0.50 — crops may not match)")

    if median_dist < 1.5:
        gates.append(f"PASS  median centre dist={median_dist:.2f}m (<1.5 m)")
    else:
        gates.append(f"WARN  median centre dist={median_dist:.2f}m (crops from far detections)")

    if any(g.startswith("FAIL") for g in gates):
        overall = "FAIL"
    elif any(g.startswith("WARN") for g in gates):
        overall = "WARN"
    else:
        overall = "PASS"

    lines = [
        "Phase 4a Crop Validation",
        f"  n_ok={n_ok}  n_fail={n_fail}  coverage={crop_pct:.1f}%",
        f"  median feature_sim={median_sim:.3f}",
        f"  median centre_dist={median_dist:.2f}m",
        "",
    ] + gates + ["", f"OVERALL: {overall}"]
    print("\n".join(lines))
    (args.out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    metrics = {
        "n_ok": n_ok, "n_fail": n_fail, "crop_pct": round(crop_pct, 1),
        "median_feature_sim": round(median_sim, 3),
        "median_centre_dist_m": round(median_dist, 3),
        "overall": overall,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return 0 if overall != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
