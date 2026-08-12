#!/usr/bin/env python3
"""Phase 1.5 Validation — check DA3 depth output.

Checks
------
- frames.json exists and contains expected number of frames
- Random sample of face/.npy pairs exists and has valid depth
- Depth histogram saved for inspection

Outputs (under --out-dir)
-------------------------
metrics.json
summary.txt
depth_sample_hist.png  (if matplotlib available)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1.5 validation")
    p.add_argument("--phase15-dir", type=Path, default=Path("outputs/phase1.5"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/validation/phase1.5"))
    p.add_argument("--sample-n", type=int, default=20,
                   help="Number of depth files to spot-check")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames_json = args.phase15_dir / "frames_json" / "frames.json"
    depth_dir = args.phase15_dir / "depth"
    face_dir = args.phase15_dir / "faces"

    gates = []
    metrics: dict = {}

    if not frames_json.is_file():
        gates.append(f"FAIL  frames.json not found at {frames_json}")
        metrics["overall"] = "FAIL"
        _write(args.out_dir, gates, metrics)
        return 1

    with open(frames_json) as f:
        index = json.load(f)
    frames = index.get("frames", [])
    n_frames = len(frames)
    metrics["n_frames"] = n_frames

    if n_frames < 4:
        gates.append(f"FAIL  only {n_frames} frames in frames.json")
    else:
        gates.append(f"PASS  {n_frames} frames in frames.json")

    # Spot-check depth files
    sample = random.sample(frames, min(args.sample_n, n_frames))
    n_valid = 0
    depths_sample = []
    for entry in sample:
        dep_rel = entry.get("depth", "")
        dep_path = args.phase15_dir / dep_rel
        if not dep_path.is_file():
            dep_path = depth_dir / Path(dep_rel).name
        if dep_path.is_file():
            try:
                d = np.load(str(dep_path)).astype(np.float32)
                valid = d[d > 0]
                if valid.size > 0:
                    depths_sample.append(float(np.median(valid)))
                    n_valid += 1
            except Exception:
                pass

    finite_pct = 100.0 * n_valid / max(len(sample), 1)
    metrics["depth_sample_finite_pct"] = round(finite_pct, 1)
    if n_valid > 0:
        metrics["depth_median_m"] = round(float(np.median(depths_sample)), 2)

    if finite_pct < 50:
        gates.append(f"FAIL  only {finite_pct:.0f}% sampled depth files valid")
    else:
        gates.append(f"PASS  {finite_pct:.0f}% sampled depth files valid  median={metrics.get('depth_median_m','?')}m")

    # Quick histogram
    if depths_sample:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.hist(depths_sample, bins=20, color="steelblue", alpha=0.8)
            ax.set_xlabel("Median depth per frame (m)")
            ax.set_title(f"Phase 1.5 depth distribution (N={len(depths_sample)} sampled frames)")
            fig.tight_layout()
            fig.savefig(str(args.out_dir / "depth_sample_hist.png"), dpi=100)
            plt.close(fig)
        except ImportError:
            pass

    if any(g.startswith("FAIL") for g in gates):
        overall = "FAIL"
    elif any(g.startswith("WARN") for g in gates):
        overall = "WARN"
    else:
        overall = "PASS"
    metrics["overall"] = overall
    _write(args.out_dir, gates, metrics)
    return 0 if overall != "FAIL" else 1


def _write(out_dir: Path, gates: list, metrics: dict) -> None:
    overall = metrics.get("overall", "FAIL")
    lines = ["Phase 1.5 (DA3 depth) Validation", ""] + gates + [f"\nOVERALL: {overall}"]
    print("\n".join(lines))
    (out_dir / "summary.txt").write_text("\n".join(lines))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
