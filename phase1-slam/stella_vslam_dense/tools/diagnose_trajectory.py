#!/usr/bin/env python3
"""Trajectory diagnostics without GT (TUM poses + optional slam.log events)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_tum(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load TUM trajectory → timestamps (N,), positions (N,3)."""
    ts_list: List[float] = []
    pos_list: List[List[float]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            ts_list.append(float(parts[0]))
            pos_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not pos_list:
        raise ValueError(f"No poses in {path}")
    return np.asarray(ts_list, dtype=np.float64), np.asarray(pos_list, dtype=np.float64)


def step_stats(pos: np.ndarray) -> Dict[str, float]:
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    if d.size == 0:
        return {"n_steps": 0, "path_length": 0.0, "median": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "n_steps": int(d.size),
        "path_length": float(d.sum()),
        "median": float(np.median(d)),
        "mean": float(d.mean()),
        "p99": float(np.percentile(d, 99)),
        "max": float(d.max()),
        "jumps_gt_5x_med": int(np.sum(d > max(5.0 * np.median(d), 1e-6))),
    }


def accel_jerk_stats(pos: np.ndarray, ts: np.ndarray) -> Dict[str, float]:
    """Second-order flags on translation; uses simple finite differences."""
    if pos.shape[0] < 3:
        return {"accel_max": 0.0, "jerk_max": 0.0, "accel_p99": 0.0, "jerk_p99": 0.0}
    dt = np.diff(ts)
    dt = np.where(dt <= 1e-9, np.nan, dt)
    v = np.diff(pos, axis=0) / dt[:, None]
    valid_v = np.isfinite(v).all(axis=1)
    if valid_v.sum() < 2:
        return {"accel_max": float("nan"), "jerk_max": float("nan"), "accel_p99": float("nan"), "jerk_p99": float("nan")}
    v = v[valid_v]
    # acceleration between velocity samples
    if v.shape[0] < 2:
        return {"accel_max": 0.0, "jerk_max": 0.0, "accel_p99": 0.0, "jerk_p99": 0.0}
    a = np.diff(v, axis=0)
    a_n = np.linalg.norm(a, axis=1)
    j = np.diff(a, axis=0) if a.shape[0] > 1 else np.zeros((0, 3))
    j_n = np.linalg.norm(j, axis=1) if j.size else np.array([0.0])
    return {
        "accel_max": float(np.nanmax(a_n)) if a_n.size else 0.0,
        "accel_p99": float(np.nanpercentile(a_n, 99)) if a_n.size else 0.0,
        "jerk_max": float(np.nanmax(j_n)) if j_n.size else 0.0,
        "jerk_p99": float(np.nanpercentile(j_n, 99)) if j_n.size else 0.0,
    }


def parse_slam_log(path: Optional[Path]) -> Dict[str, int]:
    counts = {
        "tracking_lost": 0,
        "local_map_tracking_failed": 0,
        "resetting_system": 0,
        "loop_detect": 0,
        "updated_the_map": 0,
        "initialization_succeeded": 0,
    }
    if path is None or not path.is_file():
        return counts
    patterns = {
        "tracking_lost": re.compile(r"tracking lost", re.I),
        "local_map_tracking_failed": re.compile(r"local map tracking failed", re.I),
        "resetting_system": re.compile(r"resetting system", re.I),
        "loop_detect": re.compile(r"\bloop\b", re.I),
        "updated_the_map": re.compile(r"updated the map", re.I),
        "initialization_succeeded": re.compile(r"initialization succeeded", re.I),
    }
    # stream large logs; avoid loading entire multi-MB progress noise if possible
    with path.open("rb") as f:
        for raw in f:
            try:
                line = raw.decode("utf-8", errors="replace")
            except Exception:
                continue
            if "Processing frames" in line and "[info]" not in line and "[warn" not in line:
                continue
            for key, pat in patterns.items():
                if pat.search(line):
                    counts[key] += 1
    return counts


def tracking_time_stats(path: Optional[Path]) -> Dict[str, float]:
    if path is None or not path.is_file():
        return {}
    vals: List[float] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # one float per line or space-separated
            for part in line.split():
                try:
                    vals.append(float(part))
                    break
                except ValueError:
                    continue
    if not vals:
        return {}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(a.size),
        "median_s": float(np.median(a)),
        "mean_s": float(a.mean()),
        "p95_s": float(np.percentile(a, 95)),
        "p99_s": float(np.percentile(a, 99)),
        "max_s": float(a.max()),
    }


def diagnose(
    traj_path: Path,
    slam_log: Optional[Path] = None,
    tracking_times: Optional[Path] = None,
    keyframe_traj: Optional[Path] = None,
) -> str:
    lines: List[str] = []
    lines.append(f"Trajectory file: {traj_path}")
    ts, pos = load_tum(traj_path)
    n = pos.shape[0]
    duration = float(ts[-1] - ts[0]) if n > 1 else 0.0
    start_end = float(np.linalg.norm(pos[-1] - pos[0]))
    steps = step_stats(pos)
    aj = accel_jerk_stats(pos, ts)

    lines.append("")
    lines.append("=== Pose path (frame trajectory) ===")
    lines.append(f"poses:              {n}")
    lines.append(f"duration_s:         {duration:.3f}")
    lines.append(f"path_length:        {steps['path_length']:.4f}  (model units)")
    lines.append(f"start_end_gap:      {start_end:.4f}  (same units; ~0 if closed loop + no drift)")
    if steps["path_length"] > 1e-9:
        lines.append(f"start_end / path:   {start_end / steps['path_length']:.4%}")
    lines.append(
        f"step median/mean:   {steps['median']:.6f} / {steps['mean']:.6f}"
    )
    lines.append(f"step p99 / max:     {steps['p99']:.6f} / {steps['max']:.6f}")
    lines.append(f"jumps (>5x median): {steps['jumps_gt_5x_med']}")
    lines.append(f"accel p99/max:      {aj['accel_p99']:.6g} / {aj['accel_max']:.6g}")
    lines.append(f"jerk  p99/max:      {aj['jerk_p99']:.6g} / {aj['jerk_max']:.6g}")

    if keyframe_traj and keyframe_traj.is_file():
        kts, kpos = load_tum(keyframe_traj)
        ksteps = step_stats(kpos)
        kgap = float(np.linalg.norm(kpos[-1] - kpos[0]))
        lines.append("")
        lines.append("=== Keyframe trajectory ===")
        lines.append(f"keyframes:          {kpos.shape[0]}")
        lines.append(f"path_length:        {ksteps['path_length']:.4f}")
        lines.append(f"start_end_gap:      {kgap:.4f}")

    log_counts = parse_slam_log(slam_log)
    lines.append("")
    lines.append("=== slam.log events (counts) ===")
    if slam_log:
        lines.append(f"log: {slam_log}")
    for k, v in log_counts.items():
        lines.append(f"{k}: {v}")

    tt = tracking_time_stats(tracking_times)
    if tt:
        lines.append("")
        lines.append("=== tracking_times.txt ===")
        for k, v in tt.items():
            if isinstance(v, float):
                lines.append(f"{k}: {v:.6f}")
            else:
                lines.append(f"{k}: {v}")

    lines.append("")
    lines.append("=== How to read (no GT) ===")
    lines.append("- start_end_gap large while you *did* return to start → drift / failed loop glue.")
    lines.append("- jumps (>5x median) and huge accel/jerk → tracking breaks or recovery glitches.")
    lines.append("- many tracking_lost / resets → brittle pose chain for global unprojection.")
    lines.append("- monocular scale is arbitrary; these metrics are about *stability/shape*, not meters.")
    lines.append("- for site check, see POSE_CHECK.md next to this run.")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnose TUM trajectory quality without ground truth")
    ap.add_argument("--traj", type=Path, required=True, help="frame_trajectory.txt (TUM)")
    ap.add_argument("--keyframe-traj", type=Path, default=None, help="keyframe_trajectory.txt")
    ap.add_argument("--slam-log", type=Path, default=None)
    ap.add_argument("--tracking-times", type=Path, default=None)
    ap.add_argument("-o", "--out", type=Path, default=None, help="Write report here")
    args = ap.parse_args()

    report = diagnose(
        args.traj,
        slam_log=args.slam_log,
        tracking_times=args.tracking_times,
        keyframe_traj=args.keyframe_traj,
    )
    print(report, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
