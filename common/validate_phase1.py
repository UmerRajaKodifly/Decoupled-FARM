#!/usr/bin/env python3
"""Phase 1 Validation — check Stella VSLAM outputs.

Checks
------
- out.db exists and has keyframes, landmarks, dense_points
- keyframe_trajectory.txt has expected number of poses
- out.ply exists (size > 0)

Outputs (under --out-dir)
-------------------------
metrics.json
summary.txt
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 validation")
    p.add_argument("--phase1-dir", type=Path, default=Path("outputs/phase1"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/validation/phase1"))
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    db_path = args.phase1_dir / "out.db"
    traj_path = args.phase1_dir / "traj" / "keyframe_trajectory.txt"
    ply_path = args.phase1_dir / "out.ply"

    gates = []
    metrics: dict = {}

    # DB checks
    if not db_path.is_file():
        gates.append("FAIL  out.db not found")
        metrics["db_ok"] = False
    else:
        try:
            conn = sqlite3.connect(str(db_path))
            n_kf = conn.execute("SELECT COUNT(*) FROM keyframes").fetchone()[0]
            n_lm = conn.execute("SELECT COUNT(*) FROM landmarks").fetchone()[0]
            has_dense = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='dense_points'"
            ).fetchone() is not None
            n_dense = conn.execute("SELECT COUNT(*) FROM dense_points").fetchone()[0] if has_dense else 0
            conn.close()
            metrics.update(n_keyframes=n_kf, n_landmarks=n_lm, n_dense_points=n_dense)
            if n_kf < 5:
                gates.append(f"FAIL  only {n_kf} keyframes (SLAM likely failed)")
            else:
                gates.append(f"PASS  {n_kf} keyframes, {n_lm} landmarks, {n_dense} dense pts")
            if n_dense == 0:
                gates.append("WARN  dense_points table is empty — Phase 3.5 will use landmarks only")
        except Exception as exc:
            gates.append(f"FAIL  could not read out.db: {exc}")
            metrics["db_ok"] = False

    # Trajectory
    if not traj_path.is_file():
        gates.append("WARN  keyframe_trajectory.txt not found")
    else:
        lines = [l for l in traj_path.read_text().splitlines() if l.strip() and not l.startswith("#")]
        metrics["traj_poses"] = len(lines)
        if len(lines) < 5:
            gates.append(f"FAIL  trajectory has only {len(lines)} poses")
        else:
            gates.append(f"PASS  trajectory: {len(lines)} poses")

    # PLY
    if not ply_path.is_file():
        gates.append("WARN  out.ply not found")
    else:
        size_mb = ply_path.stat().st_size / 1e6
        metrics["ply_size_mb"] = round(size_mb, 1)
        gates.append(f"PASS  out.ply {size_mb:.0f} MB")

    if any(g.startswith("FAIL") for g in gates):
        overall = "FAIL"
    elif any(g.startswith("WARN") for g in gates):
        overall = "WARN"
    else:
        overall = "PASS"
    metrics["overall"] = overall

    lines = ["Phase 1 (Stella) Validation", f"  phase1_dir: {args.phase1_dir}", ""] + gates + [f"\nOVERALL: {overall}"]
    print("\n".join(lines))
    (args.out_dir / "summary.txt").write_text("\n".join(lines))
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return 0 if overall != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
