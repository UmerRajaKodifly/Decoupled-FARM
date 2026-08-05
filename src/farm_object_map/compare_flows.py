"""Same-object comparison between two depth-source mapping outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load_objects(summary_path: Path) -> list[dict]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return list(payload.get("objects") or [])


def compare_object_summaries(
    dl_summary: str | Path,
    mvs_summary: str | Path,
    *,
    max_mean_dist: float = 2.0,
) -> dict:
    """Match objects by label + nearest Gaussian mean (SfM/metric units as stored)."""
    dl_objs = _load_objects(Path(dl_summary))
    mvs_objs = _load_objects(Path(mvs_summary))
    used_mvs: set[int] = set()
    matches: list[dict] = []
    unmatched_dl: list[dict] = []

    for dl in dl_objs:
        label = dl.get("label")
        mean_dl = np.asarray(dl.get("mean"), dtype=np.float64)
        best = None
        best_dist = None
        for j, mvs in enumerate(mvs_objs):
            if j in used_mvs or mvs.get("label") != label:
                continue
            mean_mvs = np.asarray(mvs.get("mean"), dtype=np.float64)
            if mean_dl.shape != mean_mvs.shape:
                continue
            dist = float(np.linalg.norm(mean_dl - mean_mvs))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = (j, mvs)
        if best is None or best_dist is None or best_dist > max_mean_dist:
            unmatched_dl.append({"object_id": dl.get("object_id"), "label": label, "mean": dl.get("mean")})
            continue
        j, mvs = best
        used_mvs.add(j)
        cov_dl = np.asarray(dl.get("cov"), dtype=np.float64) if dl.get("cov") is not None else None
        cov_mvs = np.asarray(mvs.get("cov"), dtype=np.float64) if mvs.get("cov") is not None else None
        cov_frob = None
        if cov_dl is not None and cov_mvs is not None and cov_dl.shape == cov_mvs.shape:
            cov_frob = float(np.linalg.norm(cov_dl - cov_mvs))
        matches.append(
            {
                "label": label,
                "dl_object_id": dl.get("object_id"),
                "mvs_object_id": mvs.get("object_id"),
                "mean_l2": best_dist,
                "cov_frobenius": cov_frob,
                "dl_mean": dl.get("mean"),
                "mvs_mean": mvs.get("mean"),
                "large_discrepancy": bool(
                    best_dist > max(0.5, 0.25 * max_mean_dist) or (cov_frob is not None and cov_frob > 1.0)
                ),
            }
        )

    unmatched_mvs = [
        {"object_id": mvs.get("object_id"), "label": mvs.get("label"), "mean": mvs.get("mean")}
        for j, mvs in enumerate(mvs_objs)
        if j not in used_mvs
    ]
    return {
        "n_dl": len(dl_objs),
        "n_mvs": len(mvs_objs),
        "n_matched": len(matches),
        "matches": matches,
        "unmatched_dl": unmatched_dl,
        "unmatched_mvs": unmatched_mvs,
        "large_discrepancies": [m for m in matches if m["large_discrepancy"]],
    }
