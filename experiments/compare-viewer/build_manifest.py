#!/usr/bin/env python3
"""Build A/B comparison manifest from baseline + experiment run directories."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load_vocab(path: Optional[Path]) -> List[str]:
    if path is None or not path.is_file():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _class_name(cid: int, vocab: List[str]) -> str:
    return vocab[cid] if 0 <= cid < len(vocab) else f"class_{cid}"


def _read_summary(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    out: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _load_objects_from_viewer(viewer_dir: Path) -> List[Dict[str, Any]]:
    obj_path = viewer_dir / "objects.json"
    if obj_path.is_file():
        return json.loads(obj_path.read_text(encoding="utf-8"))
    return []


def _load_objects_from_scene(scene_path: Path, vocab: List[str]) -> List[Dict[str, Any]]:
    import torch

    ss = torch.load(scene_path, map_location="cpu", weights_only=False)
    means = ss["means"].numpy()
    active = ss.get("active")
    act = active.numpy().astype(bool) if isinstance(active, torch.Tensor) else np.ones(len(means), bool)
    cids = ss.get("class_ids")
    cid_arr = cids.numpy() if isinstance(cids, torch.Tensor) else np.full(len(means), -1)
    objs: List[Dict[str, Any]] = []
    for i, ai in enumerate(np.where(act)[0]):
        m = means[ai]
        cid = int(cid_arr[ai])
        objs.append({
            "id": int(ai),
            "index": i,
            "label": _class_name(cid, vocab),
            "class_id": cid,
            "mean": [float(m[0]), float(m[1]), float(m[2])],
        })
    return objs


def _resolve_viewer_dir(run_dir: Path) -> Path:
    cand = run_dir / "validation" / "3d-viewer"
    return cand if cand.is_dir() else run_dir / "validation" / "3d-viewer"


def _spatial_match(
    a_objs: List[Dict[str, Any]],
    b_objs: List[Dict[str, Any]],
    *,
    max_dist: float = 2.0,
) -> Tuple[List[Dict], List[int], List[int]]:
    """Greedy nearest-neighbour match by 3D mean position."""
    used_b: set[int] = set()
    pairs: List[Dict] = []
    unmatched_a: List[int] = []
    b_means = np.array([o["mean"] for o in b_objs], dtype=np.float64) if b_objs else np.zeros((0, 3))

    for ai, ao in enumerate(a_objs):
        am = np.array(ao["mean"], dtype=np.float64)
        best_j, best_d = -1, float("inf")
        for bj, bo in enumerate(b_objs):
            if bj in used_b:
                continue
            d = float(np.linalg.norm(am - b_means[bj]))
            if d < best_d and d <= max_dist:
                best_d, best_j = d, bj
        if best_j < 0:
            unmatched_a.append(ai)
            continue
        used_b.add(best_j)
        bo = b_objs[best_j]
        pairs.append({
            "dist_m": round(best_d, 3),
            "a_index": ai,
            "b_index": best_j,
            "a_id": ao.get("id"),
            "b_id": bo.get("id"),
            "a_label": ao.get("label", ""),
            "b_label": bo.get("label", ""),
            "label_changed": ao.get("label") != bo.get("label"),
        })

    unmatched_b = [j for j in range(len(b_objs)) if j not in used_b]
    return pairs, unmatched_a, unmatched_b


def _class_histogram(objs: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(o.get("label", "?") for o in objs))


def build_manifest(
    baseline_dir: Path,
    experiment_dir: Path,
    *,
    baseline_label: str = "Baseline",
    experiment_label: str = "Phase A",
    match_dist_m: float = 2.0,
) -> Dict[str, Any]:
    baseline_dir = baseline_dir.resolve()
    experiment_dir = experiment_dir.resolve()

    a_viewer = _resolve_viewer_dir(baseline_dir)
    b_viewer = _resolve_viewer_dir(experiment_dir)

    vocab_a = _load_vocab(baseline_dir / ".." / ".." / "vocab" / "construction_vocab.txt")
    vocab_b_path = _REPO / "vocab" / "construction_vocab_concise.txt"
    vocab_b = _load_vocab(vocab_b_path)

    a_objs = _load_objects_from_viewer(a_viewer)
    if not a_objs:
        scene = baseline_dir / "phase3.5" / "scene_state_stella.pt"
        if not scene.is_file():
            scene = baseline_dir / "phase3" / "scene_state.pt"
        a_objs = _load_objects_from_scene(scene, vocab_a or _load_vocab(_REPO / "vocab" / "construction_vocab.txt"))

    b_objs = _load_objects_from_viewer(b_viewer)
    if not b_objs:
        scene = experiment_dir / "phase3.5" / "scene_state_stella.pt"
        if not scene.is_file():
            scene = experiment_dir / "phase3" / "scene_state.pt"
        b_objs = _load_objects_from_scene(scene, vocab_b or _load_vocab(_REPO / "vocab" / "construction_vocab_concise.txt"))

    pairs, unmatched_a, unmatched_b = _spatial_match(a_objs, b_objs, max_dist=match_dist_m)
    label_changes = [p for p in pairs if p["label_changed"]]

    manifest: Dict[str, Any] = {
        "baseline": {
            "run_id": baseline_dir.name,
            "label": baseline_label,
            "viewer_dir": str(a_viewer),
            "n_objects": len(a_objs),
            "class_histogram": _class_histogram(a_objs),
            "phase35_summary": _read_summary(baseline_dir / "validation" / "phase3.5" / "summary.txt"),
            "phase4_summary": _read_summary(baseline_dir / "validation" / "phase4" / "summary.txt"),
        },
        "experiment": {
            "run_id": experiment_dir.name,
            "label": experiment_label,
            "viewer_dir": str(b_viewer),
            "n_objects": len(b_objs),
            "class_histogram": _class_histogram(b_objs),
            "phase35_summary": _read_summary(experiment_dir / "validation" / "phase3.5" / "summary.txt"),
            "phase4_summary": _read_summary(experiment_dir / "validation" / "phase4" / "summary.txt"),
        },
        "diff": {
            "matched_pairs": len(pairs),
            "label_changes": len(label_changes),
            "only_baseline": len(unmatched_a),
            "only_experiment": len(unmatched_b),
            "pairs": pairs,
            "label_change_samples": label_changes[:50],
            "unmatched_baseline_indices": unmatched_a[:30],
            "unmatched_experiment_indices": unmatched_b[:30],
        },
        "match_dist_m": match_dist_m,
    }
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description="Build A/B comparison manifest")
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--experiment-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline-label", default="Baseline (48-class)")
    p.add_argument("--experiment-label", default="Experiment")
    p.add_argument("--match-dist", type=float, default=2.0)
    args = p.parse_args()

    manifest = build_manifest(
        args.baseline_dir,
        args.experiment_dir,
        baseline_label=args.baseline_label,
        experiment_label=args.experiment_label,
        match_dist_m=args.match_dist,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    d = manifest["diff"]
    print(f"[compare] baseline objects: {manifest['baseline']['n_objects']}")
    print(f"[compare] experiment objects: {manifest['experiment']['n_objects']}")
    print(f"[compare] matched: {d['matched_pairs']}  label changes: {d['label_changes']}")
    print(f"[compare] only baseline: {d['only_baseline']}  only experiment: {d['only_experiment']}")
    print(f"[compare] manifest → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
