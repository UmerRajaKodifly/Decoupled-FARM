"""Ablation experiment registry for compare viewer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[2]
ABLATION_DIR = _REPO / "outputs" / "ablation"
INDEX_PATH = ABLATION_DIR / "index.json"
MANIFEST_DIR = ABLATION_DIR / "manifests"


def ablation_id(conf: float, vote: float, margin: float) -> str:
    return f"c{int(round(conf * 100)):03d}_v{int(round(vote * 100)):03d}_m{int(round(margin * 100)):03d}"


def ablation_label(conf: float, vote: float, margin: float) -> str:
    return f"conf={conf:.2f} vote={vote:.2f} margin={margin:.2f}"


def _default_index() -> Dict[str, Any]:
    baseline = ""
    bm = _REPO / "outputs" / "baselines" / "manifest.json"
    if bm.is_file():
        baseline = json.loads(bm.read_text()).get("baseline_run_id", "")
    return {
        "baseline_run_id": baseline,
        "vocab": "construction_vocab.txt",
        "updated_at": "",
        "experiments": [],
    }


def load_index() -> Dict[str, Any]:
    if INDEX_PATH.is_file():
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    return _default_index()


def save_index(index: Dict[str, Any]) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    index["updated_at"] = datetime.now(timezone.utc).isoformat()
    INDEX_PATH.write_text(json.dumps(index, indent=2), encoding="utf-8")


def register_experiment(
    *,
    conf: float,
    vote: float,
    margin: float,
    run_id: str,
    manifest_path: Path,
    baseline_run_id: str,
    vocab: str = "construction_vocab.txt",
    note: str = "",
) -> Dict[str, Any]:
    """Add or update an experiment entry in the ablation index."""
    index = load_index()
    if baseline_run_id:
        index["baseline_run_id"] = baseline_run_id
    index["vocab"] = vocab

    aid = ablation_id(conf, vote, margin)
    entry = {
        "id": aid,
        "label": ablation_label(conf, vote, margin),
        "params": {
            "yoloe_conf": conf,
            "label_min_score": vote,
            "label_margin": margin,
            "vocab": vocab,
        },
        "run_id": run_id,
        "manifest": str(manifest_path.relative_to(_REPO)),
        "note": note,
    }

    exps: List[Dict[str, Any]] = [
        e for e in index.get("experiments", []) if e.get("id") != aid
    ]
    exps.append(entry)
    exps.sort(key=lambda e: e["id"])
    index["experiments"] = exps
    save_index(index)
    return entry


def get_experiment(exp_id: str) -> Optional[Dict[str, Any]]:
    index = load_index()
    for e in index.get("experiments", []):
        if e.get("id") == exp_id:
            return e
    return None


def load_manifest_for_experiment(exp_id: str) -> Optional[Dict[str, Any]]:
    entry = get_experiment(exp_id)
    if not entry:
        return None
    path = _REPO / entry["manifest"]
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
