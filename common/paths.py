"""Resolve farm_src / models for repo layout and containers.

Layout expectations
-------------------
repo/
  farm_src/          # src/scene_graph + third_party/yoloe
  models/            # yoloe, mobileclip, dinov3-vits16, da3, orb_vocab
  phase2/  phase3/

Environment overrides (container-friendly)
------------------------------------------
  FARM_ROOT                 path to farm_src (contains src/ + third_party/)
  SCENE_GRAPH_MODEL_DIR     path to models directory
  MOBILECLIP_CHECKPOINT / MOBILECLIP_BLT_CKPT / MOBILECLIP_WEIGHTS_DIR
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def resolve_farm_root(caller_file: str | Path | None = None) -> Path:
    env = os.environ.get("FARM_ROOT", "").strip()
    if env:
        p = Path(env).resolve()
        if (p / "src" / "scene_graph").is_dir() or (p / "scene_graph").is_dir():
            return p

    here = Path(caller_file or __file__).resolve()
    # repo/phase2/foo.py → repo/farm_src
    # repo/common/paths.py → repo/farm_src
    for parent in [here.parent, *here.parents]:
        candidate = parent / "farm_src"
        if (candidate / "src" / "scene_graph").is_dir():
            return candidate.resolve()
        # less-likely: parent IS farm_src
        if (parent / "src" / "scene_graph").is_dir() and parent.name == "farm_src":
            return parent.resolve()

    # Legacy layout: farm-git/FARM-Project next to pipeline/
    for parent in here.parents:
        legacy = parent / "FARM-Project"
        if (legacy / "src" / "scene_graph").is_dir():
            return legacy.resolve()

    raise FileNotFoundError(
        "Could not locate farm_src (need src/scene_graph). "
        "Set FARM_ROOT or place farm_src next to phase2/phase3."
    )


def farm_src_python_path(farm_root: Path | None = None) -> Path:
    """Directory that must be on sys.path for `import scene_graph`."""
    root = farm_root or resolve_farm_root()
    if (root / "src" / "scene_graph").is_dir():
        return root / "src"
    if (root / "scene_graph").is_dir():
        return root
    raise FileNotFoundError(f"No scene_graph package under {root}")


def yoloe_python_path(farm_root: Path | None = None) -> Path:
    root = farm_root or resolve_farm_root()
    yoloe = root / "third_party" / "yoloe"
    if not yoloe.is_dir():
        raise FileNotFoundError(f"YOLOE missing at {yoloe}")
    return yoloe


def resolve_models_dir(farm_root: Path | None = None) -> Path:
    env = os.environ.get("SCENE_GRAPH_MODEL_DIR", "").strip()
    if env and Path(env).is_dir():
        return Path(env).resolve()

    root = farm_root or resolve_farm_root()
    # Preferred: repo/models (sibling of farm_src)
    repo_models = root.parent / "models"
    if repo_models.is_dir():
        return repo_models.resolve()
    if (root / "models").is_dir():
        return (root / "models").resolve()
    return repo_models


def ensure_sys_path(caller_file: str | Path | None = None) -> Path:
    """Insert farm src + yoloe onto sys.path; return farm_root."""
    root = resolve_farm_root(caller_file)
    for p in (str(farm_src_python_path(root)), str(yoloe_python_path(root))):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


def apply_model_env(models_dir: Path | None = None) -> Path:
    """Set SCENE_GRAPH_MODEL_DIR and MobileCLIP env vars (setdefault)."""
    models = models_dir or resolve_models_dir()
    os.environ.setdefault("SCENE_GRAPH_MODEL_DIR", str(models))
    mclip = models / "mobileclip" / "mobileclip_blt.pt"
    os.environ.setdefault("MOBILECLIP_CHECKPOINT", str(mclip))
    os.environ.setdefault("MOBILECLIP_BLT_CKPT", str(mclip))
    os.environ.setdefault("MOBILECLIP_WEIGHTS_DIR", str(models / "mobileclip"))
    return models
