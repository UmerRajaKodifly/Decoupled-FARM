"""Repo-relative locations. No machine-specific sibling checkouts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = REPO_ROOT / "third_party"
DATA_DIR = REPO_ROOT / "data"

DEFAULT_VOCAB_JSON = DATA_DIR / "construction_site_object_vocabulary.json"


def _existing_dir(*candidates: Path | None) -> Path | None:
    for raw in candidates:
        if raw is None:
            continue
        path = Path(raw).expanduser()
        if path.is_dir():
            return path.resolve()
    return None


def farm_project_root() -> Path:
    env = os.environ.get("FARM_PROJECT_ROOT", "").strip()
    src_env = os.environ.get("FARM_PROJECT_SRC", "").strip()
    from_src = Path(src_env).expanduser().parent if src_env else None
    found = _existing_dir(
        Path(env) if env else None,
        from_src,
        THIRD_PARTY / "FARM-Project",
    )
    if found is None:
        raise FileNotFoundError(
            "FARM-Project not found. Clone into third_party/FARM-Project "
            "(see third_party/README.md) or set FARM_PROJECT_ROOT."
        )
    return found


def farm_src() -> Path:
    root = farm_project_root()
    src = root / "src"
    return src if src.is_dir() else root


def ss3dgs_root() -> Path:
    env = os.environ.get("SS3DGS_ROOT", "").strip()
    found = _existing_dir(Path(env) if env else None, THIRD_PARTY / "ss-3dgs")
    if found is None:
        raise FileNotFoundError(
            "ss-3dgs not found. Clone into third_party/ss-3dgs "
            "(see third_party/README.md) or set SS3DGS_ROOT."
        )
    return found


def scene_graph_model_dir() -> Path:
    env = os.environ.get("SCENE_GRAPH_MODEL_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return farm_project_root() / "models"


def ensure_farm_on_path() -> Path:
    src = farm_src()
    os.environ.setdefault("SCENE_GRAPH_MODEL_DIR", str(scene_graph_model_dir()))
    inserted = str(src)
    if inserted not in sys.path:
        sys.path.insert(0, inserted)
    return src


def ensure_ss3dgs_on_path() -> Path:
    root = ss3dgs_root()
    inserted = str(root)
    if inserted not in sys.path:
        sys.path.insert(0, inserted)
    return root


def dl_depth_search_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("DL_DEPTH_ROOT", "FARM_DL_DEPTH_ROOT"):
        extra = os.environ.get(key, "").strip()
        if extra:
            roots.append(Path(extra).expanduser())
    roots.extend(
        [
            REPO_ROOT / "third_party" / "dl_depth_v1",
            REPO_ROOT / "models",
            REPO_ROOT / "work",
        ]
    )
    seen: set[Path] = set()
    out: list[Path] = []
    for raw in roots:
        path = Path(raw).expanduser()
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not path.is_dir():
            continue
        seen.add(resolved)
        out.append(path)
    return out
