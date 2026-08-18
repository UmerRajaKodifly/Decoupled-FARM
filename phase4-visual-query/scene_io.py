"""Read/write caption + embedding fields on scene_state dicts."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# FARM structured caption parser
_REPO = Path(__file__).resolve().parent.parent
_FARM_SRC = _REPO / "farm_src" / "src"
if str(_FARM_SRC) not in sys.path:
    sys.path.insert(0, str(_FARM_SRC))

from scene_graph.captioning.structured import StructuredCaption, parse_structured_caption  # noqa: E402

_OBJ_CROP_RE = re.compile(r"obj_(\d+)_o\d+\.jpg$", re.I)

# text-embedding-004 is 768-d; leftover mock vectors were 64-d.
REAL_EMBED_MIN_DIM = 128


def object_count(scene_state: dict) -> int:
    means = scene_state.get("means")
    if isinstance(means, torch.Tensor):
        return int(means.shape[0])
    return 0


def is_active(scene_state: dict, idx: int) -> bool:
    active = scene_state.get("active")
    if isinstance(active, torch.Tensor) and 0 <= idx < active.shape[0]:
        return bool(active[idx].item())
    return True


def _ensure_list(key: str, scene_state: dict, n: int, default: Any) -> list:
    lst = scene_state.get(key)
    if not isinstance(lst, list):
        lst = []
    while len(lst) < n:
        lst.append(default() if callable(default) else default)
    scene_state[key] = lst
    return lst


def ensure_caption_fields(scene_state: dict) -> None:
    n = object_count(scene_state)
    _ensure_list("object_caption", scene_state, n, "")
    _ensure_list("object_category", scene_state, n, "")
    _ensure_list("object_supercategory", scene_state, n, "")
    _ensure_list("object_key_attributes", scene_state, n, list)
    _ensure_list("object_caption_decision", scene_state, n, "")
    _ensure_list("object_caption_embedding", scene_state, n, list)
    _ensure_list("object_category_candidates", scene_state, n, list)


@dataclass
class FaceView:
    """One full perspective face plus the object's pixel bbox on that image."""

    rgb_path: Path
    bbox_xyxy: Tuple[float, float, float, float]
    image_width: int
    image_height: int


def infer_run_dir(scene_state_path: Optional[Path] = None) -> Optional[Path]:
    if scene_state_path is None:
        return None
    p = scene_state_path.resolve()
    # …/run_XXX/phase4/scene_state_*.pt → run_XXX
    if p.parent.name == "phase4":
        return p.parent.parent
    return p.parent


def infer_faces_dir(
    scene_state: dict,
    scene_state_path: Optional[Path] = None,
) -> Optional[Path]:
    explicit = scene_state.get("_faces_dir")
    if explicit:
        p = Path(str(explicit))
        if p.is_dir():
            return p
    run_dir = infer_run_dir(scene_state_path)
    if run_dir is not None:
        cand = run_dir / "phase1.5" / "faces"
        if cand.is_dir():
            return cand
    if scene_state_path is not None:
        cand = scene_state_path.parent.parent / "phase1.5" / "faces"
        if cand.is_dir():
            return cand
    return None


def infer_crops_dir(scene_state: dict, scene_state_path: Optional[Path] = None) -> Optional[Path]:
    """Best-effort crops/ directory (handles Docker /phase4/crops paths)."""
    explicit = scene_state.get("_crops_dir")
    if explicit:
        p = Path(str(explicit))
        if p.is_dir():
            return p
    if scene_state_path is not None:
        cand = scene_state_path.parent / "crops"
        if cand.is_dir():
            return cand
    return None


def _resolve_crop_path(raw: str, *, crops_dir: Optional[Path]) -> Optional[Path]:
    if not raw:
        return None
    p = Path(raw)
    if p.is_file():
        return p
    name = p.name
    # Docker container path → host crops dir
    if crops_dir and name.endswith(".jpg"):
        cand = crops_dir / name
        if cand.is_file():
            return cand
    if crops_dir and raw.startswith("/phase4/crops/"):
        cand = crops_dir / name
        if cand.is_file():
            return cand
    return None


def _resolve_face_path(raw: str, *, faces_dir: Optional[Path]) -> Optional[Path]:
    if not raw:
        return None
    p = Path(raw)
    if p.is_file():
        return p
    name = p.name
    if faces_dir and name.endswith((".jpg", ".jpeg", ".png")):
        cand = faces_dir / name
        if cand.is_file():
            return cand
    if faces_dir and ("/phase1.5/faces/" in raw.replace("\\", "/") or raw.startswith("/phase1.5/")):
        cand = faces_dir / name
        if cand.is_file():
            return cand
    return None


def _bbox_for_object(scene_state: dict, idx: int) -> Optional[Tuple[float, float, float, float]]:
    bbox_t = scene_state.get("best_view_bbox_xyxy")
    if isinstance(bbox_t, torch.Tensor) and bbox_t.ndim == 2 and idx < bbox_t.shape[0]:
        row = bbox_t[idx].tolist()
        if len(row) == 4 and float(row[2]) > float(row[0]) and float(row[3]) > float(row[1]):
            return (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
    obs = scene_state.get("rgb_observations") or []
    if idx < len(obs) and obs[idx]:
        rec = obs[idx][0] if isinstance(obs[idx], list) and obs[idx] else obs[idx]
        if isinstance(rec, dict):
            bb = rec.get("bbox") or rec.get("bbox_xyxy")
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                return (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
    return None


def _image_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = int(im.size[0]), int(im.size[1])
            if w > 0 and h > 0:
                return w, h
    except Exception:
        return None
    return None


def face_view_for_object(
    scene_state: dict,
    idx: int,
    *,
    faces_dir: Optional[Path] = None,
    scene_state_path: Optional[Path] = None,
) -> Optional[FaceView]:
    """Resolve the full perspective face + bbox used for captioning."""
    faces_dir = faces_dir or infer_faces_dir(scene_state, scene_state_path)
    bbox = _bbox_for_object(scene_state, idx)

    rgb_paths = scene_state.get("best_view_rgb_path") or []
    if idx < len(rgb_paths) and rgb_paths[idx]:
        resolved = _resolve_face_path(str(rgb_paths[idx]), faces_dir=faces_dir)
        if resolved is not None and bbox is not None:
            size = _image_size(resolved)
            if size is None:
                return None
            w, h = size
            return FaceView(resolved, bbox, w, h)

    obs = scene_state.get("rgb_observations") or []
    if idx < len(obs) and obs[idx]:
        rec = obs[idx][0] if isinstance(obs[idx], list) and obs[idx] else obs[idx]
        if isinstance(rec, dict):
            for k in ("full_rgb_path", "rgb_path"):
                if rec.get(k):
                    resolved = _resolve_face_path(str(rec[k]), faces_dir=faces_dir)
                    if resolved is not None:
                        bb = bbox
                        if bb is None:
                            raw_bb = rec.get("bbox") or rec.get("bbox_xyxy")
                            if isinstance(raw_bb, (list, tuple)) and len(raw_bb) == 4:
                                bb = (float(raw_bb[0]), float(raw_bb[1]), float(raw_bb[2]), float(raw_bb[3]))
                        if bb is not None:
                            size = _image_size(resolved)
                            if size is None:
                                return None
                            w, h = size
                            return FaceView(resolved, bb, w, h)
    return None


def crop_path_for_object(
    scene_state: dict,
    idx: int,
    *,
    crops_dir: Optional[Path] = None,
) -> Optional[Path]:
    crops_dir = crops_dir or infer_crops_dir(scene_state)

    paths = scene_state.get("best_view_crop_path") or []
    if idx < len(paths) and paths[idx]:
        resolved = _resolve_crop_path(str(paths[idx]), crops_dir=crops_dir)
        if resolved:
            return resolved

    # rgb_observations fallback
    obs = scene_state.get("rgb_observations") or []
    if idx < len(obs) and obs[idx]:
        rec = obs[idx][0] if isinstance(obs[idx], list) and obs[idx] else obs[idx]
        if isinstance(rec, dict):
            for k in ("storage_path", "source_ref"):
                if rec.get(k):
                    resolved = _resolve_crop_path(str(rec[k]), crops_dir=crops_dir)
                    if resolved:
                        return resolved

    # Glob by object index in crops dir
    if crops_dir and crops_dir.is_dir():
        for pattern in (f"obj_{idx:06d}_*.jpg", f"obj_{idx:06d}_*.jpeg", f"obj_{idx:06d}_*.png"):
            hits = sorted(crops_dir.glob(pattern))
            if hits:
                return hits[0]
    return None


_CAPTION_OVERLAY_KEYS = (
    "object_caption",
    "object_category",
    "object_supercategory",
    "object_key_attributes",
    "object_caption_decision",
    "object_category_candidates",
)


def caption_checkpoint_is_mock(scene_state: dict) -> bool:
    """True if a checkpoint looks like MOCK=1 output (must not resume into a paid run)."""
    for cap in scene_state.get("object_caption") or []:
        if isinstance(cap, str) and "mock caption" in cap.lower():
            return True
    for attrs in scene_state.get("object_key_attributes") or []:
        if isinstance(attrs, list) and any(str(a).lower() == "mock" for a in attrs):
            return True
    return False


def overlay_captions(dst: dict, src: dict) -> int:
    """Copy keep/drop captions from a checkpoint onto dst. Returns how many objects were filled."""
    ensure_caption_fields(dst)
    ensure_caption_fields(src)
    n = min(object_count(dst), object_count(src))
    copied = 0
    for i in range(n):
        decision = str((src.get("object_caption_decision") or [""])[i] or "")
        if decision not in {"keep", "drop"}:
            continue
        for key in _CAPTION_OVERLAY_KEYS:
            dst_list = dst.get(key)
            src_list = src.get(key)
            if isinstance(dst_list, list) and isinstance(src_list, list) and i < len(src_list):
                val = src_list[i]
                dst_list[i] = list(val) if isinstance(val, list) else val
        copied += 1
    return copied


def overlay_embeddings(dst: dict, src: dict, *, min_dim: int = REAL_EMBED_MIN_DIM) -> int:
    """Copy real caption embeddings from a checkpoint. Skips short leftover mock vectors."""
    ensure_caption_fields(dst)
    ensure_caption_fields(src)
    n = min(object_count(dst), object_count(src))
    copied = 0
    src_emb = src.get("object_caption_embedding") or []
    dst_emb = dst["object_caption_embedding"]
    for i in range(n):
        vec = src_emb[i] if i < len(src_emb) else None
        if not isinstance(vec, list) or len(vec) < min_dim:
            continue
        dst_emb[i] = [float(x) for x in vec]
        copied += 1
    return copied


def write_caption(
    scene_state: dict,
    idx: int,
    parsed: StructuredCaption,
    *,
    embedding: Optional[List[float]] = None,
) -> None:
    ensure_caption_fields(scene_state)
    n = object_count(scene_state)
    if idx < 0 or idx >= n:
        return

    decision = parsed.decision if parsed.decision in {"keep", "drop"} else None
    if decision is None:
        return

    scene_state["object_caption_decision"][idx] = decision

    if decision == "drop" or not parsed.is_clear_object:
        scene_state["object_caption"][idx] = ""
        scene_state["object_category"][idx] = ""
        scene_state["object_supercategory"][idx] = ""
        scene_state["object_key_attributes"][idx] = []
        if embedding is not None:
            scene_state["object_caption_embedding"][idx] = []
        return

    desc = parsed.description or parsed.category or ""
    scene_state["object_caption"][idx] = desc
    scene_state["object_category"][idx] = parsed.category or ""
    scene_state["object_supercategory"][idx] = parsed.supercategory or ""
    scene_state["object_key_attributes"][idx] = list(parsed.attributes or [])
    if parsed.category:
        scene_state["object_category_candidates"][idx] = [parsed.category]

    if embedding is not None:
        scene_state["object_caption_embedding"][idx] = embedding


def caption_summary(scene_state: dict) -> Dict[str, int]:
    ensure_caption_fields(scene_state)
    n = object_count(scene_state)
    kept = dropped = empty = 0
    for i in range(n):
        if not is_active(scene_state, i):
            continue
        d = str(scene_state["object_caption_decision"][i] or "")
        if d == "keep":
            kept += 1
        elif d == "drop":
            dropped += 1
        else:
            empty += 1
    return {"n_total": n, "n_keep": kept, "n_drop": dropped, "n_uncaptioned": empty}
