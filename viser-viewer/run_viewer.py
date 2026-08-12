#!/usr/bin/env python3
"""Offline Viser viewer for construction-pipeline Phase 2 / Phase 3 outputs.

Wires FARM's ``PipelineViserVisualizer`` to your packs and ``scene_state.pt``
without running the full FARM offline mapper.

Examples
--------
# Phase 3 map (boxes + optional Gaussian ellipsoids + voxels)
python run_viewer.py \\
  --scene-state ../phase3-associate-fuse-map/output/scene_state.pt \\
  --vocab ../phase2-detect-segment-embed/vocab/construction_vocab.txt

# Map + trajectory / sparse det points from Phase 2 packs
python run_viewer.py \\
  --scene-state ../phase3-associate-fuse-map/output/scene_state.pt \\
  --det-dir ../phase2-detect-segment-embed/output \\
  --show-detections --bg-from-det-points

# Phase 2 proposals only (no fused map)
python run_viewer.py \\
  --det-dir ../phase2-detect-segment-embed/output \\
  --mode detections
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# FARM path + viser
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parent.parent  # repo/
# Resolve FARM src: prefer repo/farm_src, fall back to adjacent FARM-Project
_FARM_SRC = _REPO_ROOT / "farm_src" / "src"
if not _FARM_SRC.is_dir():
    _FARM_SRC = _REPO_ROOT.parent / "FARM-Project" / "src"
if _FARM_SRC.is_dir() and str(_FARM_SRC) not in sys.path:
    sys.path.insert(0, str(_FARM_SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("viser_viewer")


def _load_vocab(path: Optional[Path]) -> List[str]:
    if path is None or not path.is_file():
        return []
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]


def _class_name(cid: int, vocab: Sequence[str]) -> str:
    if 0 <= int(cid) < len(vocab):
        return str(vocab[int(cid)])
    if int(cid) < 0:
        return "unknown"
    return f"class_{int(cid)}"


def _enrich_labels(scene_state: dict, vocab: Sequence[str]) -> None:
    """Fill empty FARM caption fields from YOLOE class_ids (+ optional votes)."""
    class_ids = scene_state.get("class_ids")
    if not isinstance(class_ids, torch.Tensor) or class_ids.numel() == 0:
        return
    n = int(scene_state["means"].shape[0]) if isinstance(scene_state.get("means"), torch.Tensor) else 0
    if n == 0:
        return

    caps = list(scene_state.get("object_caption") or [""] * n)
    cats = list(scene_state.get("object_category") or [""] * n)
    vote_mass = scene_state.get("class_vote_mass")
    while len(caps) < n:
        caps.append("")
    while len(cats) < n:
        cats.append("")

    for i in range(min(n, int(class_ids.numel()))):
        cid = int(class_ids[i].item())
        name = _class_name(cid, vocab)
        extra = ""
        if isinstance(vote_mass, list) and i < len(vote_mass) and isinstance(vote_mass[i], dict) and vote_mass[i]:
            ranked = sorted(vote_mass[i].items(), key=lambda kv: float(kv[1]), reverse=True)[:3]
            parts = [f"{_class_name(int(c), vocab)}:{float(w):.2f}" for c, w in ranked]
            extra = " | votes: " + ", ".join(parts)
        label = f"{name} (id={cid}){extra}"
        if not str(caps[i]).strip():
            caps[i] = label
        if not str(cats[i]).strip():
            cats[i] = name

    scene_state["object_caption"] = caps
    scene_state["object_category"] = cats


def _sorted_packs(det_dir: Path) -> List[Path]:
    paths = sorted(det_dir.glob("detections_kf*.pt"))
    if not paths:
        raise FileNotFoundError(f"No detections_kf*.pt in {det_dir}")
    return paths


def _collect_poses(packs: Sequence[dict]) -> List[torch.Tensor]:
    poses: List[torch.Tensor] = []
    for pack in packs:
        pw = pack.get("poses_world") or []
        for p in pw:
            if isinstance(p, torch.Tensor) and p.shape == (4, 4):
                poses.append(p.detach().cpu().float())
            else:
                arr = np.asarray(p, dtype=np.float32)
                if arr.shape == (4, 4):
                    poses.append(torch.from_numpy(arr))
    return poses


def _subsample_det_points(
    packs: Sequence[dict],
    *,
    max_points: int = 250_000,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build a sparse world RGB-ish cloud from Phase 2 det_points_flat."""
    chunks: List[np.ndarray] = []
    for pack in packs:
        flat = pack.get("det_points_flat")
        if not isinstance(flat, torch.Tensor) or flat.numel() == 0:
            continue
        pts = flat.detach().cpu().numpy().astype(np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            continue
        chunks.append(pts)
    if not chunks:
        return None, None
    all_pts = np.concatenate(chunks, axis=0)
    n = all_pts.shape[0]
    if n > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_points, replace=False)
        all_pts = all_pts[idx]
    # Soft height coloring (Z) so the cloud is readable without RGB frames
    z = all_pts[:, 2]
    z0, z1 = float(np.percentile(z, 5)), float(np.percentile(z, 95))
    t = np.clip((z - z0) / max(z1 - z0, 1e-3), 0.0, 1.0)
    colors = np.stack(
        [
            (40 + 200 * t),
            (80 + 140 * (1.0 - t)),
            (160 + 80 * (1.0 - t)),
        ],
        axis=1,
    ).astype(np.uint8)
    return all_pts, colors


def _resolve_path(path_str: str) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        # Try cwd and repo pipeline root
        for base in (Path.cwd(), _HERE.parent.parent, _HERE):
            cand = (base / p).resolve()
            if cand.is_file():
                return cand
        p = p.resolve()
    return p if p.is_file() else None


def _load_rgb_uint8(path: Path, *, max_side: int = 320) -> Optional[np.ndarray]:
    try:
        from PIL import Image
    except ImportError:
        return None
    if not path.is_file():
        return None
    try:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"))
    except Exception:
        return None
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return None
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest > max_side > 0:
        scale = max_side / float(longest)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        # Pillow < 9 compatibility
        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR  # type: ignore[attr-defined]
        arr = np.asarray(Image.fromarray(arr).resize((nw, nh), resample))
    return arr.astype(np.uint8, copy=False)


def _hydrate_object_images(
    scene_state: dict,
    *,
    crops_dir: Optional[Path] = None,
    max_side: int = 320,
) -> int:
    """Load best-view crop JPEGs into ``rgb_observations`` as numpy arrays.

    FARM's Viser panel only shows ``obs["image"]`` as an array (or live tensors).
    Paths alone (`storage_path` / ``best_view_crop_path``) are **not** decoded on click.
    """
    means = scene_state.get("means")
    n = int(means.shape[0]) if isinstance(means, torch.Tensor) else 0
    if n == 0:
        return 0

    object_ids = scene_state.get("object_id")
    crop_paths = scene_state.get("best_view_crop_path")
    if not isinstance(crop_paths, list):
        crop_paths = [""] * n
    while len(crop_paths) < n:
        crop_paths.append("")

    rgb_obs = scene_state.get("rgb_observations")
    if not isinstance(rgb_obs, list):
        rgb_obs = []
    while len(rgb_obs) < n:
        rgb_obs.append([])

    # optional directory of crops named obj_{object_id:06d}_o{idx:04d}.jpg
    by_oid: Dict[int, Path] = {}
    by_oindex: Dict[int, Path] = {}
    if crops_dir is not None and crops_dir.is_dir():
        for p in crops_dir.glob("obj_*.jpg"):
            # obj_000012_o0014.jpg
            name = p.stem
            parts = name.split("_")
            try:
                if len(parts) >= 3 and parts[0] == "obj" and parts[2].startswith("o"):
                    by_oid[int(parts[1])] = p
                    by_oindex[int(parts[2][1:])] = p
            except ValueError:
                continue

    n_loaded = 0
    for i in range(n):
        path_str = ""
        if i < len(crop_paths) and crop_paths[i]:
            path_str = str(crop_paths[i])
        if not path_str and isinstance(rgb_obs[i], list) and rgb_obs[i]:
            first = rgb_obs[i][0]
            if isinstance(first, dict):
                path_str = str(
                    first.get("storage_path")
                    or first.get("source_ref")
                    or first.get("full_rgb_path")
                    or ""
                )
        if not path_str:
            oid = int(object_ids[i].item()) if isinstance(object_ids, torch.Tensor) else i
            if i in by_oindex:
                path_str = str(by_oindex[i])
            elif oid in by_oid:
                path_str = str(by_oid[oid])

        path = _resolve_path(path_str) if path_str else None
        arr = _load_rgb_uint8(path, max_side=max_side) if path is not None else None
        if arr is None:
            continue

        obs: Dict[str, Any]
        if isinstance(rgb_obs[i], list) and rgb_obs[i] and isinstance(rgb_obs[i][0], dict):
            obs = dict(rgb_obs[i][0])
        else:
            obs = {}
        obs["image"] = arr
        obs["storage_path"] = str(path.resolve()) if path is not None else path_str
        obs["source_ref"] = obs["storage_path"]
        rgb_obs[i] = [obs]
        crop_paths[i] = obs["storage_path"]
        n_loaded += 1

    scene_state["rgb_observations"] = rgb_obs
    scene_state["best_view_crop_path"] = crop_paths
    return n_loaded


def _build_oid_to_crop_path(scene_state: dict) -> Dict[int, str]:
    """Map object_id → absolute crop path for click-time reload."""
    out: Dict[int, str] = {}
    means = scene_state.get("means")
    n = int(means.shape[0]) if isinstance(means, torch.Tensor) else 0
    object_ids = scene_state.get("object_id")
    crop_paths = scene_state.get("best_view_crop_path") or []
    rgb_obs = scene_state.get("rgb_observations") or []
    for i in range(n):
        oid = int(object_ids[i].item()) if isinstance(object_ids, torch.Tensor) else i
        path = ""
        if i < len(crop_paths) and crop_paths[i]:
            path = str(crop_paths[i])
        if not path and i < len(rgb_obs) and isinstance(rgb_obs[i], list) and rgb_obs[i]:
            first = rgb_obs[i][0]
            if isinstance(first, dict):
                path = str(first.get("storage_path") or first.get("source_ref") or "")
        resolved = _resolve_path(path)
        if resolved is not None:
            out[oid] = str(resolved)
    return out


def _install_click_image_fix(vis: Any, oid_to_crop: Dict[int, str]) -> None:
    """Ensure Object image panel updates with crop array on cube click."""
    vis._pipeline_oid_to_crop = dict(oid_to_crop)
    orig_prepare = vis._prepare_image
    orig_click = vis._handle_object_click
    orig_set_image = vis._set_clicked_image

    def _prepare_image_with_path(self: Any, image: object | None) -> Optional[np.ndarray]:
        # Decode storage_path if FARM left only a path on the obs dict
        if isinstance(image, dict):
            arr = image.get("image")
            if arr is None:
                for key in ("storage_path", "source_ref", "full_rgb_path"):
                    p = _resolve_path(str(image.get(key) or ""))
                    if p is not None:
                        loaded = _load_rgb_uint8(p, max_side=480)
                        if loaded is not None:
                            image = dict(image)
                            image["image"] = loaded
                            break
            elif isinstance(arr, str):
                p = _resolve_path(arr)
                if p is not None:
                    loaded = _load_rgb_uint8(p, max_side=480)
                    if loaded is not None:
                        image = dict(image)
                        image["image"] = loaded
        return orig_prepare(image)

    def _set_clicked_image_safe(self: Any, image: Optional[np.ndarray]) -> None:
        display = getattr(self, "_image_display", None)
        if display is None:
            log.warning(
                "Object image GUI handle is missing — panel will not update "
                "(viser gui add_image failed?)"
            )
            return
        img = image if image is not None else np.zeros((64, 64, 3), dtype=np.uint8)
        # Viser 0.x / 1.x attribute differences
        applied = False
        for attr in ("image", "value"):
            if hasattr(display, attr):
                try:
                    setattr(display, attr, img)
                    applied = True
                except Exception as exc:
                    log.debug("set %s failed: %s", attr, exc)
        if not applied:
            # try original
            with contextlib.suppress(Exception):
                orig_set_image(img)
        else:
            # force notifier if present
            with contextlib.suppress(Exception):
                if hasattr(display, "removed") and not display.removed:
                    pass

    def _handle_object_click_with_crop(self: Any, obj_id: int) -> None:
        orig_click(obj_id)
        # If gallery still empty, load from our oid→path map and force-set
        display = getattr(self, "_image_display", None)
        needs = True
        if display is not None:
            cur = getattr(display, "image", None)
            if isinstance(cur, np.ndarray) and cur.size > 64 * 64:
                # non-placeholder (placeholder is 64x64 black)
                if not (cur.shape[0] == 64 and cur.shape[1] == 64 and int(cur.sum()) == 0):
                    needs = False
        if not needs:
            return
        path = getattr(self, "_pipeline_oid_to_crop", {}).get(int(obj_id))
        if not path:
            log.info("Object %s: no crop image on file", obj_id)
            return
        arr = _load_rgb_uint8(Path(path), max_side=480)
        if arr is None:
            log.warning("Object %s: failed to load crop %s", obj_id, path)
            return
        log.info("Object %s: showing crop %s %s", obj_id, path, arr.shape)
        self._set_clicked_image(arr)

    vis._prepare_image = types.MethodType(_prepare_image_with_path, vis)
    vis._set_clicked_image = types.MethodType(_set_clicked_image_safe, vis)
    vis._handle_object_click = types.MethodType(_handle_object_click_with_crop, vis)
    # Also rewire existing cube on_click handlers require cubes rebuilt — done on next update()
    log.info(
        "Click image loader installed for %d object crops "
        "(click a **cube**, not a floating label)",
        len(oid_to_crop),
    )


def _detections_as_scene(
    packs: Sequence[dict],
    vocab: Sequence[str],
    *,
    feature_dim: int = 384,
) -> dict:
    """Synthetic scene_state from Phase 2 packs (one object per detection)."""
    means_l: List[torch.Tensor] = []
    cov_l: List[torch.Tensor] = []
    feat_l: List[torch.Tensor] = []
    cid_l: List[int] = []
    score_l: List[float] = []
    for pack in packs:
        m = pack.get("means")
        c = pack.get("cov6")
        if not isinstance(m, torch.Tensor) or m.numel() == 0:
            continue
        means_l.append(m.detach().cpu().float())
        if isinstance(c, torch.Tensor) and c.shape[0] == m.shape[0]:
            cov_l.append(c.detach().cpu().float())
        else:
            eye = torch.tensor([0.05, 0, 0, 0.05, 0, 0.05], dtype=torch.float32)
            cov_l.append(eye.unsqueeze(0).expand(m.shape[0], -1).clone())
        f = pack.get("features")
        if isinstance(f, torch.Tensor) and f.shape[0] == m.shape[0]:
            feat_l.append(f.detach().cpu().float())
        cls = pack.get("class_ids")
        sc = pack.get("scores")
        for i in range(m.shape[0]):
            cid_l.append(int(cls[i].item()) if isinstance(cls, torch.Tensor) else -1)
            score_l.append(float(sc[i].item()) if isinstance(sc, torch.Tensor) else 0.0)

    if not means_l:
        raise RuntimeError("No detections with means in packs")

    means = torch.cat(means_l, dim=0)
    cov6 = torch.cat(cov_l, dim=0)
    n = int(means.shape[0])
    if feat_l:
        features = torch.cat(feat_l, dim=0)
        feature_dim = int(features.shape[1])
    else:
        features = torch.zeros(n, feature_dim)

    class_ids = torch.tensor(cid_l, dtype=torch.long)
    captions = [
        f"{_class_name(c, vocab)} score={s:.2f}" for c, s in zip(cid_l, score_l)
    ]
    cats = [_class_name(c, vocab) for c in cid_l]

    return {
        "means": means,
        "cov6": cov6,
        "features": features,
        "active": torch.ones(n, dtype=torch.bool),
        "object_id": torch.arange(n, dtype=torch.long),
        "class_ids": class_ids,
        "object_caption": captions,
        "object_caption_decision": ["keep"] * n,
        "object_category": cats,
        "object_supercategory": [""] * n,
        "object_key_attributes": [[] for _ in range(n)],
        "object_detection_category_conf": [{} for _ in range(n)],
        "rgb_observations": [[] for _ in range(n)],
        "object_image_ids": [[] for _ in range(n)],
        "viewpoint_image_ids": [[] for _ in range(n)],
        "is_locked": [False] * n,
        "images": [],
        "image_positions": [],
        "current_robot_position": None,
        "region_ids": [],
        "region_labels": [],
        "region_object_lists": [],
        "region_centroids": [],
        "region_label_confidence": [],
        "region_version": 0,
    }


def _detection_payload_from_packs(packs: Sequence[dict], vocab: Sequence[str]) -> dict:
    means_l, cov_l, caps, imgs = [], [], [], []
    for pack in packs:
        m = pack.get("means")
        c = pack.get("cov6")
        if not isinstance(m, torch.Tensor) or m.numel() == 0:
            continue
        means_l.append(m.detach().cpu())
        cov_l.append(
            c.detach().cpu()
            if isinstance(c, torch.Tensor) and c.shape[0] == m.shape[0]
            else torch.zeros(m.shape[0], 6)
        )
        cls = pack.get("class_ids")
        sc = pack.get("scores")
        for i in range(m.shape[0]):
            cid = int(cls[i].item()) if isinstance(cls, torch.Tensor) else -1
            score = float(sc[i].item()) if isinstance(sc, torch.Tensor) else 0.0
            caps.append(f"{_class_name(cid, vocab)} ({score:.2f})")
            imgs.append(None)
    if not means_l:
        return {}
    return {
        "means": torch.cat(means_l, dim=0),
        "cov6": torch.cat(cov_l, dim=0),
        "captions": caps,
        "images": imgs,
    }


def _short_label(text: str, *, max_len: int = 36) -> str:
    t = " ".join(str(text or "").split())
    # Prefer first line / pre-vote summary (enrich_labels may append " | votes: …")
    t = t.split(" | ")[0].strip()
    if len(t) > max_len:
        return t[: max_len - 1] + "…"
    return t


def _install_label_and_transparency_controls(
    vis: Any,
    *,
    cube_opacity: float = 0.12,
    cube_wireframe: bool = True,
    show_floating_labels: bool = True,
) -> None:
    """Post-process FARM cubes so labels stay usable with see-through boxes.

    FARM only attaches click handlers (and the side-panel caption) to object
    **boxes** — voxel point clouds are not clickable. Hiding cubes therefore
    removes labels. We keep cubes as transparent/wireframe hit targets and add
    optional floating ``add_label`` markers at each object center.
    """
    vis._view_cube_opacity = float(np.clip(cube_opacity, 0.0, 1.0))
    vis._view_cube_wireframe = bool(cube_wireframe)
    vis._view_show_labels = bool(show_floating_labels)
    if not hasattr(vis, "_object_label_handles"):
        vis._object_label_handles: Dict[int, Any] = {}

    orig_update_cubes = vis._update_object_cubes

    def _apply_cube_style(self: Any) -> None:
        opacity = float(np.clip(getattr(self, "_view_cube_opacity", 0.12), 0.0, 1.0))
        wireframe = bool(getattr(self, "_view_cube_wireframe", True))
        for handle in list(getattr(self, "_object_cube_handles", {}).values()):
            with contextlib.suppress(Exception):
                handle.opacity = opacity
            with contextlib.suppress(Exception):
                if hasattr(handle, "wireframe"):
                    handle.wireframe = wireframe

    def _sync_floating_labels(self: Any) -> None:
        if self._server is None:
            return
        handles: Dict[int, Any] = getattr(self, "_object_label_handles", {})
        show = bool(getattr(self, "_view_show_labels", True))
        if not show:
            for oid, h in list(handles.items()):
                with contextlib.suppress(Exception):
                    h.remove()
            self._object_label_handles = {}
            return

        ids = getattr(self, "_latest_ids", None)
        captions = getattr(self, "_latest_captions", None) or []
        edit_texts = getattr(self, "_latest_caption_edit_texts", None) or []
        if ids is None:
            return
        # centers from existing cube handles when possible
        cubes = getattr(self, "_object_cube_handles", {}) or {}
        seen: set[int] = set()
        for idx, oid in enumerate(np.asarray(ids).reshape(-1).tolist()):
            oid = int(oid)
            cube = cubes.get(oid)
            if cube is None:
                continue
            try:
                pos = np.asarray(cube.position, dtype=np.float32).reshape(3)
            except Exception:
                continue
            if not np.all(np.isfinite(pos)):
                continue
            # Prefer plain edit caption (class name) over markdown JSON
            raw = ""
            if idx < len(edit_texts) and str(edit_texts[idx]).strip():
                raw = str(edit_texts[idx])
            elif idx < len(captions):
                raw = str(captions[idx])
            # Strip markdown bold noise if any
            raw = raw.replace("**", "").replace("`", "")
            text = _short_label(raw)
            if not text:
                text = f"obj {oid}"
            # Place label slightly above the box
            try:
                dims = np.asarray(getattr(cube, "dimensions", (0.2, 0.2, 0.2)), dtype=np.float32).reshape(3)
                lift = 0.5 * float(dims[2]) + 0.05
            except Exception:
                lift = 0.15
            label_pos = pos + np.array([0.0, 0.0, lift], dtype=np.float32)
            seen.add(oid)
            existing = handles.get(oid)
            if existing is not None:
                with contextlib.suppress(Exception):
                    existing.position = label_pos
                with contextlib.suppress(Exception):
                    if hasattr(existing, "text"):
                        existing.text = text
                continue
            try:
                h = self._server.scene.add_label(
                    name=f"/object_labels/lbl_{oid}",
                    text=text,
                    position=label_pos,
                    font_size_mode="screen",
                    font_screen_scale=0.85,
                    anchor="bottom-center",
                    depth_test=False,
                )
                handles[oid] = h
            except Exception:
                continue
        for oid in [k for k in handles if k not in seen]:
            h = handles.pop(oid, None)
            if h is not None:
                with contextlib.suppress(Exception):
                    h.remove()
        self._object_label_handles = handles

    def _update_object_cubes_wrapped(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_update_cubes(*args, **kwargs)
        _apply_cube_style(self)
        _sync_floating_labels(self)

    vis._update_object_cubes = types.MethodType(_update_object_cubes_wrapped, vis)
    vis._apply_cube_view_style = types.MethodType(_apply_cube_style, vis)
    vis._sync_object_floating_labels = types.MethodType(_sync_floating_labels, vis)

    # GUI: transparency + labels (does not replace FARM Filters folder)
    gui = getattr(getattr(vis, "_server", None), "gui", None)
    if gui is None:
        return

    def _add_slider(label: str, min_v: float, max_v: float, step: float, initial: float):
        fn = getattr(gui, "add_slider", None)
        if fn is None:
            return None
        try:
            return fn(label, min=min_v, max=max_v, step=step, initial_value=initial)
        except TypeError:
            with contextlib.suppress(Exception):
                return fn(label, min_v, max_v, step, initial)
        except Exception:
            return None

    def _add_checkbox(label: str, initial: bool):
        fn = getattr(gui, "add_checkbox", None)
        if fn is None:
            return None
        try:
            return fn(label, initial_value=initial)
        except TypeError:
            with contextlib.suppress(Exception):
                return fn(label, initial)
        except Exception:
            return None

    try:
        with gui.add_folder("Display (pipeline)"):
            op_slider = _add_slider(
                "Cube opacity",
                0.0,
                1.0,
                0.02,
                float(vis._view_cube_opacity),
            )
            wire_cb = _add_checkbox("Cube wireframe", bool(vis._view_cube_wireframe))
            label_cb = _add_checkbox("Show class labels", bool(vis._view_show_labels))
            help_md = getattr(gui, "add_markdown", None)
            if callable(help_md):
                with contextlib.suppress(Exception):
                    help_md(
                        "Transparent cubes stay **clickable** so the side panel "
                        "shows the full label. Floating labels sit on each object "
                        "center (per fused object / voxel cluster). "
                        "Voxel **points** themselves are not clickable in Viser."
                    )
    except Exception as exc:
        log.warning("Could not add Display GUI: %s", exc)
        return

    def _refresh_style() -> None:
        if getattr(vis, "_server", None) is None:
            return
        with contextlib.suppress(Exception):
            with vis._server.atomic():
                vis._apply_cube_view_style()
                vis._sync_object_floating_labels()
            vis._server.flush()

    if op_slider is not None:
        on_update = getattr(op_slider, "on_update", None)
        if callable(on_update):
            @on_update
            def _(_e=None, slider=op_slider):
                try:
                    vis._view_cube_opacity = float(getattr(slider, "value"))
                except Exception:
                    return
                _refresh_style()

    if wire_cb is not None:
        on_update = getattr(wire_cb, "on_update", None)
        if callable(on_update):
            @on_update
            def _(_e=None, cb=wire_cb):
                try:
                    vis._view_cube_wireframe = bool(getattr(cb, "value"))
                except Exception:
                    return
                # Wireframe cannot always be toggled in-place; rebuild boxes.
                state = getattr(vis, "_latest_scene_state", None)
                if state is not None and vis._server is not None:
                    with contextlib.suppress(Exception):
                        with vis._server.atomic():
                            vis._update_gaussians(state)
                        vis._server.flush()

    if label_cb is not None:
        on_update = getattr(label_cb, "on_update", None)
        if callable(on_update):
            @on_update
            def _(_e=None, cb=label_cb):
                try:
                    vis._view_show_labels = bool(getattr(cb, "value"))
                except Exception:
                    return
                _refresh_style()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline Viser viewer for pipeline Phase 2/3 artifacts"
    )
    p.add_argument(
        "--scene-state",
        type=Path,
        default=None,
        help="Path to scene_state.pt — prefers outputs/phase4/scene_state_with_crops.pt "
        "then outputs/phase3.5/scene_state_stella.pt then outputs/phase3/scene_state.pt",
    )
    p.add_argument(
        "--crops-dir",
        type=Path,
        default=None,
        help="Optional Phase 4a crops/ dir to hydrate Object image thumbnails",
    )
    p.add_argument(
        "--no-hydrate-crops",
        action="store_true",
        help="Do not load crop JPEGs into rgb_observations for the image panel",
    )
    p.add_argument(
        "--det-dir",
        type=Path,
        default=None,
        help="Directory of Phase 2 detections_kf*.pt",
    )
    p.add_argument(
        "--vocab",
        type=Path,
        default=_REPO_ROOT / "vocab" / "construction_vocab.txt",
        help="Text vocab for class_id → name",
    )
    p.add_argument(
        "--mode",
        choices=("map", "detections", "both"),
        default="map",
        help="map: scene_state only; detections: packs as objects; "
        "both: map as objects + det Gaussians overlay",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--no-gaussians", action="store_true",
                   help="Hide cov ellipsoid Gaussian splats")
    p.add_argument("--no-voxels", action="store_true",
                   help="Hide per-object voxel point clouds")
    p.add_argument("--covisibility", action="store_true",
                   help="Draw covisibility edges (can be dense)")
    p.add_argument("--show-detections", action="store_true",
                   help="Overlay Phase 2 detections when --det-dir is set")
    p.add_argument(
        "--bg-from-det-points",
        action="store_true",
        help="Subsample det_points_flat as a background world cloud",
    )
    p.add_argument("--bg-max-points", type=int, default=250_000)
    p.add_argument(
        "--max-packs",
        type=int,
        default=0,
        help="Limit Phase 2 packs loaded (0 = all; useful for large runs)",
    )
    p.add_argument(
        "--max-voxel-points-per-object",
        type=int,
        default=500,
        help="Cap points drawn per object voxel cloud (0 = unlimited)",
    )
    p.add_argument(
        "--cube-opacity",
        type=float,
        default=0.12,
        help="Object cube opacity 0..1 (default 0.12 translucent; keep >0 to click)",
    )
    p.add_argument(
        "--solid-cubes",
        action="store_true",
        help="Filled solid cubes (default is wireframe + transparency)",
    )
    p.add_argument(
        "--no-labels",
        action="store_true",
        help="Disable floating class labels at object centers",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    # Auto-resolve best available scene state if not specified
    if args.scene_state is None:
        for candidate in [
            _REPO_ROOT / "outputs" / "phase4" / "scene_state_with_crops.pt",
            _REPO_ROOT / "outputs" / "phase3.5" / "scene_state_stella.pt",
            _REPO_ROOT / "outputs" / "phase3" / "scene_state.pt",
        ]:
            if candidate.is_file():
                args.scene_state = candidate
                log.info("Auto-selected scene state: %s", candidate)
                break

    vocab = _load_vocab(args.vocab)
    if vocab:
        log.info("Loaded vocab (%d classes) from %s", len(vocab), args.vocab)

    packs: List[dict] = []
    if args.det_dir is not None:
        paths = _sorted_packs(args.det_dir)
        if args.max_packs > 0:
            paths = paths[: args.max_packs]
        log.info("Loading %d Phase 2 packs from %s …", len(paths), args.det_dir)
        for path in paths:
            packs.append(torch.load(path, map_location="cpu", weights_only=False))

    mode = args.mode
    if mode == "map" and args.scene_state is None:
        if packs:
            log.warning("--scene-state missing; falling back to --mode detections")
            mode = "detections"
        else:
            log.error("Need --scene-state and/or --det-dir")
            return 2

    scene_state: Optional[dict] = None
    if mode in ("map", "both"):
        if args.scene_state is None or not args.scene_state.is_file():
            log.error("scene_state.pt not found: %s", args.scene_state)
            return 2
        log.info("Loading scene state %s", args.scene_state)
        scene_state = torch.load(args.scene_state, map_location="cpu", weights_only=False)
        _enrich_labels(scene_state, vocab)
        n = int(scene_state["means"].shape[0]) if isinstance(scene_state.get("means"), torch.Tensor) else 0
        n_act = int(scene_state["active"].sum().item()) if isinstance(scene_state.get("active"), torch.Tensor) else 0
        log.info("Map objects: total=%d active=%d", n, n_act)

        crops_dir = args.crops_dir
        if crops_dir is None:
            # Look for phase4 crops in repo/ layout
            for guess in [
                _REPO_ROOT / "outputs" / "phase4" / "crops",
                _REPO_ROOT.parent / "pipeline" / "phase4-caption-best-view" / "output" / "crops",
            ]:
                if guess.is_dir():
                    crops_dir = guess
                    break
        if not args.no_hydrate_crops:
            n_img = _hydrate_object_images(scene_state, crops_dir=crops_dir)
            log.info(
                "Hydrated %d Object images for Viser click panel%s",
                n_img,
                f" (crops_dir={crops_dir})" if crops_dir else "",
            )
            if n_img == 0:
                log.warning(
                    "No object crop images available — Object image panel will stay blank. "
                    "Run Phase 4a then launch with e.g.\n"
                    "  --scene-state ../phase4-caption-best-view/output/scene_state_with_crops.pt\n"
                    "  --crops-dir   ../phase4-caption-best-view/output/crops"
                )
            else:
                # keep map for click-time force load
                scene_state["_pipeline_n_hydrated_images"] = n_img

    if mode == "detections":
        if not packs:
            log.error("--mode detections requires --det-dir")
            return 2
        scene_state = _detections_as_scene(packs, vocab)
        log.info("Synthetic scene from detections: %d objects", scene_state["means"].shape[0])

    assert scene_state is not None

    from scene_graph.visualization.viser_visualizer import PipelineViserVisualizer

    # Keep live_rgb folder so GUI folder order matches FARM; "Object image" is sibling.
    vis = PipelineViserVisualizer(
        enabled=True,
        host=args.host,
        port=args.port,
        live_rgb_enabled=True,
        object_gaussians_enabled=not args.no_gaussians,
        object_voxel_cloud_enabled=(
            not args.no_voxels and mode in ("map", "both")
        ),
        object_voxel_max_points_per_object=args.max_voxel_points_per_object,
        object_box_from_voxels=True,
        covisibility_connections_enabled=bool(args.covisibility),
        image_pose_axes_enabled=False,
        regions_enabled=False,
        object_connections_enabled=False,
        object_image_connections_enabled=False,
    )
    if not vis.enabled:
        log.error("Viser failed to start (is `viser` installed? pip install viser)")
        return 1

    if vis._image_display is None:
        log.error(
            "Viser did not create the Object image panel (gui.add_image failed). "
            "Check the viser version / browser console."
        )

    oid_to_crop = _build_oid_to_crop_path(scene_state)
    _install_click_image_fix(vis, oid_to_crop)

    _install_label_and_transparency_controls(
        vis,
        cube_opacity=args.cube_opacity,
        cube_wireframe=not args.solid_cubes,
        show_floating_labels=not args.no_labels,
    )
    log.info(
        "Display: cube_opacity=%.2f wireframe=%s floating_labels=%s "
        "(GUI folder 'Display (pipeline)' can change these live)",
        args.cube_opacity,
        not args.solid_cubes,
        not args.no_labels,
    )

    det_info: Optional[dict] = None
    if (args.show_detections or mode == "both") and packs:
        det_info = _detection_payload_from_packs(packs, vocab) or None
        if det_info is not None:
            log.info(
                "Detection overlay: %d proposals",
                int(det_info["means"].shape[0]),
            )

    poses = _collect_poses(packs) if packs else []
    # Empty colors/depths — we only need object geometry from scene_state.
    # Pass last poses so camera frame / trajectory helpers have something.
    vis.update(
        [],
        [],
        [],
        poses[-8:] if poses else [],
        scene_state,
        detection_info=det_info,
        detection_neighbors=None,
    )

    means = scene_state.get("means")
    active = scene_state.get("active")
    if isinstance(means, torch.Tensor) and means.numel() > 0:
        m_np = means.detach().cpu().numpy()
        if isinstance(active, torch.Tensor) and active.shape[0] == means.shape[0]:
            m_np = m_np[active.detach().cpu().numpy().astype(bool)]
        if m_np.size:
            vis.set_home_view(m_np)

    if poses:
        try:
            pose_stack = torch.stack(poses, dim=0).numpy()
            vis.add_trajectory(pose_stack, name="/kf_poses", axes_length=0.12)
            log.info("Trajectory: %d face poses from Phase 2 packs", len(poses))
        except Exception as exc:
            log.warning("Could not add trajectory: %s", exc)

    if args.bg_from_det_points and packs:
        pts, cols = _subsample_det_points(packs, max_points=args.bg_max_points)
        if pts is not None:
            vis.add_background_point_cloud(pts, cols, point_size=0.02, name="/det_points")
            log.info("Background det cloud: %d points", pts.shape[0])
        else:
            log.warning("No det_points_flat found for background cloud")

    url = f"http://{args.host}:{args.port}"
    log.info("Viser is serving — open %s", url)
    log.info(
        "What you should see: translucent/wireframe object boxes (click for full "
        "caption), floating class labels, optional voxel clouds / Gaussians."
    )
    log.info(
        "Keep cubes slightly visible (non-zero opacity) to click labels — "
        "voxel points alone are not interactive in Viser."
    )
    log.info("Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        log.info("Shutting down.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
