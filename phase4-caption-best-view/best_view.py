"""Best-view selection + RGB crop extraction for Phase 4 (pre-caption).

Map objects already have ``object_image_ids`` (global face ids assigned in
Phase 3: ``kf_index * 4 + face_index``). We did **not** store per-view RGB
crops or scores at fuse time, so we recover the best supporting detection by
re-scanning Phase 2 packs and ranking with:

    quality = detector_score * sqrt(num_pixels)
    match   = cosine(object_feature, det_feature)
    gate    = match >= feat_sim_min and center-distance within max_dist_m

The winner's 2D box (padded) is cropped from the face RGB on disk.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("phase4.best_view")

_KF_RE = re.compile(r"detections_kf(\d+)\.pt$")


@dataclass
class BestViewResult:
    object_index: int
    object_id: int
    class_id: int
    image_id: int
    kf_index: int
    face_index: int
    det_index: int
    score: float
    num_pixels: float
    feature_sim: float
    mean_dist_m: float
    quality: float
    bbox_xyxy: Tuple[float, float, float, float]
    rgb_path: str
    crop_path: str
    ok: bool
    reason: str = ""


def sorted_pack_paths(det_dir: Path) -> List[Path]:
    paths = sorted(det_dir.glob("detections_kf*.pt"))
    if not paths:
        raise FileNotFoundError(f"No detections_kf*.pt in {det_dir}")
    return paths


def image_id_to_kf_face(image_id: int) -> Tuple[int, int]:
    """Invert Phase 3: ``image_id = kf_index * 4 + face_index``."""
    image_id = int(image_id)
    return image_id // 4, image_id % 4


def _as_cpu_float(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return None
    return t.detach().to("cpu", dtype=torch.float32)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.view(-1).float()
    b = b.view(-1).float()
    na = float(torch.linalg.vector_norm(a))
    nb = float(torch.linalg.vector_norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


class PackIndex:
    """Lazy cache: Phase 3 kf_index → pack dict (+ path).

    ``kf_index`` is the **enumerate** index over sorted packs (same as Phase 3),
    not necessarily the integer in ``kf000123`` filenames.
    """

    def __init__(self, det_dir: Path):
        self.paths = sorted_pack_paths(det_dir)
        self._cache: Dict[int, dict] = {}

    def __len__(self) -> int:
        return len(self.paths)

    def get(self, kf_index: int) -> Optional[dict]:
        if kf_index < 0 or kf_index >= len(self.paths):
            return None
        if kf_index not in self._cache:
            self._cache[kf_index] = torch.load(
                self.paths[kf_index], map_location="cpu", weights_only=False
            )
        return self._cache[kf_index]

    def face_rgb_path(self, kf_index: int, face_index: int) -> Optional[Path]:
        pack = self.get(kf_index)
        if pack is None:
            return None
        meta = pack.get("face_meta") or []
        if face_index < 0 or face_index >= len(meta):
            return None
        rgb = meta[face_index].get("rgb") if isinstance(meta[face_index], dict) else None
        if not rgb:
            return None
        p = Path(rgb)
        return p if p.is_file() else None


def _quality(score: float, num_pixels: float) -> float:
    return max(0.0, float(score)) * math.sqrt(max(1.0, float(num_pixels)))


def _pad_box(
    box: Sequence[float],
    w: int,
    h: int,
    *,
    pad_frac: float = 0.08,
    pad_px: int = 8,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(v) for v in box]
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    px = max(pad_px, int(round(pad_frac * bw)))
    py = max(pad_px, int(round(pad_frac * bh)))
    x0i = max(0, int(math.floor(x0 - px)))
    y0i = max(0, int(math.floor(y0 - py)))
    x1i = min(w, int(math.ceil(x1 + px)))
    y1i = min(h, int(math.ceil(y1 + py)))
    if x1i <= x0i:
        x1i = min(w, x0i + 1)
    if y1i <= y0i:
        y1i = min(h, y0i + 1)
    return x0i, y0i, x1i, y1i


def _det_slice_for_face(pack: dict, face_index: int) -> List[int]:
    batch = pack.get("batch_ids")
    if not isinstance(batch, torch.Tensor) or batch.numel() == 0:
        return []
    return [i for i, b in enumerate(batch.tolist()) if int(b) == int(face_index)]


def find_best_detection_for_object(
    obj_idx: int,
    scene_state: dict,
    pack_index: PackIndex,
    *,
    feat_sim_min: float = 0.35,
    max_center_dist_m: float = 4.0,
    prefer_same_class: bool = True,
) -> Optional[dict]:
    """Return dict describing the best matching detection, or None."""
    means = scene_state["means"]
    feats = scene_state.get("features")
    class_ids = scene_state.get("class_ids")
    image_ids_list = scene_state.get("object_image_ids") or []

    if obj_idx < 0 or obj_idx >= int(means.shape[0]):
        return None
    obj_mean = means[obj_idx].detach().cpu().float().view(3)
    obj_feat = None
    if isinstance(feats, torch.Tensor) and feats.shape[0] > obj_idx:
        obj_feat = feats[obj_idx].detach().cpu().float()
    obj_cls = (
        int(class_ids[obj_idx].item())
        if isinstance(class_ids, torch.Tensor) and class_ids.numel() > obj_idx
        else -1
    )

    image_ids = image_ids_list[obj_idx] if obj_idx < len(image_ids_list) else []
    if not image_ids:
        return None

    best: Optional[dict] = None
    best_rank = -1.0

    for raw_iid in image_ids:
        try:
            image_id = int(raw_iid)
        except (TypeError, ValueError):
            continue
        kf_i, face_i = image_id_to_kf_face(image_id)
        pack = pack_index.get(kf_i)
        if pack is None:
            continue
        det_ids = _det_slice_for_face(pack, face_i)
        if not det_ids:
            continue

        pack_means = _as_cpu_float(pack.get("means"))
        pack_feats = _as_cpu_float(pack.get("features"))
        pack_scores = _as_cpu_float(pack.get("scores"))
        pack_pixels = _as_cpu_float(pack.get("num_pixels"))
        pack_boxes = _as_cpu_float(pack.get("boxes_xyxy"))
        pack_cls = pack.get("class_ids")
        if pack_means is None or pack_boxes is None:
            continue

        for di in det_ids:
            if di >= pack_means.shape[0]:
                continue
            dmean = pack_means[di].view(3)
            dist = float(torch.linalg.vector_norm(dmean - obj_mean))
            if dist > float(max_center_dist_m):
                continue

            # Zero / near-zero feature vectors are common when mask-pooled
            # DINOv3 had no support; treat them as missing so crops still work.
            sim = 0.5
            if obj_feat is not None and pack_feats is not None and di < pack_feats.shape[0]:
                if float(torch.linalg.vector_norm(obj_feat)) >= 1e-6:
                    sim = _cosine(obj_feat, pack_feats[di])
            if sim < float(feat_sim_min):
                continue

            sc = float(pack_scores[di].item()) if pack_scores is not None and di < pack_scores.shape[0] else 0.5
            px = float(pack_pixels[di].item()) if pack_pixels is not None and di < pack_pixels.shape[0] else 100.0
            q = _quality(sc, px)
            class_bonus = 0.0
            if prefer_same_class and isinstance(pack_cls, torch.Tensor) and di < pack_cls.numel():
                if int(pack_cls[di].item()) == obj_cls and obj_cls >= 0:
                    class_bonus = 0.15 * q

            # Rank: prioritize quality, then feature match, lightly penalize distance
            rank = q * (0.55 + 0.45 * max(0.0, sim)) + class_bonus - 0.05 * dist
            if rank > best_rank:
                box = pack_boxes[di].tolist()
                rgb = pack_index.face_rgb_path(kf_i, face_i)
                best_rank = rank
                best = {
                    "image_id": image_id,
                    "kf_index": kf_i,
                    "face_index": face_i,
                    "det_index": di,
                    "score": sc,
                    "num_pixels": px,
                    "feature_sim": sim,
                    "mean_dist_m": dist,
                    "quality": q,
                    "bbox_xyxy": tuple(float(x) for x in box),
                    "rgb_path": str(rgb) if rgb is not None else "",
                    "rank": rank,
                }
    return best


def crop_and_save(
    rgb_path: Path,
    bbox_xyxy: Sequence[float],
    out_path: Path,
    *,
    pad_frac: float = 0.08,
) -> Tuple[bool, str, Tuple[int, int, int, int]]:
    if not rgb_path.is_file():
        return False, f"missing rgb {rgb_path}", (0, 0, 0, 0)
    try:
        img = Image.open(rgb_path).convert("RGB")
    except Exception as exc:
        return False, f"open failed: {exc}", (0, 0, 0, 0)
    w, h = img.size
    box = _pad_box(bbox_xyxy, w, h, pad_frac=pad_frac)
    crop = img.crop(box)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path, quality=90)
    return True, "", box


def select_best_views(
    scene_state: dict,
    det_dir: Path,
    crops_dir: Path,
    *,
    only_active: bool = True,
    feat_sim_min: float = 0.35,
    max_center_dist_m: float = 4.0,
    pad_frac: float = 0.08,
    max_objects: int = 0,
) -> List[BestViewResult]:
    """Write crops + return per-object results (includes failures)."""
    pack_index = PackIndex(det_dir)
    n = int(scene_state["means"].shape[0])
    active = scene_state.get("active")
    object_ids = scene_state.get("object_id")
    class_ids = scene_state.get("class_ids")

    results: List[BestViewResult] = []
    crops_dir.mkdir(parents=True, exist_ok=True)

    indices = list(range(n))
    if only_active and isinstance(active, torch.Tensor):
        indices = [i for i in indices if bool(active[i].item())]
    if max_objects > 0:
        indices = indices[: max_objects]

    log.info(
        "Best-view selection for %d objects (packs=%d) → %s",
        len(indices),
        len(pack_index),
        crops_dir,
    )

    for k, obj_idx in enumerate(indices):
        oid = (
            int(object_ids[obj_idx].item())
            if isinstance(object_ids, torch.Tensor)
            else obj_idx
        )
        cid = (
            int(class_ids[obj_idx].item())
            if isinstance(class_ids, torch.Tensor) and class_ids.numel() > obj_idx
            else -1
        )
        match = find_best_detection_for_object(
            obj_idx,
            scene_state,
            pack_index,
            feat_sim_min=feat_sim_min,
            max_center_dist_m=max_center_dist_m,
        )
        if match is None:
            results.append(
                BestViewResult(
                    object_index=obj_idx,
                    object_id=oid,
                    class_id=cid,
                    image_id=-1,
                    kf_index=-1,
                    face_index=-1,
                    det_index=-1,
                    score=0.0,
                    num_pixels=0.0,
                    feature_sim=0.0,
                    mean_dist_m=float("inf"),
                    quality=0.0,
                    bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
                    rgb_path="",
                    crop_path="",
                    ok=False,
                    reason="no matching detection in object_image_ids",
                )
            )
            continue

        crop_path = crops_dir / f"obj_{oid:06d}_o{obj_idx:04d}.jpg"
        ok, reason, used_box = crop_and_save(
            Path(match["rgb_path"]),
            match["bbox_xyxy"],
            crop_path,
            pad_frac=pad_frac,
        )
        results.append(
            BestViewResult(
                object_index=obj_idx,
                object_id=oid,
                class_id=cid,
                image_id=int(match["image_id"]),
                kf_index=int(match["kf_index"]),
                face_index=int(match["face_index"]),
                det_index=int(match["det_index"]),
                score=float(match["score"]),
                num_pixels=float(match["num_pixels"]),
                feature_sim=float(match["feature_sim"]),
                mean_dist_m=float(match["mean_dist_m"]),
                quality=float(match["quality"]),
                bbox_xyxy=tuple(float(x) for x in used_box),  # type: ignore[arg-type]
                rgb_path=str(match["rgb_path"]),
                crop_path=str(crop_path) if ok else "",
                ok=ok,
                reason=reason,
            )
        )
        if (k + 1) % 25 == 0 or k == 0:
            n_ok = sum(1 for r in results if r.ok)
            log.info("  progress %d/%d | crops_ok=%d", k + 1, len(indices), n_ok)

    return results


def apply_best_views_to_scene_state(
    scene_state: dict,
    results: Sequence[BestViewResult],
) -> dict:
    """Write best-view fields into *scene_state* (in-place) and return it."""
    n = int(scene_state["means"].shape[0])

    def _ensure_list(key: str, fill: Any) -> list:
        lst = scene_state.get(key)
        if not isinstance(lst, list):
            lst = []
        while len(lst) < n:
            lst.append(fill() if callable(fill) else fill)
        scene_state[key] = lst
        return lst

    paths = _ensure_list("best_view_crop_path", "")
    rgb_paths = _ensure_list("best_view_rgb_path", "")
    scores = _ensure_list("best_view_score", 0.0)
    sims = _ensure_list("best_view_feature_sim", 0.0)
    pixels = _ensure_list("best_view_num_pixels", 0.0)
    image_ids = _ensure_list("best_view_image_id", -1)
    bbox_t = torch.full((n, 4), -1.0, dtype=torch.float32)
    prev_bbox = scene_state.get("best_view_bbox_xyxy")
    if isinstance(prev_bbox, torch.Tensor) and prev_bbox.shape == (n, 4):
        bbox_t = prev_bbox.detach().cpu().float().clone()

    # FARM-style observation slots used by Viser click-to-image when present
    rgb_obs = _ensure_list("rgb_observations", lambda: [])
    hq_views = _ensure_list("high_quality_views", lambda: [])
    view_means = _ensure_list("view_means", lambda: [])
    view_cov6 = _ensure_list("view_cov6", lambda: [])
    hq_flag = _ensure_list("high_quality_captioning", False)

    means = scene_state["means"]
    cov6 = scene_state.get("cov6")

    for r in results:
        i = int(r.object_index)
        if i < 0 or i >= n:
            continue
        if not r.ok:
            continue
        paths[i] = r.crop_path
        rgb_paths[i] = r.rgb_path
        scores[i] = r.score
        sims[i] = r.feature_sim
        pixels[i] = r.num_pixels
        image_ids[i] = r.image_id
        bbox_t[i] = torch.tensor(r.bbox_xyxy, dtype=torch.float32)
        hq_flag[i] = True

        obs = {
            "storage_path": r.crop_path,
            "source_ref": r.crop_path,
            "full_rgb_path": r.rgb_path,
            "bbox": list(r.bbox_xyxy),
            "score": r.score,
            "num_pixels": r.num_pixels,
            "image_id": r.image_id,
            "feature_sim": r.feature_sim,
            "object_id": r.object_id,
            # Keep a small decoded thumbnail? skip to save memory — Viser can load path
            "image": None,
            "image_caption": None,
        }
        # Replace best-view list with single best crop record (Phase 4 Gemini later)
        rgb_obs[i] = [obs]
        hq_views[i] = [None]
        if isinstance(means, torch.Tensor):
            view_means[i] = [means[i].detach().cpu().float().clone()]
        if isinstance(cov6, torch.Tensor) and cov6.shape[0] > i:
            view_cov6[i] = [cov6[i].detach().cpu().float().clone()]

    scene_state["best_view_bbox_xyxy"] = bbox_t
    scene_state["best_view_crop_path"] = paths
    scene_state["best_view_rgb_path"] = rgb_paths
    scene_state["best_view_score"] = scores
    scene_state["best_view_feature_sim"] = sims
    scene_state["best_view_num_pixels"] = pixels
    scene_state["best_view_image_id"] = image_ids
    scene_state["rgb_observations"] = rgb_obs
    scene_state["high_quality_views"] = hq_views
    scene_state["view_means"] = view_means
    scene_state["view_cov6"] = view_cov6
    scene_state["high_quality_captioning"] = hq_flag
    return scene_state


def results_to_jsonable(results: Sequence[BestViewResult]) -> List[dict]:
    return [asdict(r) for r in results]
