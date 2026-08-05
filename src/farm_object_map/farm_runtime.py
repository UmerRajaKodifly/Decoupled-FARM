"""FARM-exact YOLOE + Hellinger/DINO/union-find association.

This module does not re-derive formulas. It constructs FARM's own
``YOLOESegmenter`` + ``PipelineOrchestrator`` from the cloned repo and runs
the same step sequence as ``scripts/run_pipeline.py``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import torch

from .associate import IoUTracker
from .detect import Detection
from .paths import ensure_farm_on_path
from .poses import FramePose

logger = logging.getLogger(__name__)

AssociationMethod = Literal["farm", "greedy_iou"]


@dataclass
class AssociationConfig:
    method: AssociationMethod = "farm"
    # get_neighbors.py / steps.find_neighbors_for_detections defaults
    feature_sim_thresh: float = 0.5
    hellinger_thresh: float = 0.8
    assignment_mode: str = "union_all"  # union_find.find_object_correspondence
    # replica.yaml + run_pipeline.py CLI defaults (not config.py dataclass)
    model_id: str = "yoloe-v8l"
    imgsz: int = 640
    conf_thres: float = 0.25
    iou_thres: float = 0.5
    mask_erosion_px: int = 3
    mahalanobis_thresh: float = 2.0
    # run_pipeline does not pass this → YOLOESegmenter.__init__ default
    depth_mode_k_mad: float = 3.0
    min_mask_pixels: int = 50
    min_depth_points: int = 50
    # Paper / EVALUATION.md / `run_pipeline.py --dino`. replica.yaml is false.
    use_dino_features: bool = True
    iou_threshold: float = 0.3
    batch_size: int = 1


@dataclass
class FarmMappingResult:
    n_farm_objects: int
    n_iou_tracks: int
    objects_dir: Path
    summary: dict[str, Any] = field(default_factory=dict)


def build_farm_segmenter(vocab_txt: Path, cfg: AssociationConfig, device: str = "cuda"):
    ensure_farm_on_path()
    from scene_graph.segmentation import DINOFeaturesExtractor, YOLOESegmenter

    dino_extractor = None
    if cfg.use_dino_features:
        # Exact run_pipeline.py construction: DinoConfig defaults, weights_path
        # left None so DINOFeaturesExtractor → resolve_dino_backbone()
        # (vits16plus if present, else checked-in dinov3-vits16).
        dino_extractor = DINOFeaturesExtractor(
            model="facebook/dinov3-vits16-pretrain-lvd1689m",
            load_size=512,
            weights_path=None,
            device=device,
        )
    return YOLOESegmenter(
        model_id=cfg.model_id,
        vocab_file=vocab_txt,
        imgsz=cfg.imgsz,
        conf_thres=cfg.conf_thres,
        iou_thres=cfg.iou_thres,
        device=device,
        use_dino_features=cfg.use_dino_features,
        dino_extractor=dino_extractor,
        mask_erosion_px=cfg.mask_erosion_px,
        mahalanobis_thresh=cfg.mahalanobis_thresh,
        min_mask_pixels=cfg.min_mask_pixels,
        min_depth_points=cfg.min_depth_points,
        depth_mode_k_mad=cfg.depth_mode_k_mad,
    )


def _bgr_to_farm_color(image_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(np.ascontiguousarray(rgb))
    if t.dtype != torch.uint8:
        t = t.to(torch.uint8)
    return t


def smoke_yoloe_masks(
    segmenter,
    image_bgr: np.ndarray,
    dummy_depth_m: np.ndarray,
    K: np.ndarray,
) -> dict:
    """Infer masks on one RGB frame. Dummy depth is only for FARM's 3D stats path."""
    color = _bgr_to_farm_color(image_bgr)
    depth = torch.from_numpy(np.asarray(dummy_depth_m, dtype=np.float32))
    K4 = torch.eye(4, dtype=torch.float32)
    K4[:3, :3] = torch.from_numpy(np.asarray(K, dtype=np.float32))
    out = segmenter([color], [depth], [K4])
    class_ids = out.get("class_ids")
    n = int(class_ids.numel()) if isinstance(class_ids, torch.Tensor) else 0
    masks = out.get("masks") or []
    coverage = []
    pixel_counts = []
    for m in masks:
        if isinstance(m, torch.Tensor):
            mf = m.to(torch.float32)
            coverage.append(float(mf.mean().item()))
            pixel_counts.append(int(mf.sum().item()))
    names = []
    if n and hasattr(segmenter, "names"):
        ids = class_ids.detach().cpu().tolist()
        names = [
            segmenter.names[int(i)] if 0 <= int(i) < len(segmenter.names) else str(i) for i in ids
        ]
    import ultralytics

    return {
        "n_masks": n,
        "mean_mask_coverage": float(np.mean(coverage)) if coverage else 0.0,
        "total_mask_pixels": int(sum(pixel_counts)),
        "frame_hw": list(image_bgr.shape[:2]),
        "labels": names[:40],
        "ultralytics_version": ultralytics.__version__,
        "ultralytics_file": getattr(ultralytics, "__file__", ""),
    }


def _export_scene_state_objects(scene_state: dict, out_dir: Path, class_names: list[str]) -> list[dict]:
    from scene_graph.utils.geometry import VOXEL_BASE_V, unpack_voxel_keys

    out_dir.mkdir(parents=True, exist_ok=True)
    means = scene_state["means"].detach().cpu().numpy()
    cov6 = scene_state["cov6"].detach().cpu().numpy()
    obj_ids = scene_state["object_id"].detach().cpu().numpy()
    class_ids = scene_state["class_ids"].detach().cpu().numpy()
    active = scene_state["active"].detach().cpu().numpy()
    keys_flat = scene_state.get("object_voxel_keys_flat")
    offsets = scene_state.get("object_voxel_keys_offsets")
    levels = scene_state.get("object_voxel_levels")
    summaries = []
    for i in range(int(means.shape[0])):
        if not bool(active[i]):
            continue
        oid = int(obj_ids[i])
        cid = int(class_ids[i])
        label = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        mean = means[i].astype(np.float32)
        c6 = cov6[i]
        cov = np.array(
            [[c6[0], c6[1], c6[2]], [c6[1], c6[3], c6[4]], [c6[2], c6[4], c6[5]]],
            dtype=np.float32,
        )
        ijk = np.zeros((0, 3), dtype=np.int32)
        voxel_size = float(VOXEL_BASE_V)
        if isinstance(keys_flat, torch.Tensor) and isinstance(offsets, torch.Tensor) and offsets.numel() > i + 1:
            start, end = int(offsets[i].item()), int(offsets[i + 1].item())
            keys = keys_flat[start:end]
            if keys.numel():
                ijk = unpack_voxel_keys(keys).detach().cpu().numpy().astype(np.int32)
            if isinstance(levels, torch.Tensor) and levels.numel() > i:
                voxel_size = float(VOXEL_BASE_V) * (2 ** int(levels[i].item()))
        path = out_dir / f"object_{oid:04d}.npz"
        np.savez_compressed(
            path,
            object_id=np.int32(oid),
            label=np.asarray(label),
            mean=mean,
            cov=cov,
            voxels_ijk=ijk,
            voxel_size=np.float32(voxel_size),
        )
        meta = {
            "object_id": oid,
            "label": label,
            "mean": mean.tolist(),
            "cov": cov.tolist(),
            "num_voxels": int(ijk.shape[0]),
            "voxel_size": voxel_size,
            "path": str(path),
        }
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        summaries.append(meta)
    return summaries


def _detections_from_seg(seg: dict, names: list[str]) -> list[Detection]:
    masks = seg.get("masks")
    if masks is None:
        masks = []
    elif isinstance(masks, torch.Tensor) and masks.numel() == 0:
        return []
    boxes = seg.get("boxes")
    scores = seg.get("scores")
    class_ids = seg.get("class_ids")
    if not isinstance(boxes, torch.Tensor) or boxes.numel() == 0:
        return []
    boxes_np = boxes.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.ones(len(masks))
    cls_np = class_ids.detach().cpu().numpy() if isinstance(class_ids, torch.Tensor) else np.zeros(len(masks))
    dets: list[Detection] = []
    for i, mask_t in enumerate(masks):
        mask_np = (
            mask_t.detach().cpu().numpy().astype(bool)
            if isinstance(mask_t, torch.Tensor)
            else np.asarray(mask_t, dtype=bool)
        )
        cid = int(cls_np[i])
        label = names[cid] if 0 <= cid < len(names) else str(cid)
        dets.append(
            Detection(
                label=label,
                score=float(scores_np[i]),
                mask=mask_np,
                bbox_xyxy=boxes_np[i].astype(np.float32),
            )
        )
    return dets


def run_farm_association_mapping(
    frames: list[FramePose],
    image_loader,
    depth_for_frame,
    vocab_txt: Path,
    out_dir: Path,
    *,
    cfg: AssociationConfig | None = None,
    device: str = "cuda",
) -> FarmMappingResult:
    """Per-frame FARM orchestrator, plus a parallel greedy-IoU track count."""
    ensure_farm_on_path()
    from scene_graph.captioning.services import CaptionManager
    from scene_graph.config import FilteringConfig
    from scene_graph.map_update.models import initialize_scene_graph_state
    from scene_graph.pipeline import FrameBatch, PipelineOrchestrator
    from scene_graph.storage.image_save_worker import ImageSaveWorker

    cfg = cfg or AssociationConfig()
    segmenter = build_farm_segmenter(vocab_txt, cfg, device=device)
    scene_state = initialize_scene_graph_state(segmenter.feature_dim, segmenter.device)
    caption_manager = CaptionManager(scene_state=scene_state, enabled=False, debug=False)
    image_storage_dir = Path(out_dir) / "image_store"
    image_storage_dir.mkdir(parents=True, exist_ok=True)
    image_save_worker = ImageSaveWorker()
    orchestrator = PipelineOrchestrator(
        segmenter,
        scene_state,
        caption_manager,
        image_save_worker,
        image_storage_dir=image_storage_dir,
        dataset_slug="farm_object_map",
        filtering_config=FilteringConfig(),
    )

    iou_tracker = IoUTracker(iou_threshold=cfg.iou_threshold)
    iou_ids: set[int] = set()
    drop_log: list[dict] = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        for frame_index, frame in enumerate(frames):
            image_bgr = image_loader(frame.frame_name)
            depth_map = depth_for_frame(frame.frame_name)
            color = _bgr_to_farm_color(image_bgr)
            depth_np = np.asarray(depth_map.depth_m, dtype=np.float32).copy()
            depth_np[~depth_map.validity()] = 0.0
            depth_t = torch.from_numpy(depth_np)
            K4 = torch.eye(4, dtype=torch.float32)
            K4[:3, :3] = torch.from_numpy(np.asarray(frame.K, dtype=np.float32))
            T = torch.from_numpy(np.asarray(frame.T_world_cam, dtype=np.float32))
            batch = FrameBatch(
                colors=[color],
                depths=[depth_t],
                intrinsics=[K4],
                poses_world=[T],
            )
            result = orchestrator.process_batch(batch)
            seg = result.seg_outputs
            dets = _detections_from_seg(seg, list(segmenter.names))
            assigned = iou_tracker.update(frame_index, dets)
            iou_ids.update(tid for tid, _ in assigned)

            n_pix = seg.get("num_pixels")
            n_depth = seg.get("n")
            if isinstance(n_pix, torch.Tensor) and n_pix.numel():
                for i in range(int(n_pix.numel())):
                    mp = float(n_pix[i].item())
                    vd = (
                        float(n_depth[i].item())
                        if isinstance(n_depth, torch.Tensor) and n_depth.numel() > i
                        else 0.0
                    )
                    drop_log.append(
                        {
                            "frame_name": frame.frame_name,
                            "mask_pixels": mp,
                            "valid_depth_pixels": vd,
                            "dropped_invalid_depth": max(0.0, mp - vd),
                            "skipped_few_points": vd < cfg.min_depth_points,
                        }
                    )
            n_state = int(scene_state["means"].shape[0]) if isinstance(scene_state.get("means"), torch.Tensor) else 0
            logger.info(
                "FARM map frame %s dets=%d state_objects=%d iou_tracks=%d",
                frame.frame_name,
                len(dets),
                n_state,
                len(iou_ids),
            )
    finally:
        orchestrator.flush_captions()
        orchestrator.shutdown()

    summaries = _export_scene_state_objects(scene_state, out_dir / "objects", list(segmenter.names))
    import ultralytics

    summary = {
        "n_farm_objects": len(summaries),
        "n_iou_tracks": len(iou_ids),
        "association": {
            "method": "farm",
            "feature_sim_thresh": cfg.feature_sim_thresh,
            "hellinger_thresh": cfg.hellinger_thresh,
            "assignment_mode": cfg.assignment_mode,
            "use_dino_features": cfg.use_dino_features,
            "note": (
                "replica.yaml sets use_dino_features: false; FARM-exact paper/eval "
                "path is `run_pipeline.py --dino`. This wrapper defaults to DINO on."
            ),
            "yoloe": {
                "model_id": cfg.model_id,
                "imgsz": cfg.imgsz,
                "conf_thres": cfg.conf_thres,
                "iou_thres": cfg.iou_thres,
                "mask_erosion_px": cfg.mask_erosion_px,
                "mahalanobis_thresh": cfg.mahalanobis_thresh,
                "depth_mode_k_mad": cfg.depth_mode_k_mad,
                "ultralytics": ultralytics.__version__,
                "ultralytics_file": getattr(ultralytics, "__file__", ""),
            },
        },
        "low_depth_masks": sum(1 for s in drop_log if s.get("skipped_few_points")),
        "avg_valid_depth_pixels": float(np.mean([s["valid_depth_pixels"] for s in drop_log] or [0])),
        "avg_valid_points_per_object_per_view": float(
            np.mean([s["valid_depth_pixels"] for s in drop_log if not s.get("skipped_few_points")] or [0])
        ),
        "n_object_views": sum(1 for s in drop_log if not s.get("skipped_few_points")),
        "objects": summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "drop_log.json").write_text(json.dumps(drop_log, indent=2))
    return FarmMappingResult(
        n_farm_objects=len(summaries),
        n_iou_tracks=len(iou_ids),
        objects_dir=out_dir / "objects",
        summary=summary,
    )
