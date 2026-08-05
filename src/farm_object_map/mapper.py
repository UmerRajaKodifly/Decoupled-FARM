"""Per-object 3D construction: mask → depth → unproject → world → Gaussian + voxels."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .associate import IoUTracker
from .depth import DepthSource
from .detect import Detection, YOLOEDetector
from .geometry import (
    ObjectGaussian,
    SparseVoxels,
    merge_voxels,
    points_to_gaussian,
    project_world_point,
    transform_points_cam_to_world,
    unproject_masked_depth,
    voxelize_points,
)
from .poses import FramePose

logger = logging.getLogger(__name__)


@dataclass
class ViewObservation:
    frame_name: str
    n_points: int
    dropped_invalid_depth: int
    mask_pixels: int


@dataclass
class MappedObject:
    object_id: int
    label: str
    gaussian: ObjectGaussian | None = None
    voxels: SparseVoxels | None = None
    points_world: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    observations: list[ViewObservation] = field(default_factory=list)

    def ingest_world_points(
        self,
        points_world: np.ndarray,
        *,
        voxel_size: float,
        observation: ViewObservation,
        keep_all_points: bool = False,
    ) -> None:
        self.observations.append(observation)
        if points_world.size == 0:
            return
        if keep_all_points:
            if self.points_world.size == 0:
                self.points_world = points_world
            else:
                self.points_world = np.concatenate([self.points_world, points_world], axis=0)
        new_vox = voxelize_points(points_world, voxel_size)
        self.voxels = new_vox if self.voxels is None else merge_voxels(self.voxels, new_vox)
        # Recompute Gaussian from accumulated voxel centres (stable, bounded)
        # plus this view's points for mean/cov fidelity on first views.
        if self.voxels is not None and self.voxels.ijk.shape[0] > 0:
            centres = (self.voxels.ijk.astype(np.float64) + 0.5) * self.voxels.voxel_size + self.voxels.origin
            self.gaussian = points_to_gaussian(centres.astype(np.float32), min_points=1)
        else:
            self.gaussian = points_to_gaussian(points_world, min_points=1)


def detection_to_world_points(
    detection: Detection,
    depth_source: DepthSource,
    frame: FramePose,
    *,
    min_depth_points: int = 30,
) -> tuple[np.ndarray, dict]:
    depth = depth_source.depth_for_frame(frame.frame_name)
    if depth.frame_hw != detection.mask.shape:
        # Depth must already match RGB. Fail loudly rather than silently warp.
        raise ValueError(
            f"Depth {depth.frame_hw} != mask {detection.mask.shape} for {frame.frame_name}. "
            "Resample depth onto the RGB grid in the DepthSource, not here."
        )
    pts_cam, stats = unproject_masked_depth(depth, detection.mask, frame.K)
    stats["label"] = detection.label
    stats["frame_name"] = frame.frame_name
    if pts_cam.shape[0] < min_depth_points:
        stats["skipped_few_points"] = True
        return np.zeros((0, 3), dtype=np.float32), stats
    stats["skipped_few_points"] = False
    pts_world = transform_points_cam_to_world(pts_cam, frame.T_world_cam)
    return pts_world, stats


def map_detections_to_objects(
    frames: list[FramePose],
    image_loader,
    detector: YOLOEDetector,
    depth_source: DepthSource,
    *,
    voxel_size: float = 0.05,
    iou_threshold: float = 0.3,
    min_depth_points: int = 30,
) -> tuple[dict[int, MappedObject], list[dict]]:
    """Run YOLOE + IoU tracker + per-detection unprojection over all posed frames."""
    tracker = IoUTracker(iou_threshold=iou_threshold)
    objects: dict[int, MappedObject] = {}
    drop_log: list[dict] = []

    for frame_index, frame in enumerate(frames):
        image = image_loader(frame.frame_name)
        detections = detector.detect(image)
        assigned = tracker.update(frame_index, detections)
        logger.info(
            "Frame %s: %d detections, %d tracks updated",
            frame.frame_name,
            len(detections),
            len(assigned),
        )
        for track_id, det in assigned:
            pts_world, stats = detection_to_world_points(
                det, depth_source, frame, min_depth_points=min_depth_points
            )
            drop_log.append(stats)
            if stats.get("dropped_invalid_depth", 0) and stats.get("mask_pixels", 0):
                drop_frac = stats["dropped_invalid_depth"] / max(stats["mask_pixels"], 1)
                if drop_frac > 0.8:
                    logger.warning(
                        "Mask/depth mismatch? %s %s dropped %.0f%% of mask pixels",
                        frame.frame_name,
                        det.label,
                        100.0 * drop_frac,
                    )
            obj = objects.get(track_id)
            if obj is None:
                obj = MappedObject(object_id=track_id, label=det.label)
                objects[track_id] = obj
            obj.ingest_world_points(
                pts_world,
                voxel_size=voxel_size,
                observation=ViewObservation(
                    frame_name=frame.frame_name,
                    n_points=int(pts_world.shape[0]),
                    dropped_invalid_depth=int(stats.get("dropped_invalid_depth", 0)),
                    mask_pixels=int(stats.get("mask_pixels", 0)),
                ),
            )
    return objects, drop_log


def save_object(obj: MappedObject, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mean = obj.gaussian.mean if obj.gaussian else np.full(3, np.nan, dtype=np.float32)
    cov = obj.gaussian.cov if obj.gaussian else np.full((3, 3), np.nan, dtype=np.float32)
    ijk = obj.voxels.ijk if obj.voxels else np.zeros((0, 3), dtype=np.int32)
    voxel_size = obj.voxels.voxel_size if obj.voxels else np.float32(0)
    np.savez_compressed(
        path,
        object_id=np.int32(obj.object_id),
        label=np.asarray(obj.label),
        mean=np.asarray(mean, dtype=np.float32),
        cov=np.asarray(cov, dtype=np.float32),
        voxels_ijk=ijk.astype(np.int32),
        voxel_size=np.float32(voxel_size),
        num_views=np.int32(len(obj.observations)),
        points_per_view=np.asarray([o.n_points for o in obj.observations], dtype=np.int32),
        frames=np.asarray([o.frame_name for o in obj.observations]),
    )
    meta = {
        "object_id": obj.object_id,
        "label": obj.label,
        "num_views": len(obj.observations),
        "num_voxels": int(ijk.shape[0]),
        "mean": mean.tolist(),
        "observations": [
            {
                "frame_name": o.frame_name,
                "n_points": o.n_points,
                "mask_pixels": o.mask_pixels,
                "dropped_invalid_depth": o.dropped_invalid_depth,
            }
            for o in obj.observations
        ],
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def reproject_mean_into_mask(
    obj: MappedObject,
    frame: FramePose,
    mask: np.ndarray,
) -> dict:
    if obj.gaussian is None:
        return {"ok": False, "reason": "no_gaussian"}
    uv, z = project_world_point(obj.gaussian.mean, frame.T_world_cam, frame.K)
    u, v = int(round(uv[0])), int(round(uv[1]))
    h, w = mask.shape[:2]
    inside_image = 0 <= u < w and 0 <= v < h
    inside_mask = bool(inside_image and mask[v, u])
    return {
        "ok": inside_mask,
        "u": u,
        "v": v,
        "z_cam": z,
        "inside_image": inside_image,
        "inside_mask": inside_mask,
        "frame_name": frame.frame_name,
        "object_id": obj.object_id,
        "label": obj.label,
    }
