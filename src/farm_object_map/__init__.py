"""FARM-style per-frame object mapping from monocular video + posed RGB-D."""

from .depth import DepthMap, DepthSource
from .geometry import (
    ObjectGaussian,
    SparseVoxels,
    points_to_gaussian,
    transform_points_cam_to_world,
    unproject_masked_depth,
    voxelize_points,
)
from .mapper import MappedObject, map_detections_to_objects

__all__ = [
    "DepthMap",
    "DepthSource",
    "ObjectGaussian",
    "SparseVoxels",
    "MappedObject",
    "points_to_gaussian",
    "transform_points_cam_to_world",
    "unproject_masked_depth",
    "voxelize_points",
    "map_detections_to_objects",
]
