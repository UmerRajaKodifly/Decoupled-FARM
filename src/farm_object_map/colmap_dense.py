"""COLMAP dense stereo depth placeholder behind the DepthSource interface."""

from __future__ import annotations

import logging
import struct
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .depth import DepthMap, DepthSource

logger = logging.getLogger(__name__)


def _run(cmd: list[str]) -> None:
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def read_colmap_depth_bin(path: str | Path) -> np.ndarray:
    """Read COLMAP ``*.geometric.bin`` / ``*.photometric.bin`` depth maps.

    Format (little-endian): text header ``width&height&channels&`` then
    ``width * height * channels`` float32 values, row-major.
    """
    data = Path(path).read_bytes()
    header_end = 0
    amps = 0
    for i, byte in enumerate(data):
        if byte == ord("&"):
            amps += 1
            if amps == 3:
                header_end = i + 1
                break
    header = data[: header_end - 1].decode("ascii")
    width_s, height_s, channels_s = header.split("&")
    width, height, channels = int(width_s), int(height_s), int(channels_s)
    payload = data[header_end:]
    expected = width * height * channels * 4
    if len(payload) < expected:
        raise ValueError(f"{path}: expected {expected} bytes of floats, got {len(payload)}")
    arr = np.frombuffer(payload[:expected], dtype=np.float32).reshape(height, width, channels)
    return arr[:, :, 0]


def undistort_and_mvs(
    image_dir: str | Path,
    sparse_model: str | Path,
    dense_workspace: str | Path,
    *,
    max_image_size: int = 2000,
) -> Path:
    """Run ``image_undistorter`` + ``patch_match_stereo``. Returns dense dir."""
    image_dir = Path(image_dir)
    sparse_model = Path(sparse_model)
    dense_workspace = Path(dense_workspace)
    dense_workspace.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "colmap",
            "image_undistorter",
            "--image_path",
            str(image_dir),
            "--input_path",
            str(sparse_model),
            "--output_path",
            str(dense_workspace),
            "--output_type",
            "COLMAP",
            "--max_image_size",
            str(max_image_size),
        ]
    )
    _run(
        [
            "colmap",
            "patch_match_stereo",
            "--workspace_path",
            str(dense_workspace),
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.geom_consistency",
            "true",
        ]
    )
    return dense_workspace


class ColmapMvsDepthSource:
    """Load per-frame geometric depth written by ``patch_match_stereo``."""

    source_id = "colmap_mvs"
    units = "sfm"

    def __init__(self, dense_workspace: str | Path, *, rgb_hw: tuple[int, int] | None = None):
        self.dense_workspace = Path(dense_workspace)
        self.depth_dir = self.dense_workspace / "stereo" / "depth_maps"
        self.undistorted_images = self.dense_workspace / "images"
        self.rgb_hw = rgb_hw  # (H, W) of original RGB if we must resize

    def _depth_path(self, frame_name: str) -> Path:
        geometric = self.depth_dir / f"{frame_name}.geometric.bin"
        photometric = self.depth_dir / f"{frame_name}.photometric.bin"
        if geometric.exists():
            return geometric
        if photometric.exists():
            return photometric
        # COLMAP sometimes prefixes with the relative image path using dots.
        matches = list(self.depth_dir.glob(f"*{Path(frame_name).name}*.geometric.bin"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"No COLMAP depth map for {frame_name} under {self.depth_dir}")

    def depth_for_frame(self, frame_name: str) -> DepthMap:
        depth = read_colmap_depth_bin(self._depth_path(frame_name))
        # Patch-match marks failures as <= 0 or NaN.
        if self.rgb_hw is not None and depth.shape != self.rgb_hw:
            depth = cv2.resize(
                depth,
                (self.rgb_hw[1], self.rgb_hw[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        valid = np.isfinite(depth) & (depth > 0)
        h, w = depth.shape
        return DepthMap(
            depth_m=depth.astype(np.float32),
            valid_mask=valid,
            frame_hw=(h, w),
            units=self.units,
            source=self.source_id,
        )


# Silence unused import if Protocol is only used for typing docs.
_: type[DepthSource] = ColmapMvsDepthSource
