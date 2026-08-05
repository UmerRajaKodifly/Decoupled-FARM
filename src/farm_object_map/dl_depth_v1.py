"""Drop-in slot for the colleague metric-depth model (`dl_depth_v1`).

Nothing here invents a depth network. When the checkpoint + infer function land,
wire them in ``register_infer_fn`` (or set ``FARM_DL_DEPTH_INFER`` to a
``module:function`` path) and the rest of the pipeline picks them up.

Until then ``build_dl_depth_v1_source()`` returns ``None`` and callers must
refuse COLMAP MVS rather than fall through.
"""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .depth import (
    DepthMap,
    apply_scale_to_poses,
    estimate_sfm_to_metric_scale,
)

logger = logging.getLogger(__name__)

InferFn = Callable[[np.ndarray, str], tuple[np.ndarray, np.ndarray]]
# infer(rgb_bgr HxWx3 uint8, frame_name) -> (depth_m HxW float32 metres, valid_mask HxW bool)

_INFER_FN: InferFn | None = None


def register_infer_fn(fn: InferFn | None) -> None:
    """Install or clear the colleague inference callback."""
    global _INFER_FN
    _INFER_FN = fn


def _load_infer_from_env() -> InferFn | None:
    spec = os.environ.get("FARM_DL_DEPTH_INFER", "").strip()
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError(
            f"FARM_DL_DEPTH_INFER={spec!r} must be 'package.module:function_name'"
        )
    mod_name, fn_name = spec.rsplit(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    if not callable(fn):
        raise TypeError(f"{spec} is not callable")
    return fn


class DlDepthV1Source:
    """Metric optical-axis depth, one map per RGB frame, ``units='m'``."""

    source_id = "dl_depth_v1"
    units = "m"

    def __init__(self, infer_fn: InferFn, *, image_loader) -> None:
        self._infer = infer_fn
        self._image_loader = image_loader

    def depth_for_frame(self, frame_name: str) -> DepthMap:
        rgb_bgr = self._image_loader(frame_name)
        depth_m, valid = self._infer(rgb_bgr, frame_name)
        depth_m = np.asarray(depth_m, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool)
        if depth_m.shape[:2] != rgb_bgr.shape[:2]:
            raise ValueError(
                f"dl_depth_v1 output {depth_m.shape[:2]} != RGB {rgb_bgr.shape[:2]} "
                f"for {frame_name}. Resample onto the RGB grid inside the infer fn."
            )
        if valid.shape != depth_m.shape:
            raise ValueError(f"valid_mask {valid.shape} != depth {depth_m.shape}")
        return DepthMap(
            depth_m=depth_m,
            valid_mask=valid,
            frame_hw=(int(depth_m.shape[0]), int(depth_m.shape[1])),
            units="m",
            source=self.source_id,
        )


def build_dl_depth_v1_source(*, image_loader) -> DlDepthV1Source | None:
    fn = _INFER_FN or _load_infer_from_env()
    if fn is None:
        return None
    return DlDepthV1Source(fn, image_loader=image_loader)


def align_poses_to_metric_depth(
    frames,
    depth_source: DlDepthV1Source,
    *,
    z_sfm_pairs: np.ndarray,
    z_metric_pairs: np.ndarray,
    min_pairs: int = 50,
) -> tuple[list, dict]:
    """Scale COLMAP ``T_world_cam`` translations into metres using paired depths.

    ``z_sfm_pairs`` / ``z_metric_pairs`` are 1-D optical-axis Z samples at the
    same pixels (e.g. COLMAP sparse point camera-Z vs DL depth). After this,
    ``frame.T_world_cam`` and ``dl_depth_v1`` share metric units.
    """
    report = estimate_sfm_to_metric_scale(z_sfm_pairs, z_metric_pairs, min_pairs=min_pairs)
    if not report["ok"]:
        return frames, report
    scale = float(report["scale"])
    from .geometry import invert_se3

    scaled = []
    for fr in frames:
        T = apply_scale_to_poses(fr.T_world_cam, scale)
        T_cw = invert_se3(T).astype(np.float32)
        scaled.append(
            type(fr)(
                frame_name=fr.frame_name,
                K=fr.K,
                T_world_cam=T.astype(np.float32),
                T_cam_world=T_cw,
                width=fr.width,
                height=fr.height,
                camera_model=fr.camera_model,
                image_id=fr.image_id,
            )
        )
    report["applied_to"] = "T_world_cam translations (metres); DL depth unchanged"
    return scaled, report


def npz_dir_depth_source(root: str | Path, *, source_id: str = "dl_depth_v1", units: str = "m"):
    """Read precomputed per-frame ``DepthMap`` npz files named like the RGB frame."""

    root = Path(root)

    class _NpzDirSource:
        source_id = source_id
        units = units

        def depth_for_frame(self, frame_name: str) -> DepthMap:
            stem = Path(frame_name).name
            for cand in (
                root / f"{stem}.npz",
                root / Path(stem).with_suffix(".npz"),
                root / f"{Path(stem).stem}.npz",
            ):
                if cand.is_file():
                    dm = DepthMap.load_npz(cand)
                    if dm.source != source_id:
                        logger.warning("npz %s has source=%s, expected %s", cand, dm.source, source_id)
                    return dm
            raise FileNotFoundError(f"No depth npz for {frame_name} under {root}")

    return _NpzDirSource()
