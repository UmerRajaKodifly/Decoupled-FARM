"""DA3 inference for cube-face batches.

Supports:
- DA3METRIC-LARGE: monocular metric (no pose-cond benefit)
- DA3NESTED / any-view: joint multi-view depths; optional COLMAP pose conditioning
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DA3_SRC = _REPO_ROOT / "src"
if str(_DA3_SRC) not in sys.path:
    sys.path.insert(0, str(_DA3_SRC))

from colmap_io import FaceEntry

logger = logging.getLogger(__name__)

# Map short names to HuggingFace model ids.
MODEL_NAME_MAP = {
    "da3metric-large": "depth-anything/DA3METRIC-LARGE",
    "da3-metric-large": "depth-anything/DA3METRIC-LARGE",
    "da3metric": "depth-anything/DA3METRIC-LARGE",
    "da3-large-1.1": "depth-anything/DA3-LARGE-1.1",
    "da3-large": "depth-anything/DA3-LARGE",
    "da3-base": "depth-anything/DA3-BASE",
    "da3-small": "depth-anything/DA3-SMALL",
    "da3-giant": "depth-anything/DA3-GIANT",
    "da3nested-giant-large-1.1": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    "da3nested": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
}

METRIC_FOCAL_NORM = 300.0


def resolve_model_name(name: str) -> str:
    key = name.strip()
    return MODEL_NAME_MAP.get(key.lower(), key)


def is_metric_model(model_name: str) -> bool:
    n = resolve_model_name(model_name).lower()
    return "da3metric" in n or "nested" in n


def is_nested_model(model_name: str) -> bool:
    return "nested" in resolve_model_name(model_name).lower()


def supports_pose_condition(model_name: str) -> bool:
    """Pose-cond is for any-view / nested / standard DA3, not monocular METRIC."""
    n = resolve_model_name(model_name).lower()
    if "da3metric" in n and "nested" not in n:
        return False
    return True


def raw_to_metric_depth(raw: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Convert DA3METRIC network output to meters using face intrinsics."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    focal = 0.5 * (fx + fy)
    return (focal * np.asarray(raw, dtype=np.float64) / METRIC_FOCAL_NORM).astype(np.float32)


def _w2c_4x4(w2c: np.ndarray) -> np.ndarray:
    E = np.asarray(w2c, dtype=np.float64)
    if E.shape == (4, 4):
        return E
    if E.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = E
        return out
    raise ValueError(f"Unexpected extrinsics shape {E.shape}")


class DA3Runner:
    def __init__(
        self,
        model_name: str = "depth-anything/DA3METRIC-LARGE",
        device: str | None = None,
        process_res: int = 504,
        ref_view_strategy: str = "saddle_balanced",
        use_ray_pose: bool = False,
    ):
        import torch
        from depth_anything_3.api import DepthAnything3

        self.model_name = resolve_model_name(model_name)
        self.process_res = process_res
        self.ref_view_strategy = ref_view_strategy
        self.use_ray_pose = bool(use_ray_pose)
        self.metric = is_metric_model(self.model_name)
        self.nested = is_nested_model(self.model_name)
        self.pose_ok = supports_pose_condition(self.model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info(
            "Loading %s (metric=%s nested=%s pose_cond_ok=%s) on %s",
            self.model_name,
            self.metric,
            self.nested,
            self.pose_ok,
            device,
        )
        self.model = DepthAnything3.from_pretrained(self.model_name)
        self.model = self.model.to(device)
        self.model.eval()
        # Ensure metric/mono DPT sky head is active when the architecture supports it.
        head = getattr(getattr(self.model, "model", None), "head", None)
        if head is not None and hasattr(head, "use_sky_head"):
            if not head.use_sky_head:
                logger.warning("Model head had use_sky_head=False; enabling sky segmentation")
                head.use_sky_head = True
            logger.info("DA3 sky head enabled (use_sky_head=%s)", head.use_sky_head)
        else:
            logger.warning(
                "Loaded model has no DPT sky head (e.g. DualDPT); sky masks will be unavailable"
            )

    def infer_faces(
        self,
        faces: Sequence[FaceEntry],
        *,
        pose_condition: bool = False,
        align_to_input_ext_scale: bool = False,
        use_ray_pose: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """
        Run DA3 on a batch of faces.

        When ``pose_condition`` and the model supports it, pass COLMAP
        extrinsics/intrinsics for pose-conditioned multi-view depth.

        Nested depths are already meters. DA3METRIC needs ``to_metric_face_depth``.

        Returns:
            depth: [N,H,W] network depth
            conf: [N,H,W] or None
            sky: [N,H,W] bool or None
        """
        image_paths: List[str] = []
        for face in faces:
            if not face.image_path.is_file():
                raise FileNotFoundError(f"Missing face image: {face.image_path}")
            image_paths.append(str(face.image_path))

        extrinsics = None
        intrinsics = None
        if pose_condition:
            if not self.pose_ok:
                logger.warning(
                    "pose_condition requested but %s is monocular METRIC; "
                    "running without COLMAP poses (use DA3NESTED / DA3-LARGE for pose-cond)",
                    self.model_name,
                )
            else:
                extrinsics = np.stack([_w2c_4x4(f.extrinsics) for f in faces], axis=0)
                intrinsics = np.stack(
                    [np.asarray(f.intrinsics, dtype=np.float64) for f in faces], axis=0
                )
                logger.info(
                    "Pose-conditioned inference: %d views (align_to_input_ext_scale=%s)",
                    len(faces),
                    align_to_input_ext_scale,
                )

        ray = self.use_ray_pose if use_ray_pose is None else bool(use_ray_pose)
        import torch

        # DA3 always runs Umeyama when extrinsics are set; panos with few unique
        # camera centers (rig faces share a center) make it degenerate.
        _orig_align = self.model._align_to_input_extrinsics_intrinsics

        def _safe_align(
            extrinsics,
            intrinsics,
            prediction,
            align_to_input_ext_scale: bool = True,
            ransac_view_thresh: int = 10,
        ):
            if extrinsics is None:
                return prediction
            try:
                return _orig_align(
                    extrinsics,
                    intrinsics,
                    prediction,
                    align_to_input_ext_scale,
                    ransac_view_thresh,
                )
            except Exception as e:
                name = type(e).__name__
                msg = str(e).lower()
                if (
                    "GeometryException" not in name
                    and "umeyama" not in msg
                    and "degenerate" not in msg
                ):
                    raise
                logger.warning(
                    "Umeyama pose align failed (%s); keeping network depth scale, "
                    "using input extrinsics (pin to sparse later)",
                    e,
                )
                ixt = (
                    intrinsics.numpy()
                    if isinstance(intrinsics, torch.Tensor)
                    else np.asarray(intrinsics)
                )
                prediction.intrinsics = ixt
                ex = (
                    extrinsics.numpy()
                    if isinstance(extrinsics, torch.Tensor)
                    else np.asarray(extrinsics)
                )
                prediction.extrinsics = ex[..., :3, :]
                return prediction

        self.model._align_to_input_extrinsics_intrinsics = _safe_align

        def _run(res: int):
            return self.model.inference(
                image=image_paths,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                align_to_input_ext_scale=(
                    bool(align_to_input_ext_scale) if extrinsics is not None else False
                ),
                use_ray_pose=ray,
                ref_view_strategy=self.ref_view_strategy,
                process_res=res,
            )

        try:
            try:
                prediction = _run(self.process_res)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                fallback = max(280, int(self.process_res * 0.75))
                if fallback >= self.process_res:
                    raise
                logger.warning(
                    "CUDA OOM at process_res=%d with %d views; retrying at %d",
                    self.process_res,
                    len(image_paths),
                    fallback,
                )
                prediction = _run(fallback)
        finally:
            self.model._align_to_input_extrinsics_intrinsics = _orig_align

        depth = np.asarray(prediction.depth, dtype=np.float32)
        conf = None
        if prediction.conf is not None:
            conf = np.asarray(prediction.conf, dtype=np.float32)
        sky = None
        if getattr(prediction, "sky", None) is not None:
            sky = np.asarray(prediction.sky).astype(bool)
        return depth, conf, sky

    def to_metric_face_depth(self, raw_resized: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Convert resized network depth to meters for the active model."""
        if not self.metric:
            return np.asarray(raw_resized, dtype=np.float32)
        if self.nested:
            return np.asarray(raw_resized, dtype=np.float32)
        return raw_to_metric_depth(raw_resized, K)


def iter_frame_windows(
    frame_ids: Sequence[str],
    window_size: int,
    overlap: int,
) -> list[list[str]]:
    """
    Sliding windows over frame ids. Overlap is in frames.
    Ensures progress even if overlap >= window_size - 1.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    ids = list(frame_ids)
    if not ids:
        return []
    step = max(1, window_size - max(0, overlap))
    windows = []
    i = 0
    while i < len(ids):
        windows.append(ids[i : i + window_size])
        if i + window_size >= len(ids):
            break
        i += step
    return windows
