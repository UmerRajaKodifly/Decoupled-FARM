"""COLMAP / pycolmap pose + intrinsic export, with explicit convention checks.

COLMAP 4.x stores ``image.cam_from_world``: world → camera (same as classic
``qvec``/``tvec``). FARM mapping needs ``T_world_cam`` (camera → world).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap

from .geometry import invert_se3

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FramePose:
    frame_name: str
    K: np.ndarray  # (3, 3)
    T_world_cam: np.ndarray  # (4, 4) camera-to-world
    T_cam_world: np.ndarray  # (4, 4) world-to-camera = cam_from_world
    width: int
    height: int
    camera_model: str
    image_id: int


def _cam_from_world_of(image) -> object:
    """pycolmap 4.x exposes ``cam_from_world`` as a property *or* a method."""
    cfw = image.cam_from_world
    return cfw() if callable(cfw) else cfw


def _as_matrix44_from_cam_from_world(cam_from_world) -> np.ndarray:
    """Build 4x4 world-to-camera from pycolmap Rigid3d / similar."""
    if callable(cam_from_world):
        cam_from_world = cam_from_world()
    if hasattr(cam_from_world, "matrix"):
        mat = np.asarray(cam_from_world.matrix(), dtype=np.float64)
        if mat.shape == (4, 4):
            return mat
        if mat.shape == (3, 4):
            out = np.eye(4, dtype=np.float64)
            out[:3, :] = mat
            return out
    rotation = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
    translation = np.asarray(cam_from_world.translation, dtype=np.float64).reshape(3)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


def camera_to_K(camera: pycolmap.Camera) -> np.ndarray:
    """Build a pinhole K from a pycolmap camera (ignores distortion for unprojection).

    Dense MVS undistorts to a pinhole camera first. For SfM-only PINHOLE /
    SIMPLE_PINHOLE / SIMPLE_RADIAL we use fx/fy/cx/cy and log a warning if
    unused distortion coefficients are present.
    """
    model = str(getattr(camera, "model_name", None) or camera.model)
    params = list(camera.params)
    if "SIMPLE_PINHOLE" in model.upper() or model.upper().endswith("SIMPLE_PINHOLE"):
        f, cx, cy = params[:3]
        fx = fy = float(f)
    elif "PINHOLE" in model.upper() and "SIMPLE" not in model.upper():
        fx, fy, cx, cy = params[:4]
    elif "SIMPLE_RADIAL" in model.upper():
        f, cx, cy = params[:3]
        fx = fy = float(f)
        logger.warning(
            "Camera %s is %s; unprojection uses pinhole K and ignores distortion k. "
            "Prefer running image_undistorter (dense path) before object mapping.",
            getattr(camera, "camera_id", "?"),
            model,
        )
    else:
        calib = camera.calibration_matrix()
        return np.asarray(calib, dtype=np.float64)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K


def load_reconstruction(model_dir: str | Path) -> pycolmap.Reconstruction:
    return pycolmap.Reconstruction(str(model_dir))


def export_frame_poses(model_dir: str | Path) -> list[FramePose]:
    rec = load_reconstruction(model_dir)
    frames: list[FramePose] = []
    for image_id, image in rec.images.items():
        has_pose = image.has_pose() if callable(getattr(image, "has_pose", None)) else bool(image.has_pose)
        if not has_pose:
            continue
        camera = rec.cameras[image.camera_id]
        T_cam_world = _as_matrix44_from_cam_from_world(_cam_from_world_of(image))
        T_world_cam = invert_se3(T_cam_world)
        frames.append(
            FramePose(
                frame_name=str(image.name),
                K=camera_to_K(camera),
                T_world_cam=T_world_cam.astype(np.float32),
                T_cam_world=T_cam_world.astype(np.float32),
                width=int(camera.width),
                height=int(camera.height),
                camera_model=str(getattr(camera, "model_name", None) or camera.model),
                image_id=int(image_id),
            )
        )
    frames.sort(key=lambda f: f.frame_name)
    return frames


def verify_pose_convention(model_dir: str | Path, *, max_images: int = 20) -> dict:
    """Sanity-check world-to-camera vs camera-to-world.

    Checks:
    1. Camera centre from ``T_world_cam[:3,3]`` matches ``image.projection_center()``.
    2. Reprojection of triangulated 3D points via ``T_cam_world`` has low error.
    """
    rec = load_reconstruction(model_dir)
    centre_deltas: list[float] = []
    reproj_errors: list[float] = []
    n_checked = 0

    for image in rec.images.values():
        has_pose = image.has_pose() if callable(getattr(image, "has_pose", None)) else bool(image.has_pose)
        if not has_pose:
            continue
        T_cam_world = _as_matrix44_from_cam_from_world(_cam_from_world_of(image))
        T_world_cam = invert_se3(T_cam_world)
        centre_from_T = T_world_cam[:3, 3]
        centre_colmap = np.asarray(image.projection_center(), dtype=np.float64)
        centre_deltas.append(float(np.linalg.norm(centre_from_T - centre_colmap)))

        camera = rec.cameras[image.camera_id]
        K = camera_to_K(camera)
        points2d = list(image.points2D)
        sampled = 0
        for p2d in points2d:
            if sampled >= 50:
                break
            if not p2d.has_point3D():
                continue
            xyz = np.asarray(rec.points3D[p2d.point3D_id].xyz, dtype=np.float64)
            p_cam = T_cam_world[:3, :3] @ xyz + T_cam_world[:3, 3]
            if p_cam[2] <= 1e-8:
                continue
            u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
            v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
            xy = np.asarray(p2d.xy, dtype=np.float64)
            reproj_errors.append(float(np.hypot(u - xy[0], v - xy[1])))
            sampled += 1

        n_checked += 1
        if n_checked >= max_images:
            break

    mean_reproj = float(np.mean(reproj_errors)) if reproj_errors else float("nan")
    max_centre = float(np.max(centre_deltas)) if centre_deltas else float("nan")
    ok = (not centre_deltas or max_centre < 1e-4) and (
        not reproj_errors or mean_reproj < 8.0
    )
    report = {
        "images_checked": n_checked,
        "max_camera_center_delta_m": max_centre,
        "mean_reproj_error_px": mean_reproj,
        "median_reproj_error_px": float(np.median(reproj_errors)) if reproj_errors else float("nan"),
        "n_reproj_samples": len(reproj_errors),
        "convention": "T_cam_world = image.cam_from_world (world→camera); "
        "T_world_cam = inv(T_cam_world) (camera→world)",
        "ok": ok,
        "colmap_mean_reproj_error_px": float(rec.compute_mean_reprojection_error()),
        "num_reg_images": int(rec.num_reg_images()),
        "num_points3D": int(rec.num_points3D()),
    }
    logger.info("Pose convention check: %s", json.dumps(report, indent=2))
    return report


def save_poses_json(frames: list[FramePose], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for fr in frames:
        payload.append(
            {
                "frame_name": fr.frame_name,
                "image_id": fr.image_id,
                "K": fr.K.tolist(),
                "T_world_cam": fr.T_world_cam.tolist(),
                "T_cam_world": fr.T_cam_world.tolist(),
                "width": fr.width,
                "height": fr.height,
                "camera_model": fr.camera_model,
            }
        )
    path.write_text(json.dumps(payload, indent=2))
