"""Cubemap / perspective-rig geometry matching COLMAP panorama_sfm."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


# Default: PERSPECTIVE_NON_OVERLAPPING (cubemap without top/bottom).
DEFAULT_NUM_STEPS_YAW = 4
DEFAULT_PITCHES_DEG: tuple[float, ...] = (0.0,)
DEFAULT_HFOV_DEG = 90.0
DEFAULT_VFOV_DEG = 90.0


def get_virtual_rotations(
    num_steps_yaw: int = DEFAULT_NUM_STEPS_YAW,
    pitches_deg: Sequence[float] = DEFAULT_PITCHES_DEG,
) -> list[np.ndarray]:
    """
    Relative rotations of virtual cameras w.r.t. the panorama (cam_from_pano).

    Matches COLMAP python/pycolmap/panorama.py::get_virtual_rotations.
    """
    cams_from_pano_r: list[np.ndarray] = []
    yaws = np.linspace(0, 360, num_steps_yaw, endpoint=False)
    for pitch_deg in pitches_deg:
        yaw_offset = (360 / num_steps_yaw / 2) if pitch_deg > 0 else 0
        for yaw_deg in yaws + yaw_offset:
            pitch, yaw = np.deg2rad([-float(pitch_deg), -float(yaw_deg)])
            cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
            cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
            rotation_x = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, cos_pitch, -sin_pitch],
                    [0.0, sin_pitch, cos_pitch],
                ],
                dtype=np.float64,
            )
            rotation_y = np.array(
                [
                    [cos_yaw, 0.0, sin_yaw],
                    [0.0, 1.0, 0.0],
                    [-sin_yaw, 0.0, cos_yaw],
                ],
                dtype=np.float64,
            )
            cams_from_pano_r.append(rotation_x @ rotation_y)
    return cams_from_pano_r


def face_rotation(face_id: int, rotations: Sequence[np.ndarray] | None = None) -> np.ndarray:
    """Return cam_from_pano rotation for face_id."""
    if rotations is None:
        rotations = get_virtual_rotations()
    return np.asarray(rotations[face_id], dtype=np.float64)


def face_intrinsics(
    face_size: tuple[int, int] | int,
    fov_deg: float = DEFAULT_HFOV_DEG,
) -> np.ndarray:
    """
    Cross-check K for a square-FOV virtual face.

    Prefer COLMAP-reported intrinsics in production; this matches
    create_virtual_camera when W and H are already the face resolution.
    """
    if isinstance(face_size, int):
        w = h = face_size
    else:
        w, h = face_size
    focal = w / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0))
    cx, cy = w / 2.0, h / 2.0
    return np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def face_size_from_pano(
    pano_hw: tuple[int, int],
    hfov_deg: float = DEFAULT_HFOV_DEG,
    vfov_deg: float = DEFAULT_VFOV_DEG,
) -> tuple[int, int]:
    """Face (W, H) from equirect size, matching COLMAP create_virtual_camera."""
    pano_h, pano_w = pano_hw
    image_width = int(pano_w * hfov_deg / 360.0)
    image_height = int(pano_h * vfov_deg / 180.0)
    return image_width, image_height


def face_intrinsics_from_pano(
    pano_hw: tuple[int, int],
    hfov_deg: float = DEFAULT_HFOV_DEG,
    vfov_deg: float = DEFAULT_VFOV_DEG,
) -> np.ndarray:
    w, h = face_size_from_pano(pano_hw, hfov_deg, vfov_deg)
    return face_intrinsics((w, h), fov_deg=hfov_deg)


def planar_to_ray_distance(depth_planar: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Convert pinhole planar (z) depth to Euclidean ray distance.

    depth_planar: [H, W]  K: 3x3
    """
    H, W = depth_planar.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    x_norm = (xs - K[0, 2]) / K[0, 0]
    y_norm = (ys - K[1, 2]) / K[1, 1]
    ray_scale = np.sqrt(1.0 + x_norm**2 + y_norm**2)
    return depth_planar.astype(np.float64) * ray_scale


def equirect_rays(equirect_hw: tuple[int, int]) -> np.ndarray:
    """
    Unit rays in panorama/camera frame for each equirect pixel.

    Convention matches COLMAP spherical_img_from_cam inverse:
      yaw = atan2(x, z), pitch = -atan2(y, hypot(x,z))
      u = (1 + yaw/pi)/2 * W,  v = (1 - pitch*2/pi)/2 * H
    Returns rays shaped [H, W, 3] with OpenCV-ish axes (x right, y down, z forward).
    """
    H, W = equirect_hw
    if W != H * 2:
        raise ValueError(f"Expected 2:1 equirect, got H={H} W={W}")
    us = (np.arange(W, dtype=np.float64) + 0.5) / W
    vs = (np.arange(H, dtype=np.float64) + 0.5) / H
    uu, vv = np.meshgrid(us, vs)
    yaw = (uu * 2.0 - 1.0) * np.pi
    pitch = (1.0 - 2.0 * vv) * (np.pi / 2.0)
    # Inverse of COLMAP: yaw=atan2(x,z), pitch=-atan2(y, hypot(x,z))
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    x = sy * cp
    y = -sp
    z = cy * cp
    rays = np.stack([x, y, z], axis=-1)
    return rays


def _project_ray_to_face(
    rays_pano: np.ndarray,
    cam_from_pano: np.ndarray,
    K: np.ndarray,
    face_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project panorama rays into a face.

    Returns (u, v, valid) where valid means z>0 and inside image + within FOV
    (z component in camera frame positive and |x/z|,|y/z| within sensor).
    """
    H, W = face_hw
    # rays in camera: R @ ray  (cam_from_pano maps pano vectors to camera)
    # rays_pano: [..., 3]
    flat = rays_pano.reshape(-1, 3)
    rays_cam = (cam_from_pano @ flat.T).T  # [N,3]
    z = rays_cam[:, 2]
    eps = 1e-8
    x = rays_cam[:, 0] / np.maximum(z, eps)
    y = rays_cam[:, 1] / np.maximum(z, eps)
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    valid = (z > eps) & (u >= 0) & (u < W - 1e-6) & (v >= 0) & (v < H - 1e-6)
    return u.reshape(rays_pano.shape[:-1]), v.reshape(rays_pano.shape[:-1]), valid.reshape(
        rays_pano.shape[:-1]
    )


def _bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear sample image[H,W] at floating pixel coords (OpenCV top-left origin)."""
    H, W = image.shape[:2]
    u_c = np.clip(np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0), 0.0, W - 1.001)
    v_c = np.clip(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), 0.0, H - 1.001)
    u0 = np.floor(u_c).astype(np.int32)
    v0 = np.floor(v_c).astype(np.int32)
    u1 = np.clip(u0 + 1, 0, W - 1)
    v1 = np.clip(v0 + 1, 0, H - 1)
    du = u_c - u0
    dv = v_c - v0
    Ia = image[v0, u0]
    Ib = image[v0, u1]
    Ic = image[v1, u0]
    Id = image[v1, u1]
    return (
        Ia * (1 - du) * (1 - dv)
        + Ib * du * (1 - dv)
        + Ic * (1 - du) * dv
        + Id * du * dv
    )


def faces_to_equirect_depth(
    face_depths_ray: Mapping[int, np.ndarray],
    face_rotations: Mapping[int, np.ndarray],
    equirect_hw: tuple[int, int],
    face_Ks: Mapping[int, np.ndarray] | None = None,
    face_confs: Mapping[int, np.ndarray] | None = None,
    hfov_deg: float = DEFAULT_HFOV_DEG,
    seam_mode: str = "conf_weight",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fuse per-face ray-distance depths (and optional conf) into equirect maps.

    For each equirect ray, collect all faces whose FOV contains the ray.
    Seam handling (``seam_mode``):
      - ``nearest``: closest optical axis only (legacy)
      - ``conf_max``: among covering faces, pick highest conf
      - ``conf_weight``: conf-weighted blend of covering faces (default when conf given;
        falls back to nearest if no conf)

    Poles / uncovered pixels are NaN depth with conf 0.

    Returns (depth [H,W], valid_mask [H,W] bool, conf [H,W]).
    """
    H_eq, W_eq = equirect_hw
    rays = equirect_rays(equirect_hw)  # [H,W,3]
    face_ids = sorted(face_depths_ray.keys())
    if not face_ids:
        raise ValueError("No face depths provided")

    use_conf = face_confs is not None and len(face_confs) > 0
    if not use_conf and seam_mode in ("conf_max", "conf_weight"):
        seam_mode = "nearest"

    forwards = []
    for fid in face_ids:
        R = np.asarray(face_rotations[fid], dtype=np.float64)
        forwards.append(R.T @ np.array([0.0, 0.0, 1.0]))
    forwards = np.stack(forwards, axis=0)  # [F,3]
    dots = rays @ forwards.T  # [H,W,F]
    nearest = np.argmax(dots, axis=-1)

    # Sample depth (and conf) from every covering face
    F = len(face_ids)
    depth_stack = np.full((F, H_eq, W_eq), np.nan, dtype=np.float64)
    conf_stack = np.zeros((F, H_eq, W_eq), dtype=np.float64)
    inside_stack = np.zeros((F, H_eq, W_eq), dtype=bool)

    for local_idx, fid in enumerate(face_ids):
        depth_face = np.asarray(face_depths_ray[fid], dtype=np.float64)
        fh, fw = depth_face.shape
        R = np.asarray(face_rotations[fid], dtype=np.float64)
        if face_Ks is not None and fid in face_Ks:
            K = np.asarray(face_Ks[fid], dtype=np.float64)
        else:
            K = face_intrinsics((fw, fh), fov_deg=hfov_deg)

        u, v, inside = _project_ray_to_face(rays, R, K, (fh, fw))
        inside_stack[local_idx] = inside
        if not np.any(inside):
            continue
        sampled_d = _bilinear_sample(depth_face, u, v)
        depth_stack[local_idx, inside] = sampled_d[inside]
        if use_conf and fid in face_confs:
            conf_face = np.asarray(face_confs[fid], dtype=np.float64)
            if conf_face.shape != depth_face.shape:
                # Resize conf to depth face size if needed
                import cv2

                conf_face = cv2.resize(
                    conf_face.astype(np.float32),
                    (fw, fh),
                    interpolation=cv2.INTER_LINEAR,
                ).astype(np.float64)
            sampled_c = _bilinear_sample(conf_face, u, v)
            conf_stack[local_idx, inside] = np.clip(sampled_c[inside], 0.0, None)

    out = np.full((H_eq, W_eq), np.nan, dtype=np.float64)
    out_conf = np.zeros((H_eq, W_eq), dtype=np.float64)
    valid = np.zeros((H_eq, W_eq), dtype=bool)

    n_cover = inside_stack.sum(axis=0)  # [H,W]

    if seam_mode == "nearest":
        for local_idx in range(F):
            use = (nearest == local_idx) & inside_stack[local_idx]
            if not np.any(use):
                continue
            out[use] = depth_stack[local_idx][use]
            out_conf[use] = conf_stack[local_idx][use]
            valid[use] = True
    elif seam_mode == "conf_max":
        # Prefer highest-conf covering face; fall back to nearest if conf ties / zero
        masked_conf = np.where(inside_stack, conf_stack, -np.inf)
        best_conf = np.argmax(masked_conf, axis=0)
        for local_idx in range(F):
            use = (best_conf == local_idx) & inside_stack[local_idx]
            # If all confs are 0/-inf for a pixel with coverage, use nearest among covering
            no_conf = use & ~np.isfinite(masked_conf[local_idx])
            use = use & ~no_conf
            if np.any(use):
                out[use] = depth_stack[local_idx][use]
                out_conf[use] = conf_stack[local_idx][use]
                valid[use] = True
        # Fallback: covering but no conf signal
        uncovered = (n_cover > 0) & ~valid
        if np.any(uncovered):
            for local_idx in range(F):
                use = uncovered & (nearest == local_idx) & inside_stack[local_idx]
                if not np.any(use):
                    continue
                out[use] = depth_stack[local_idx][use]
                out_conf[use] = conf_stack[local_idx][use]
                valid[use] = True
    else:  # conf_weight
        # Soft blend among covering faces; single-cover pixels take that face.
        conf_pos = np.where(inside_stack, np.maximum(conf_stack, 0.0), 0.0)
        # If conf is flat/zero on covering faces, fall back to uniform over covering
        wsum = conf_pos.sum(axis=0)
        need_uniform = (n_cover > 0) & (wsum <= 1e-12)
        if np.any(need_uniform):
            conf_pos[:, need_uniform] = inside_stack[:, need_uniform].astype(np.float64)
            wsum = conf_pos.sum(axis=0)

        depth_num = np.nansum(
            np.where(inside_stack, depth_stack * conf_pos, 0.0),
            axis=0,
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            out_d = depth_num / np.maximum(wsum, 1e-12)
            out_c = np.sum(conf_pos * conf_stack, axis=0) / np.maximum(wsum, 1e-12)

        use = n_cover > 0
        out[use] = out_d[use]
        out_conf[use] = out_c[use]
        valid[use] = True

    return out, valid, out_conf


def render_face_from_equirect(
    pano: np.ndarray,
    cam_from_pano: np.ndarray,
    face_hw: tuple[int, int],
    K: np.ndarray | None = None,
    hfov_deg: float = DEFAULT_HFOV_DEG,
) -> np.ndarray:
    """
    Render one perspective face from an equirect image (host-side helper).

    pano: [H,W,3] uint8, face_hw: (W,H) note width/height order for size,
    returns [H_face, W_face, 3].
    """
    import cv2

    fw, fh = face_hw
    if K is None:
        K = face_intrinsics((fw, fh), fov_deg=hfov_deg)
    # Pixel rays in camera
    ys, xs = np.mgrid[0:fh, 0:fw].astype(np.float64)
    x_norm = (xs + 0.5 - K[0, 2]) / K[0, 0]
    y_norm = (ys + 0.5 - K[1, 2]) / K[1, 1]
    rays_cam = np.stack([x_norm, y_norm, np.ones_like(x_norm)], axis=-1)
    rays_cam /= np.linalg.norm(rays_cam, axis=-1, keepdims=True)
    # To pano: pano_from_cam = R^T
    R = np.asarray(cam_from_pano, dtype=np.float64)
    rays_pano = rays_cam @ R  # (R^T applied via row vectors: r_cam @ R = (R.T @ r_cam.T).T
    # Actually cam_from_pano @ r_pano = r_cam => r_pano = R.T @ r_cam
    # With row vectors: r_pano = r_cam @ R
    # Wait: (R @ r_pano)_row = r_pano @ R.T if column, so r_pano.T = R.T @ r_cam.T => r_pano = r_cam @ R
    # Yes r_pano = r_cam @ R when R = cam_from_pano.

    pano_h, pano_w = pano.shape[:2]
    r = rays_pano
    yaw = np.arctan2(r[..., 0], r[..., 2])
    pitch = -np.arctan2(r[..., 1], np.linalg.norm(r[..., [0, 2]], axis=-1))
    u = (1.0 + yaw / np.pi) / 2.0 * pano_w - 0.5
    v = (1.0 - pitch * 2.0 / np.pi) / 2.0 * pano_h - 0.5
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    return cv2.remap(pano, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
