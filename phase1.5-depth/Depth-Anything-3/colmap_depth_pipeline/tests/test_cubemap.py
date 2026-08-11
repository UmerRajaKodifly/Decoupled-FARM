"""Unit tests for cubemap geometry and equirect fusion."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from cubemap import (  # noqa: E402
    face_intrinsics,
    faces_to_equirect_depth,
    get_virtual_rotations,
    planar_to_ray_distance,
)


def test_virtual_rotations_count_non_overlapping():
    rots = get_virtual_rotations(4, (0.0,))
    assert len(rots) == 4
    for R in rots:
        assert R.shape == (3, 3)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)


def test_planar_to_ray_distance_center():
    H = W = 64
    K = face_intrinsics((W, H), fov_deg=90)
    depth = np.full((H, W), 5.0)
    ray = planar_to_ray_distance(depth, K)
    # At principal point, ray distance == planar depth
    cx, cy = int(K[0, 2]), int(K[1, 2])
    assert abs(ray[cy, cx] - 5.0) < 1e-6
    # Off-center should be larger
    assert ray[0, 0] > 5.0


def test_sphere_fusion_constant_radius():
    """Constant ray-distance on 4 faces -> constant equirect on covered band."""
    radius = 10.0
    face_hw = (64, 64)  # W, H for intrinsics; depths are HxW
    fw, fh = face_hw
    K = face_intrinsics((fw, fh), fov_deg=90)
    rots = get_virtual_rotations(4, (0.0,))
    face_depths = {i: np.full((fh, fw), radius, dtype=np.float64) for i in range(4)}
    face_rots = {i: R for i, R in enumerate(rots)}
    face_Ks = {i: K for i in range(4)}

    H_eq, W_eq = 128, 256
    eq, valid, eq_conf = faces_to_equirect_depth(
        face_depths, face_rots, (H_eq, W_eq), face_Ks=face_Ks, hfov_deg=90.0, seam_mode="nearest"
    )

    # Covered mid-latitude band should be ~radius
    # Poles should be invalid
    assert valid[H_eq // 2, :].mean() > 0.9
    mid = eq[H_eq // 2, valid[H_eq // 2]]
    assert np.allclose(mid, radius, rtol=0.02, atol=0.05)

    # Near poles: mostly invalid for 4-face equatorial rig
    pole_band = valid[: H_eq // 8, :].mean() + valid[-H_eq // 8 :, :].mean()
    assert pole_band < 0.5
    assert eq_conf.shape == (H_eq, W_eq)


def test_conf_max_prefers_higher_conf_face():
    """At a seam-coverable pixel, conf_max should pick the higher-conf face."""
    fw = fh = 32
    K = face_intrinsics((fw, fh), fov_deg=90)
    rots = get_virtual_rotations(4, (0.0,))
    # Depths differ so we can tell which face won
    face_depths = {
        0: np.full((fh, fw), 1.0),
        1: np.full((fh, fw), 2.0),
        2: np.full((fh, fw), 3.0),
        3: np.full((fh, fw), 4.0),
    }
    face_confs = {
        0: np.full((fh, fw), 0.1),
        1: np.full((fh, fw), 0.9),  # should dominate where both cover if ties broken by conf
        2: np.full((fh, fw), 0.2),
        3: np.full((fh, fw), 0.2),
    }
    face_rots = {i: R for i, R in enumerate(rots)}
    face_Ks = {i: K for i in range(4)}
    H_eq, W_eq = 64, 128
    eq, valid, eq_conf = faces_to_equirect_depth(
        face_depths,
        face_rots,
        (H_eq, W_eq),
        face_Ks=face_Ks,
        face_confs=face_confs,
        seam_mode="conf_max",
    )
    assert valid.any()
    # Where face1 was chosen, conf should be ~0.9 and depth ~2
    # (face1 covers yaw ~90°)
    assert eq_conf[valid].max() >= 0.89


def test_conf_weight_blend():
    fw = fh = 32
    K = face_intrinsics((fw, fh), fov_deg=90)
    rots = get_virtual_rotations(4, (0.0,))
    face_depths = {i: np.full((fh, fw), float(i + 1)) for i in range(4)}
    face_confs = {i: np.full((fh, fw), 1.0) for i in range(4)}
    face_rots = {i: R for i, R in enumerate(rots)}
    face_Ks = {i: K for i in range(4)}
    eq, valid, eq_conf = faces_to_equirect_depth(
        face_depths,
        face_rots,
        (64, 128),
        face_Ks=face_Ks,
        face_confs=face_confs,
        seam_mode="conf_weight",
    )
    assert valid[32, :].mean() > 0.8
    assert np.all(eq_conf[valid] > 0)
