"""Unit tests for COLMAP face naming / grouping helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from colmap_io import (  # noqa: E402
    default_face_meta,
    parse_face_image_name,
    reproject_point_to_face,
)


def test_parse_face_image_name():
    face_id, frame = parse_face_image_name("pano_camera2/frame_0001.jpg")
    assert face_id == 2
    assert frame == "frame_0001.jpg"


def test_parse_face_image_name_nested():
    face_id, frame = parse_face_image_name("pano_camera0/seq/a.png")
    assert face_id == 0
    assert frame == "seq/a.png"


def test_parse_invalid():
    with pytest.raises(ValueError):
        parse_face_image_name("camera0/foo.jpg")


def test_default_face_meta_four_faces():
    meta = default_face_meta(num_faces=4)
    assert meta["num_faces"] == 4
    assert len(meta["cam_from_pano"]) == 4
    assert meta["render_type"] == "perspective_non_overlapping"


def test_reproject_identity():
    # Camera at origin looking +Z, point at (0,0,5) -> center pixel for centered K
    K = np.array([[100.0, 0, 50.0], [0, 100.0, 50.0], [0, 0, 1.0]])
    E = np.eye(4)
    uv, z = reproject_point_to_face(np.array([0.0, 0.0, 5.0]), E, K)
    assert abs(z - 5.0) < 1e-9
    assert abs(uv[0] - 50.0) < 1e-9
    assert abs(uv[1] - 50.0) < 1e-9
