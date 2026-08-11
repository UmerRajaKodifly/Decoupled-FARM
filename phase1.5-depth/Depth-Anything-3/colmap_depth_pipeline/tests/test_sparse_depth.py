"""Unit tests for sparse depth helpers (synthetic)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from colmap_io import ColmapModel, FaceEntry  # noqa: E402
from sparse_depth import get_sparse_depth  # noqa: E402


def _make_face(image_id=1, face_id=0, frame_id="f0.jpg"):
    E = np.eye(4)
    K = np.array([[100.0, 0, 32.0], [0, 100.0, 32.0], [0, 0, 1.0]])
    return FaceEntry(
        frame_id=frame_id,
        face_id=face_id,
        image_id=image_id,
        image_name=f"pano_camera{face_id}/{frame_id}",
        image_path=Path("dummy"),
        extrinsics=E,
        intrinsics=K,
        camera_id=1,
        width=64,
        height=64,
        qvec=np.array([1.0, 0, 0, 0]),
        tvec=np.zeros(3),
    )


def test_get_sparse_depth_filters():
    face = _make_face()
    # point with good track
    p_good = SimpleNamespace(
        xyz=np.array([0.0, 0.0, 5.0]),
        error=0.5,
        image_ids=[1, 2, 3, 4],
    )
    # point with short track
    p_bad = SimpleNamespace(
        xyz=np.array([1.0, 0.0, 5.0]),
        error=0.5,
        image_ids=[1],
    )
    image = SimpleNamespace(
        xys=np.array([[32.0, 32.0], [40.0, 32.0]]),
        point3D_ids=np.array([10, 11]),
    )
    model = ColmapModel(
        sparse_dir=Path("."),
        images_dir=Path("."),
        faces=[face],
        frames={face.frame_id: [face]},
        cameras={},
        images={1: image},
        points3D={10: p_good, 11: p_bad},
    )
    coords, depths = get_sparse_depth(model, face, min_track_length=3, max_reproj_error=4.0)
    assert len(depths) == 1
    assert abs(depths[0] - 5.0) < 1e-6
    assert abs(coords[0, 0] - 32.0) < 1e-6
