"""Load COLMAP sparse models produced by panorama SfM (perspective faces)."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Allow importing DA3 read_write_model without installing the package editable quirks.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DA3_SRC = _REPO_ROOT / "src"
if str(_DA3_SRC) not in sys.path:
    sys.path.insert(0, str(_DA3_SRC))

from depth_anything_3.utils.read_write_model import (  # noqa: E402
    qvec2rotmat,
    read_model,
)

FACE_NAME_RE = re.compile(r"^pano_camera(\d+)/(.*)$")


@dataclass
class FaceEntry:
    frame_id: str
    face_id: int
    image_id: int
    image_name: str
    image_path: Path
    extrinsics: np.ndarray  # 4x4 world-to-camera
    intrinsics: np.ndarray  # 3x3
    camera_id: int
    width: int
    height: int
    qvec: np.ndarray
    tvec: np.ndarray


@dataclass
class ColmapModel:
    sparse_dir: Path
    images_dir: Path
    faces: List[FaceEntry]
    frames: Dict[str, List[FaceEntry]]
    cameras: dict
    images: dict
    points3D: dict
    face_meta: dict = field(default_factory=dict)

    @property
    def frame_ids(self) -> List[str]:
        return list(self.frames.keys())


def parse_face_image_name(name: str) -> Tuple[int, str]:
    """Parse `pano_camera{idx}/{pano_name}` -> (face_id, frame_id)."""
    m = FACE_NAME_RE.match(name)
    if not m:
        raise ValueError(f"Unexpected face image name (expected pano_cameraK/...): {name!r}")
    return int(m.group(1)), m.group(2)


def camera_to_K(camera) -> np.ndarray:
    if camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = camera.params
        fx = fy = float(f)
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
        fx, fy, cx, cy = float(fx), float(fy), float(cx), float(cy)
    else:
        fx = fy = float(camera.params[0])
        cx = camera.width / 2.0
        cy = camera.height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def image_to_extrinsic(image) -> np.ndarray:
    R = qvec2rotmat(image.qvec)
    t = np.asarray(image.tvec, dtype=np.float64)
    E = np.eye(4, dtype=np.float64)
    E[:3, :3] = R
    E[:3, 3] = t
    return E


def find_sparse_model_dir(colmap_dir: Path) -> Path:
    """
    Resolve sparse model directory.

    Accepts:
      - path already pointing at a model (has cameras.bin/txt)
      - project root with sparse/0
      - project root with sparse/ (single model)
    """
    colmap_dir = Path(colmap_dir)
    if (colmap_dir / "cameras.bin").exists() or (colmap_dir / "cameras.txt").exists():
        return colmap_dir
    sparse0 = colmap_dir / "sparse" / "0"
    if (sparse0 / "cameras.bin").exists() or (sparse0 / "cameras.txt").exists():
        return sparse0
    sparse = colmap_dir / "sparse"
    if (sparse / "cameras.bin").exists() or (sparse / "cameras.txt").exists():
        return sparse
    # Pick first numeric subdirectory
    if sparse.is_dir():
        subs = sorted(
            [p for p in sparse.iterdir() if p.is_dir()],
            key=lambda p: (not p.name.isdigit(), p.name),
        )
        for p in subs:
            if (p / "cameras.bin").exists() or (p / "cameras.txt").exists():
                return p
    raise FileNotFoundError(f"No COLMAP sparse model found under {colmap_dir}")


def load_face_meta(path: Path | None) -> dict:
    if path is None or not Path(path).is_file():
        return {}
    return json.loads(Path(path).read_text())


def write_face_meta(path: Path, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2))


def default_face_meta(
    num_faces: int = 4,
    num_steps_yaw: int = 4,
    pitches_deg: Sequence[float] = (0.0,),
    hfov_deg: float = 90.0,
    vfov_deg: float = 90.0,
) -> dict:
    from cubemap import get_virtual_rotations

    rotations = get_virtual_rotations(num_steps_yaw, pitches_deg)
    return {
        "render_type": "perspective_non_overlapping",
        "num_faces": num_faces,
        "num_steps_yaw": num_steps_yaw,
        "pitches_deg": list(pitches_deg),
        "hfov_deg": hfov_deg,
        "vfov_deg": vfov_deg,
        "face_prefixes": [f"pano_camera{i}/" for i in range(num_faces)],
        "cam_from_pano": [R.tolist() for R in rotations],
    }


def load_colmap_model(
    sparse_dir: str | Path,
    images_dir: str | Path | None = None,
    face_meta_path: str | Path | None = None,
    expected_faces: Optional[int] = 4,
) -> ColmapModel:
    sparse_dir = find_sparse_model_dir(Path(sparse_dir))
    project_dir = sparse_dir.parent
    if project_dir.name == "sparse":
        project_dir = project_dir.parent
    if images_dir is None:
        candidates = [
            project_dir / "images",
            sparse_dir.parent.parent / "images",
        ]
        images_dir = next((c for c in candidates if c.is_dir()), project_dir / "images")
    else:
        images_dir = Path(images_dir)

    cameras, images, points3D = read_model(str(sparse_dir))

    faces: List[FaceEntry] = []
    for image_id, image in images.items():
        face_id, frame_id = parse_face_image_name(image.name)
        cam = cameras[image.camera_id]
        faces.append(
            FaceEntry(
                frame_id=frame_id,
                face_id=face_id,
                image_id=image_id,
                image_name=image.name,
                image_path=images_dir / image.name,
                extrinsics=image_to_extrinsic(image),
                intrinsics=camera_to_K(cam),
                camera_id=image.camera_id,
                width=int(cam.width),
                height=int(cam.height),
                qvec=np.asarray(image.qvec, dtype=np.float64),
                tvec=np.asarray(image.tvec, dtype=np.float64),
            )
        )

    # Group by frame, sort frames by name, faces by face_id
    frames: Dict[str, List[FaceEntry]] = {}
    for face in faces:
        frames.setdefault(face.frame_id, []).append(face)
    for fid in frames:
        frames[fid] = sorted(frames[fid], key=lambda f: f.face_id)
    frames = dict(sorted(frames.items(), key=lambda kv: kv[0]))

    if expected_faces is not None:
        for frame_id, flist in frames.items():
            if len(flist) != expected_faces:
                # Allow incomplete registration but warn via exception only if empty
                if len(flist) == 0:
                    raise RuntimeError(f"Frame {frame_id} has no registered faces")

    meta_path = Path(face_meta_path) if face_meta_path else project_dir / "face_meta.json"
    face_meta = load_face_meta(meta_path)
    if not face_meta:
        face_meta = default_face_meta(num_faces=expected_faces or 4)

    # Flatten faces in frame order
    ordered_faces = [f for flist in frames.values() for f in flist]
    return ColmapModel(
        sparse_dir=sparse_dir,
        images_dir=images_dir,
        faces=ordered_faces,
        frames=frames,
        cameras=cameras,
        images=images,
        points3D=points3D,
        face_meta=face_meta,
    )


def reproject_point_to_face(
    xyz: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Project world point to pixel (u,v) and camera-frame depth z."""
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    Xc = R @ np.asarray(xyz, dtype=np.float64) + t
    z = float(Xc[2])
    u = intrinsics[0, 0] * Xc[0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * Xc[1] / z + intrinsics[1, 2]
    return np.array([u, v], dtype=np.float64), z
