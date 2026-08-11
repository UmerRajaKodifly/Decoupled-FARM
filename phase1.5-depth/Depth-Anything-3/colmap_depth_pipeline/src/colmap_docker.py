"""Docker helpers for COLMAP SfM (pycolmap + CASPAR-capable image by default)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Sequence

# CASPAR-enabled COLMAP build. Override with COLMAP_DOCKER_IMAGE or configs/default.yaml.
DEFAULT_COLMAP_IMAGE = "gcr.io/spatialsense/spatialsense-3dgs-job:3dgsbase1.5"
COLMAP_IMAGE = os.environ.get("COLMAP_DOCKER_IMAGE", DEFAULT_COLMAP_IMAGE)

# Imaging deps needed by vendored panorama.py but may be absent from the image.
_DOCKER_PIP_DEPS = ("opencv-python-headless", "pillow", "scipy", "tqdm")


def set_colmap_image(image: str | None) -> str:
    """Set the active COLMAP docker image for subsequent run_* calls."""
    global COLMAP_IMAGE
    if image:
        COLMAP_IMAGE = str(image)
    return COLMAP_IMAGE


def _docker_base(
    *,
    mounts: Sequence[tuple[Path, str]],
    use_gpu: bool = False,
    entrypoint: str | None = "colmap",
    image: str | None = None,
) -> list[str]:
    cmd = ["docker", "run", "--rm"]
    if use_gpu:
        cmd += ["--gpus", "all"]
    for host, container in mounts:
        cmd += ["-v", f"{Path(host).resolve()}:{container}"]
    if entrypoint is not None:
        cmd += ["--entrypoint", entrypoint]
    cmd.append(image or COLMAP_IMAGE)
    return cmd


def run_colmap(
    args: list[str],
    *,
    mounts: Sequence[tuple[Path, str]],
    use_gpu: bool = False,
    image: str | None = None,
) -> None:
    """Run `colmap <args>` inside the image with explicit mounts."""
    cmd = _docker_base(
        mounts=mounts, use_gpu=use_gpu, entrypoint="colmap", image=image
    )
    cmd += list(args)
    subprocess.run(cmd, check=True)


def run_python(
    args: list[str],
    *,
    mounts: Sequence[tuple[Path, str]],
    use_gpu: bool = False,
    ensure_imaging_deps: bool = False,
    image: str | None = None,
) -> None:
    """
    Run python3 inside the image.

    If ensure_imaging_deps is True, pip-installs opencv/pillow/scipy/tqdm when
    missing (needed for vendored panorama rendering). Uses
    ``--break-system-packages`` because modern images enforce PEP 668.
    """
    if ensure_imaging_deps:
        deps = " ".join(_DOCKER_PIP_DEPS)
        # Install only if import fails; allow PEP 668 images via --break-system-packages.
        inner = (
            "python3 -c 'import cv2,PIL,scipy,tqdm' 2>/dev/null "
            f"|| pip3 install -q --break-system-packages {deps}; "
            "python3 " + " ".join(args)
        )
        cmd = _docker_base(
            mounts=mounts, use_gpu=use_gpu, entrypoint="sh", image=image
        )
        cmd += ["-c", inner]
    else:
        cmd = _docker_base(
            mounts=mounts, use_gpu=use_gpu, entrypoint="python3", image=image
        )
        cmd += list(args)
    subprocess.run(cmd, check=True)


def model_converter(
    input_path: Path,
    output_path: Path,
    output_type: str = "TXT",
) -> None:
    """Convert a COLMAP model (binary <-> text) via Docker."""
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    # Mount a common ancestor so both paths are visible.
    common = Path(os_path_common([str(input_path), str(output_path)]))
    in_rel = input_path.relative_to(common)
    out_rel = output_path.relative_to(common)
    run_colmap(
        [
            "model_converter",
            "--input_path",
            f"/workspace/{in_rel.as_posix()}",
            "--output_path",
            f"/workspace/{out_rel.as_posix()}",
            "--output_type",
            output_type,
        ],
        mounts=[(common, "/workspace")],
    )


def model_analyzer(model_path: Path) -> None:
    """Print COLMAP model stats via Docker."""
    model_path = Path(model_path).resolve()
    parent = model_path.parent
    run_colmap(
        ["model_analyzer", "--path", f"/workspace/{model_path.name}"],
        mounts=[(parent, "/workspace")],
    )


def os_path_common(paths: Sequence[str]) -> str:
    import os as _os

    return _os.path.commonpath(list(paths))
