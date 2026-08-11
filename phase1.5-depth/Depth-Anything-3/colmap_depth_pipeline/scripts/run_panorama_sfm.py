#!/usr/bin/env python3
"""Thin host CLI: run panorama SfM inside the CASPAR-capable COLMAP docker image.

Accepts either a frames directory (``--pano_dir``) or an equirect video
(``--video``); the latter is extracted via ffmpeg before SfM.

Default image: ``gcr.io/spatialsense/spatialsense-3dgs-job:3dgsbase1.5``
(override via ``--colmap_image``, ``COLMAP_DOCKER_IMAGE``, or config).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from colmap_docker import COLMAP_IMAGE, run_python, set_colmap_image  # noqa: E402
from video_extract import (  # noqa: E402
    DEFAULT_EXTRACT_FPS,
    resolve_pano_input,
    video_defaults_from_config,
)


def _load_cfg(config: Path | None) -> dict:
    cfg_path = config or (_ROOT / "configs" / "default.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pano_dir", type=Path, help="Input equirect frames directory")
    src.add_argument("--video", type=Path, help="Input equirect video (ffmpeg extract, then SfM)")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML for video extract / colmap image defaults (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=f"Frame extract FPS when using --video (default: config / {DEFAULT_EXTRACT_FPS})",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Cap extracted frames when using --video (default: config / no cap)",
    )
    parser.add_argument(
        "--frames_dir",
        type=Path,
        default=None,
        help="Where to write extracted frames (default: <video>_frames_<fps>fps)",
    )
    parser.add_argument(
        "--overwrite_frames",
        action="store_true",
        help="Re-extract frames even if frames_dir already has images",
    )
    parser.add_argument("--out_dir", type=Path, required=True, help="COLMAP project output")
    parser.add_argument(
        "--pano_render_type",
        default="perspective_non_overlapping",
        choices=[
            "perspective_overlapping",
            "perspective_non_overlapping",
            "spherical",
        ],
    )
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--use_cpu", dest="use_gpu", action="store_false")
    parser.add_argument("--gpu_index", default="-1")
    parser.add_argument(
        "--ba_backend",
        default="caspar",
        choices=["caspar", "ceres"],
        help="Bundle adjustment backend (default: caspar; falls back to GPU Ceres)",
    )
    parser.add_argument(
        "--colmap_image",
        default=None,
        help="Docker image for SfM (default: config / SpatialSense 3dgsbase1.5)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    cfg = _load_cfg(args.config)
    vcfg = video_defaults_from_config(cfg)
    fps = float(args.fps if args.fps is not None else vcfg["fps"])
    max_frames = args.max_frames if args.max_frames is not None else vcfg["max_frames"]
    image = set_colmap_image(args.colmap_image or cfg.get("colmap_image") or COLMAP_IMAGE)

    pano_dir = resolve_pano_input(
        pano_dir=args.pano_dir,
        video=args.video,
        frames_out_dir=args.frames_dir,
        fps=fps,
        max_frames=max_frames,
        pattern=vcfg["pattern"],
        quality=vcfg["quality"],
        overwrite=args.overwrite_frames,
    )
    logging.info("Using panorama frames: %s", pano_dir)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    vendor = (_ROOT / "vendor" / "panorama").resolve()

    py_args = [
        "/pipeline/run_reconstruct.py",
        "--input_image_path",
        "/data/pano",
        "--output_path",
        "/data/out",
        "--pano_render_type",
        args.pano_render_type,
        "--gpu_index",
        str(args.gpu_index),
        "--ba_backend",
        args.ba_backend,
    ]
    if args.use_gpu:
        py_args.append("--use_gpu")
    else:
        py_args.append("--use_cpu")

    logging.info("Running panorama SfM via %s (ba_backend=%s) ...", image, args.ba_backend)
    run_python(
        py_args,
        mounts=[
            (pano_dir, "/data/pano"),
            (out_dir, "/data/out"),
            (vendor, "/pipeline"),
        ],
        use_gpu=args.use_gpu,
        ensure_imaging_deps=True,
        image=image,
    )
    logging.info("Done. Sparse model under %s", out_dir / "sparse")
    logging.info("Frames used: %s", pano_dir)


if __name__ == "__main__":
    main()
