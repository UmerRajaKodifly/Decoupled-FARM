#!/usr/bin/env python3
"""Main CLI: COLMAP relative poses + DA3METRIC dense depth (meters).

``--video`` extracts equirect frames via ffmpeg before depth (same as SfM).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_REPO = _ROOT.parent
for p in (_SRC, _REPO / "src", _REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pipeline import run_pipeline  # noqa: E402
from video_extract import (  # noqa: E402
    DEFAULT_EXTRACT_FPS,
    resolve_pano_input,
    video_defaults_from_config,
)


def _load_video_cfg(config: Path | None) -> dict:
    cfg_path = config or (_ROOT / "configs" / "default.yaml")
    with open(cfg_path) as f:
        return video_defaults_from_config(yaml.safe_load(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap_dir", type=Path, required=True, help="Panorama SfM project dir")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pano_dir", type=Path, help="Original equirect frames")
    src.add_argument("--video", type=Path, help="Equirect video (ffmpeg extract, then depth)")
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
    parser.add_argument("--overwrite_frames", action="store_true")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--window_size", type=int, default=None, help="Frames per DA3 window")
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--save_format", default=None, choices=["npy", "npz", "png16"])
    parser.add_argument("--skip_da3", action="store_true", help="Reuse saved face depths")
    parser.add_argument(
        "--export_ply",
        action="store_true",
        help="Write pointcloud.ply (COLMAP K + poses) at end of run",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    vcfg = _load_video_cfg(args.config)
    fps = float(args.fps if args.fps is not None else vcfg["fps"])
    max_frames = args.max_frames if args.max_frames is not None else vcfg["max_frames"]

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

    run_pipeline(
        args.colmap_dir,
        pano_dir,
        args.out_dir,
        config_path=args.config,
        model_name=args.model_name,
        window_size=args.window_size,
        overlap=args.overlap,
        save_format=args.save_format,
        skip_da3=args.skip_da3,
        device=args.device,
        export_ply=args.export_ply,
    )


if __name__ == "__main__":
    main()
