#!/usr/bin/env python3
"""Extract equirect frames from a video with ffmpeg at a chosen FPS."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from video_extract import DEFAULT_EXTRACT_FPS, extract_frames_from_video  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_EXTRACT_FPS,
        help=f"Extract FPS (default: {DEFAULT_EXTRACT_FPS})",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Stop after this many extracted frames (default: no cap)",
    )
    parser.add_argument("--pattern", default="frame_%06d.jpg")
    parser.add_argument("--quality", type=int, default=2, help="JPEG qscale 2-31 (lower=better)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    frames = extract_frames_from_video(
        args.video,
        args.out_dir,
        fps=args.fps,
        max_frames=args.max_frames,
        pattern=args.pattern,
        quality=args.quality,
        overwrite=args.overwrite,
    )
    cap = f", max_frames={args.max_frames}" if args.max_frames else ""
    print(f"Extracted {len(frames)} frames at {args.fps:g} fps{cap} -> {args.out_dir}")


if __name__ == "__main__":
    main()
