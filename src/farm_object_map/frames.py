"""Extract video frames at a declared FPS. Never silently subsample."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def probe_video(video_path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration_s = (n / fps) if fps > 0 else 0.0
    return {
        "path": str(video_path),
        "fps": fps,
        "num_frames": n,
        "width": w,
        "height": h,
        "duration_s": duration_s,
        "estimated_extracted_at_2fps": int(duration_s * 2.0) if duration_s else 0,
    }


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    fps: float,
    image_ext: str = "jpg",
) -> list[Path]:
    """Extract frames at ``fps`` using ffmpeg. Returns sorted frame paths."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / f"frame_%06d.{image_ext}"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-qscale:v",
        "2",
        str(pattern),
    ]
    logger.info("Extracting frames at %.4g fps: %s", fps, " ".join(cmd))
    subprocess.run(cmd, check=True)
    frames = sorted(output_dir.glob(f"frame_*.{image_ext}"))
    logger.info("Extracted %d frames into %s", len(frames), output_dir)
    return frames
