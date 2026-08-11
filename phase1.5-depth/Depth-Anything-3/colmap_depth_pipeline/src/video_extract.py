"""Extract equirectangular frames from video via ffmpeg subprocess."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_EXTRACT_FPS = 2.0
DEFAULT_JPEG_QUALITY = 2
DEFAULT_FRAME_PATTERN = "frame_%06d.jpg"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

logger = logging.getLogger(__name__)


def ensure_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg to extract frames from video."
        )
    return path


def list_image_frames(directory: Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def video_defaults_from_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Pull extract settings from ``configs/default.yaml`` ``video:`` block."""
    v = (cfg or {}).get("video") or {}
    max_frames = v.get("max_frames", None)
    if max_frames is not None:
        max_frames = int(max_frames)
    return {
        "fps": float(v.get("extract_fps", DEFAULT_EXTRACT_FPS)),
        "max_frames": max_frames,
        "quality": int(v.get("jpeg_quality", DEFAULT_JPEG_QUALITY)),
        "pattern": str(v.get("frame_pattern", DEFAULT_FRAME_PATTERN)),
    }


def extract_frames_from_video(
    video_path: Path,
    out_dir: Path,
    *,
    fps: float = DEFAULT_EXTRACT_FPS,
    max_frames: int | None = None,
    pattern: str = DEFAULT_FRAME_PATTERN,
    quality: int = DEFAULT_JPEG_QUALITY,
    overwrite: bool = False,
) -> list[Path]:
    """
    Extract frames from ``video_path`` into ``out_dir`` at the given fps.

    Runs an ffmpeg subprocess with the ``fps`` filter. If ``max_frames`` is set,
    stops after that many output frames (``-frames:v``). Default quality=2 is
    high-quality JPEG (2–31 scale, lower is better). Returns sorted frame paths.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError(f"max_frames must be positive when set, got {max_frames}")

    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list_image_frames(out_dir)
    if existing and not overwrite:
        logger.info(
            "Reusing %d existing frames in %s (pass overwrite=True to re-extract)",
            len(existing),
            out_dir,
        )
        if max_frames is not None and len(existing) > max_frames:
            return existing[:max_frames]
        return existing

    if existing and overwrite:
        for p in existing:
            p.unlink()

    ffmpeg = ensure_ffmpeg()
    out_pattern = str(out_dir / pattern)
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        "-q:v",
        str(quality),
    ]
    if max_frames is not None:
        cmd += ["-frames:v", str(int(max_frames))]
    cmd.append(out_pattern)

    cap = f", max_frames={max_frames}" if max_frames is not None else ""
    logger.info(
        "Extracting frames via ffmpeg: %s -> %s (fps=%g%s)",
        video_path,
        out_dir,
        fps,
        cap,
    )
    logger.debug("ffmpeg cmd: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg failed (exit {e.returncode}) extracting frames from {video_path}"
        ) from e

    frames = list_image_frames(out_dir)
    if max_frames is not None and len(frames) > max_frames:
        for extra in frames[max_frames:]:
            extra.unlink()
        frames = frames[:max_frames]

    if not frames:
        raise RuntimeError(
            f"ffmpeg produced no frames from {video_path} at fps={fps} into {out_dir}"
        )

    meta = {
        "video_path": str(video_path),
        "fps": float(fps),
        "max_frames": max_frames,
        "n_frames": len(frames),
        "pattern": pattern,
        "quality": quality,
        "ffmpeg_cmd": cmd,
        "frames": [p.name for p in frames],
    }
    # Keep meta outside the frame folder — panorama SfM treats every file in
    # the input dir as an image (including .json).
    meta_path = out_dir.parent / f"{out_dir.name}_extract_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    # Remove a stale in-folder meta from earlier versions if present.
    stale = out_dir / "extract_meta.json"
    if stale.is_file():
        stale.unlink()
    logger.info("Extracted %d frames -> %s", len(frames), out_dir)
    return frames


def resolve_pano_input(
    *,
    pano_dir: Path | None = None,
    video: Path | None = None,
    frames_out_dir: Path | None = None,
    fps: float = DEFAULT_EXTRACT_FPS,
    max_frames: int | None = None,
    pattern: str = DEFAULT_FRAME_PATTERN,
    quality: int = DEFAULT_JPEG_QUALITY,
    overwrite: bool = False,
) -> Path:
    """
    Resolve a directory of equirect frames from either ``pano_dir`` or ``video``.

    If ``video`` is set, runs ffmpeg to extract frames under ``frames_out_dir``
    (default: ``<video_parent>/<stem>_frames_<fps>fps[_nN]``).
    """
    if video is not None and pano_dir is not None:
        raise ValueError("Pass only one of --pano_dir or --video")
    if video is None and pano_dir is None:
        raise ValueError("Provide --pano_dir or --video")

    if pano_dir is not None:
        pano_dir = Path(pano_dir)
        if not pano_dir.is_dir():
            raise FileNotFoundError(f"pano_dir not found: {pano_dir}")
        return pano_dir.resolve()

    video = Path(video)
    if frames_out_dir is None:
        suffix = f"{video.stem}_frames_{fps:g}fps"
        if max_frames is not None:
            suffix += f"_n{int(max_frames)}"
        frames_out_dir = video.resolve().parent / suffix
    frames = extract_frames_from_video(
        video,
        Path(frames_out_dir),
        fps=fps,
        max_frames=max_frames,
        pattern=pattern,
        quality=quality,
        overwrite=overwrite,
    )
    return frames[0].parent
