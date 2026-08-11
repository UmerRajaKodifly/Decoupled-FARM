"""Depth I/O and simple visualization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np


def face_folder(root: Path, face_id: int) -> Path:
    """Per-face subdirectory: ``face_depth/face0``, ``face_conf/face1``, …"""
    return Path(root) / f"face{int(face_id)}"


def face_depth_path(
    face_depth_dir: Path,
    stem: str,
    face_id: int,
    *,
    raw: bool = False,
) -> Path:
    """``face_depth/face{id}/{stem}.npy`` (or ``{stem}_raw.npy``)."""
    name = f"{stem}_raw.npy" if raw else f"{stem}.npy"
    return face_folder(face_depth_dir, face_id) / name


def face_conf_path(face_conf_dir: Path, stem: str, face_id: int) -> Path:
    return face_folder(face_conf_dir, face_id) / f"{stem}_conf.npy"


def face_sky_path(face_sky_dir: Path, stem: str, face_id: int) -> Path:
    return face_folder(face_sky_dir, face_id) / f"{stem}_sky.npy"


def face_vis_path(face_vis_dir: Path, stem: str, face_id: int) -> Path:
    return face_folder(face_vis_dir, face_id) / f"{stem}.png"


def resolve_face_depth_path(
    face_depth_dir: Path,
    stem: str,
    face_id: int,
    *,
    raw: bool = False,
) -> Path | None:
    """Prefer new layout; fall back to legacy flat ``{stem}_face{id}.npy``."""
    new = face_depth_path(face_depth_dir, stem, face_id, raw=raw)
    if new.is_file():
        return new
    legacy = face_depth_dir / (f"{stem}_face{face_id}_raw.npy" if raw else f"{stem}_face{face_id}.npy")
    if legacy.is_file():
        return legacy
    return None


def resolve_face_conf_path(face_conf_dir: Path, stem: str, face_id: int) -> Path | None:
    new = face_conf_path(face_conf_dir, stem, face_id)
    if new.is_file():
        return new
    legacy = face_conf_dir / f"{stem}_face{face_id}_conf.npy"
    if legacy.is_file():
        return legacy
    return None


def resolve_face_sky_path(face_sky_dir: Path, stem: str, face_id: int) -> Path | None:
    new = face_sky_path(face_sky_dir, stem, face_id)
    if new.is_file():
        return new
    legacy = face_sky_dir / f"{stem}_face{face_id}_sky.npy"
    if legacy.is_file():
        return legacy
    return None


def save_depth(path: Path, depth: np.ndarray, fmt: str = "npy") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()
    if fmt == "npy":
        out = path if path.suffix == ".npy" else path.with_suffix(".npy")
        np.save(out, depth.astype(np.float32))
        return out
    if fmt == "npz":
        out = path if path.suffix == ".npz" else path.with_suffix(".npz")
        np.savez_compressed(out, depth=depth.astype(np.float32))
        return out
    if fmt in {"png16", "png"}:
        import cv2

        out = path if path.suffix == ".png" else path.with_suffix(".png")
        d = depth.astype(np.float64)
        valid = np.isfinite(d) & (d > 0)
        enc = np.zeros(d.shape, dtype=np.uint16)
        # Store depth in millimeters, clipped
        mm = np.clip(d * 1000.0, 0, 65535)
        enc[valid] = mm[valid].astype(np.uint16)
        cv2.imwrite(str(out), enc)
        return out
    raise ValueError(f"Unknown depth format: {fmt}")


def load_depth(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    if path.suffix == ".npz":
        return np.load(path)["depth"]
    if path.suffix == ".png":
        import cv2

        enc = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        d = enc.astype(np.float32) / 1000.0
        d[enc == 0] = np.nan
        return d
    raise ValueError(f"Unsupported depth file: {path}")


def depth_to_vis(depth: np.ndarray, percentile: tuple[float, float] = (2, 98)) -> np.ndarray:
    """Return uint8 turbo-like grayscale visualization."""
    import cv2

    d = depth.astype(np.float64)
    valid = np.isfinite(d) & (d > 0)
    vis = np.zeros((*d.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return vis
    lo, hi = np.percentile(d[valid], percentile)
    hi = max(hi, lo + 1e-6)
    norm = np.clip((d - lo) / (hi - lo), 0, 1)
    norm[~valid] = 0
    gray = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def save_depth_vis(path: Path, depth: np.ndarray) -> Path:
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vis = depth_to_vis(depth)
    cv2.imwrite(str(path), vis)
    return path


def save_conf(path: Path, conf: np.ndarray, fmt: str = "npy") -> Path:
    """Save a confidence map (float)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()
    if fmt == "npy":
        out = path if path.suffix == ".npy" else path.with_suffix(".npy")
        np.save(out, conf.astype(np.float32))
        return out
    if fmt == "npz":
        out = path if path.suffix == ".npz" else path.with_suffix(".npz")
        np.savez_compressed(out, conf=conf.astype(np.float32))
        return out
    raise ValueError(f"Unknown conf format: {fmt}")


def conf_to_vis(conf: np.ndarray, percentile: tuple[float, float] = (2, 98)) -> np.ndarray:
    """Visualize confidence with a perceptually clear colormap."""
    import cv2

    c = conf.astype(np.float64)
    valid = np.isfinite(c)
    vis = np.zeros((*c.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return vis
    lo, hi = np.percentile(c[valid], percentile)
    hi = max(hi, lo + 1e-6)
    norm = np.clip((c - lo) / (hi - lo), 0, 1)
    norm[~valid] = 0
    gray = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)


def save_conf_vis(path: Path, conf: np.ndarray) -> Path:
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), conf_to_vis(conf))
    return path


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
