"""Swappable per-frame depth interface.

This is the contract the colleague's DL depth model should implement.
The COLMAP patch-match stereo placeholder is one backend behind the same type.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .paths import dl_depth_search_roots

# Sentinel used in float32 depth arrays for "no measurement".
# NaN is also accepted by downstream filters. Zero is treated as invalid.
INVALID_DEPTH = 0.0


@dataclass(frozen=True)
class DepthMap:
    """Metric (or SfM-scale) depth aligned to an RGB frame.

    Attributes
    ----------
    depth_m:
        ``(H, W)`` float32. Values are **along the camera optical axis** (OpenCV
        ``Z``), in the same unit system as ``T_world_cam`` translation.
        For the COLMAP MVS placeholder this is SfM units (not necessarily metres).
        For the colleague DL model this should be metres, and poses must then
        also be metric.
    valid_mask:
        ``(H, W)`` bool. True where depth may be used. Invalid pixels may also
        be encoded as ``0``, ``NaN``, or non-finite values in ``depth_m``.
    frame_hw:
        ``(H, W)`` of the RGB frame this map is registered to. Must match
        ``depth_m.shape``. If the depth network outputs a different resolution,
        resample onto the RGB grid *before* constructing this object.
    units:
        Human-readable unit label. ``"m"`` for metric metres, ``"sfm"`` for
        COLMAP's arbitrary reconstruction scale.
    source:
        Backend id, e.g. ``"colmap_mvs"`` or ``"dl_depth_v1"``.
    invalid_value:
        Canonical fill used when writing files. Readers must still treat 0/NaN
        as invalid regardless of this field.
    """

    depth_m: np.ndarray
    valid_mask: np.ndarray
    frame_hw: tuple[int, int]
    units: str = "m"
    source: str = "unknown"
    invalid_value: float = INVALID_DEPTH

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth_m)
        mask = np.asarray(self.valid_mask)
        if depth.ndim != 2:
            raise ValueError(f"depth_m must be HxW, got {depth.shape}")
        if mask.shape != depth.shape:
            raise ValueError(f"valid_mask shape {mask.shape} != depth {depth.shape}")
        if tuple(int(x) for x in depth.shape) != tuple(int(x) for x in self.frame_hw):
            raise ValueError(
                f"frame_hw {self.frame_hw} does not match depth shape {depth.shape}"
            )
        object.__setattr__(self, "depth_m", depth.astype(np.float32, copy=False))
        object.__setattr__(self, "valid_mask", mask.astype(bool, copy=False))

    @property
    def height(self) -> int:
        return int(self.frame_hw[0])

    @property
    def width(self) -> int:
        return int(self.frame_hw[1])

    def validity(self) -> np.ndarray:
        """Combine explicit mask with numeric sentinels."""
        z = self.depth_m
        return self.valid_mask & np.isfinite(z) & (z > 0.0)

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            depth_m=self.depth_m,
            valid_mask=self.valid_mask.astype(np.uint8),
            frame_hw=np.asarray(self.frame_hw, dtype=np.int32),
            units=np.asarray(self.units),
            source=np.asarray(self.source),
            invalid_value=np.float32(self.invalid_value),
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "DepthMap":
        with np.load(path, allow_pickle=False) as data:
            hw = tuple(int(x) for x in data["frame_hw"].tolist())
            units = str(data["units"])
            source = str(data["source"])
            invalid = float(data["invalid_value"]) if "invalid_value" in data.files else INVALID_DEPTH
            return cls(
                depth_m=data["depth_m"],
                valid_mask=data["valid_mask"].astype(bool),
                frame_hw=(hw[0], hw[1]),
                units=units,
                source=source,
                invalid_value=invalid,
            )


@runtime_checkable
class DepthSource(Protocol):
    """Per-frame depth provider. Swap COLMAP MVS ↔ DL model here."""

    source_id: str
    units: str

    def depth_for_frame(self, frame_name: str) -> DepthMap:
        """Return a depth map registered to the RGB frame named ``frame_name``.

        ``frame_name`` is the COLMAP image name (e.g. ``frame_000123.jpg``),
        not a filesystem path. Implementations may map that to a file internally.
        """
        ...


# Optional extra roots via DL_DEPTH_ROOT / FARM_DL_DEPTH_ROOT, plus in-repo dirs.
_DL_DEPTH_NAME_HINTS = (
    "dl_depth",
    "dl-depth",
    "metric_depth",
    "metric-depth",
    "depth_anything",
    "zoedepth",
    "metric3d",
    "depth_v1",
)


def probe_dl_depth_v1() -> dict:
    """Check whether a deployable DL metric-depth model is present.

    Looks for a checkpoint + inference entrypoint. Returns a structured report;
    does **not** fall back to COLMAP MVS.
    """
    hits: list[str] = []
    for root in dl_depth_search_roots():
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            rel = str(path)
            if any(h in name or h in rel.lower() for h in _DL_DEPTH_NAME_HINTS):
                if path.suffix.lower() in {".pt", ".pth", ".onnx", ".ckpt", ".safetensors", ".py"}:
                    # Skip our own contract / docs / unrelated 3DGS EXR dumps
                    if "farm_object_map" in rel and path.suffix == ".py":
                        continue
                    if "vanilla_3dgs" in rel or "dl-3dgs" in rel.lower():
                        continue
                    hits.append(rel)
    return {
        "available": False,
        "source": "dl_depth_v1",
        "units": "m",
        "hits": hits[:50],
        "blocker": (
            "No colleague DL metric-depth checkpoint + inference script found in "
            "this workspace. Mapping must not silently fall back to COLMAP MVS."
        ),
    }


def estimate_sfm_to_metric_scale(
    z_sfm: np.ndarray,
    z_metric: np.ndarray,
    *,
    min_pairs: int = 50,
) -> dict:
    """Robust scale ``s`` such that ``z_metric ≈ s * z_sfm`` (optical-axis Z).

    COLMAP monocular poses/MVS live in an arbitrary SfM scale. Independently
    trained metric depth does **not** share that scale. Apply ``s`` to
    ``T_world_cam[:3, 3]`` (and any SfM-unit points) before combining with
    ``source='dl_depth_v1'`` depth, **or** divide metric depth by ``s`` to stay
    in SfM units — pick one space and stay there.

    ``z_sfm`` / ``z_metric`` are 1-D arrays of paired valid depths (e.g. COLMAP
    sparse point camera-Z vs DL depth at the same pixel).
    """
    z_sfm = np.asarray(z_sfm, dtype=np.float64).reshape(-1)
    z_metric = np.asarray(z_metric, dtype=np.float64).reshape(-1)
    if z_sfm.shape != z_metric.shape:
        raise ValueError(f"scale pair length mismatch {z_sfm.shape} vs {z_metric.shape}")
    ok = np.isfinite(z_sfm) & np.isfinite(z_metric) & (z_sfm > 1e-6) & (z_metric > 1e-6)
    z_sfm, z_metric = z_sfm[ok], z_metric[ok]
    if z_sfm.size < min_pairs:
        return {
            "ok": False,
            "reason": "too_few_pairs",
            "n_pairs": int(z_sfm.size),
            "min_pairs": min_pairs,
            "scale": None,
        }
    ratios = z_metric / z_sfm
    scale = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios - scale)))
    return {
        "ok": True,
        "n_pairs": int(z_sfm.size),
        "scale": scale,
        "scale_mad": mad,
        "note": "Multiply SfM translations by `scale` to enter metric metres, then use DL depth as-is.",
    }


def apply_scale_to_poses(T_world_cam: np.ndarray, scale: float) -> np.ndarray:
    """Scale camera-to-world translation; rotation unchanged."""
    out = np.asarray(T_world_cam, dtype=np.float64).copy()
    out[:3, 3] *= float(scale)
    return out
