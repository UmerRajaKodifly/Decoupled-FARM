"""Phase 2 runner: Detect-Segment-Embed.

Reads the frames.json produced by the DA3 depth-reconstruction pipeline,
runs open-vocabulary instance segmentation (YOLOE + MobileCLIP vocab) and
DINOv3 mask-pool feature extraction on each face tile, computes per-detection
3-D Gaussians from metric depth, transforms them into the shared world frame
using Stella VSLAM poses, and saves one detection-pack .pt file per keyframe.

This is a standalone adaptation of FARM's pipeline/steps.segment_and_transform
and utils/geometry.transform_segmentation_to_world, operating on 360-cubemap
face tiles rather than robot RGB-D frames.

Usage
-----
    python phase2_runner.py \\
        --frames-json /path/to/da3_scan_depth/frames_json/frames.json \\
        --data-root   /path/to/da3_scan_depth \\
        --output-dir  ./output \\
        [--device cuda] [--conf 0.35] [--batch-size 4] [--stride 1]

Outputs
-------
  <output-dir>/detections_kf<NNNNNN>.pt  — one file per keyframe
  <output-dir>/phase2_summary.json        — high-level run stats

Each .pt is a dict (see Phase2Output below) that Phase 3 loads directly.

Input data contract
-------------------
frames.json entry fields (DA3 naming):
    rgb           : str   relative path to 504×504 JPEG face image
    depth         : str   relative path to 504×504 float32 .npy (metric metres)
    K             : 3×3   pinhole intrinsics [[fx,0,cx],[0,fy,cy],[0,0,1]]
    T_wc          : 4×4   camera-to-world (Stella VSLAM pose, metric)
    depth_encoding: str   must be "float32_m"
    camera        : str   "face"

Frame ordering: 1280 entries = 320 keyframes × 4 faces (face0..face3).
Keyframe id is extracted from the rgb filename: kfNNNNNN_faceN.jpg.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Resolve paths and imports
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
_COMMON = _HERE.parent.parent / "common"
if _COMMON.is_dir() and str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))
from paths import apply_model_env, ensure_sys_path, resolve_models_dir  # noqa: E402
from pipeline_logger import (  # noqa: E402
    log_elapsed,
    log_stage_banner,
    setup_logger,
    tqdm_kwargs,
)

_FARM_ROOT = ensure_sys_path(_HERE)
_MODELS_DIR = apply_model_env(resolve_models_dir(_FARM_ROOT))

from segmenter import ConstructionSegmenter   # noqa: E402 (local)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **_kw):  # type: ignore
        return x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_rgb(path: Path) -> torch.Tensor:
    """Load a JPEG/PNG face image → (H, W, 3) uint8 tensor."""
    img = Image.open(path).convert("RGB")
    return torch.from_numpy(np.array(img, dtype=np.uint8))


def _load_depth(path: Path) -> torch.Tensor:
    """Load a float32 .npy depth map → (H, W) float32 tensor (metres)."""
    arr = np.load(str(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]   # (H, W, 1) → (H, W)
    return torch.from_numpy(arr)


def _parse_k(k_raw) -> torch.Tensor:
    """Accept 3×3 list-of-lists or flat list → (3, 3) float32 tensor."""
    k = np.array(k_raw, dtype=np.float32)
    if k.shape == (9,):
        k = k.reshape(3, 3)
    return torch.from_numpy(k)


def _parse_pose(t_raw) -> torch.Tensor:
    """Accept 4×4 or 3×4 list-of-lists → (4, 4) float32 tensor."""
    t = np.array(t_raw, dtype=np.float32)
    if t.shape == (12,):
        t = t.reshape(3, 4)
    if t.shape == (3, 4):
        bottom = np.array([[0, 0, 0, 1]], dtype=np.float32)
        t = np.vstack([t, bottom])
    return torch.from_numpy(t)


def _kf_id_from_rgb(rgb_name: str) -> str:
    """Extract keyframe id string from 'kfNNNNNN_faceN.jpg' → 'kfNNNNNN'."""
    return rgb_name.split("_face")[0]


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def run_phase2(
    frames_json: Path,
    data_root: Path,
    output_dir: Path,
    device: str = "cuda",
    conf: float = 0.35,
    batch_size: int = 4,
    stride: int = 1,
    vocab_file: Optional[Path] = None,
    detector: str = "yoloe",
) -> Path:
    """Full Phase 2 pipeline.

    Parameters
    ----------
    frames_json : Path
        frames.json from DA3 pipeline.
    data_root : Path
        Root directory relative to which rgb/depth paths in frames.json resolve.
    output_dir : Path
        Where to write detections_kf*.pt and phase2_summary.json.
    device : str
        'cuda' or 'cpu'.
    conf : float
        YOLOE confidence threshold.
    batch_size : int
        Number of face tiles to forward through YOLOE in one call.
        4 = one full keyframe (all faces) — recommended default.
    stride : int
        Process every Nth keyframe. 1 = all.
    vocab_file : Path | None
        Override vocabulary file.

    Returns
    -------
    Path
        Path to phase2_summary.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load frame index
    # ------------------------------------------------------------------
    with open(frames_json) as f:
        index = json.load(f)
    all_frames: List[dict] = index["frames"]

    # Group by keyframe id (preserving order)
    kf_order: List[str] = []
    kf_groups: Dict[str, List[dict]] = defaultdict(list)
    for entry in all_frames:
        kf_id = _kf_id_from_rgb(entry["rgb"])
        if kf_id not in kf_groups:
            kf_order.append(kf_id)
        kf_groups[kf_id].append(entry)

    # Apply stride
    kf_order = kf_order[::stride]
    print(f"[phase2] keyframes to process: {len(kf_order)} "
          f"(total={len(kf_groups)}, stride={stride}, detector={detector})")

    # ------------------------------------------------------------------
    # 2. Build segmenter
    # ------------------------------------------------------------------
    t0 = time.time()
    detector = (detector or "yoloe").strip().lower()
    if detector == "sam3":
        from sam3_segmenter import SAM3Segmenter
        print("[phase2] loading SAM3 + DINOv3 …")
        segmenter = SAM3Segmenter(
            vocab_file=vocab_file,
            device=device,
            conf=conf,
        )
    else:
        print("[phase2] loading YOLOE + DINOv3 …")
        segmenter = ConstructionSegmenter(
            vocab_file=vocab_file,
            device=device,
            conf=conf,
        )
    print(f"[phase2] models loaded in {time.time()-t0:.1f}s  "
          f"(vocab={len(segmenter.names)} labels, feature_dim={segmenter.feature_dim})")

    # ------------------------------------------------------------------
    # 3. Process keyframes
    # ------------------------------------------------------------------
    # Also import FARM's pose-aware normaliser for the output dict
    try:
        from scene_graph.map_update.filtering import normalize_seg_outputs
        _have_farm_normalise = True
    except ImportError:
        _have_farm_normalise = False

    stats = {"keyframes": 0, "detections_total": 0, "skipped_kf": 0}
    log = setup_logger("phase2", stage="phase2")
    log_stage_banner(log, "Phase 2 — Detect / Segment / Embed")
    t0 = time.time()

    for kf_id in tqdm(kf_order, **tqdm_kwargs(desc="Phase 2 detect")):
        faces = kf_groups[kf_id]
        out_path = output_dir / f"detections_{kf_id}.pt"

        if out_path.exists():
            log.info("skip %s (already exists)", kf_id)
            stats["skipped_kf"] += 1
            continue

        # ---- load all face tiles for this keyframe --------------------
        colors: List[torch.Tensor] = []
        depths: List[torch.Tensor] = []
        intrinsics: List[torch.Tensor] = []
        poses: List[torch.Tensor] = []
        face_meta: List[dict] = []

        for entry in faces:
            rgb_name = entry["rgb"]
            dep_name = entry["depth"]
            # Resolve DA3 layout variants:
            #   data_root/faces/*.jpg + data_root/depth/*.npy
            #   frames_json/images/*.jpg + frames_json/depths/*.npy
            #   data_root/*.jpg (flat)
            rgb_candidates = [
                Path(rgb_name) if Path(rgb_name).is_absolute() else None,
                data_root / rgb_name,
                data_root / "faces" / rgb_name,
                data_root / "images" / rgb_name,
                frames_json.parent / rgb_name,
                frames_json.parent / "images" / rgb_name,
                frames_json.parent / "faces" / rgb_name,
            ]
            dep_candidates = [
                Path(dep_name) if Path(dep_name).is_absolute() else None,
                data_root / dep_name,
                data_root / "depth" / dep_name,
                data_root / "depths" / dep_name,
                frames_json.parent / dep_name,
                frames_json.parent / "depths" / dep_name,
                frames_json.parent / "depth" / dep_name,
            ]
            rgb_path = next((p for p in rgb_candidates if p is not None and p.exists()), None)
            dep_path = next((p for p in dep_candidates if p is not None and p.exists()), None)
            if rgb_path is None or dep_path is None:
                raise FileNotFoundError(
                    f"Could not resolve rgb/depth for {kf_id}: "
                    f"rgb={rgb_name!r} depth={dep_name!r} (data_root={data_root})"
                )

            rgb_t = _load_rgb(rgb_path)
            dep_t = _load_depth(dep_path)
            k_t = _parse_k(entry["K"])
            # frames.json uses T_wc; FARM convention is T_world_cam — same thing
            pose_t = _parse_pose(entry["T_wc"])

            colors.append(rgb_t)
            depths.append(dep_t)
            intrinsics.append(k_t)
            poses.append(pose_t)
            face_meta.append({
                "rgb": str(rgb_path),
                "depth": str(dep_path),
                "timestamp": entry.get("timestamp"),
                "face_id": entry.get("camera", "face"),
            })

        if not colors:
            continue

        # ---- run segmentation + depth-unproject + DINOv3 features ----
        # This mirrors FARM's pipeline/steps.segment_and_transform:
        #   seg_outputs = segmenter(colors, depths, intrinsics)
        #   transform_segmentation_to_world(seg_outputs, poses_world)
        #
        # YOLOESegmenter internally handles:
        #   - letterbox resize (YOLOE) + NMS
        #   - mask erosion + depth-mode MAD filter
        #   - Mahalanobis outlier reject
        #   - compute_weighted_stats → means (cam) + cov6 (cam)
        #   - mask-pooled DINOv3 features
        # After the call we apply the world transform.

        seg_out = segmenter(colors, depths, intrinsics)

        if _have_farm_normalise:
            from scene_graph.map_update.filtering import normalize_seg_outputs
            dev = getattr(segmenter, "device", torch.device("cpu"))
            seg_out = normalize_seg_outputs(seg_out, fallback_device=dev)

        # World transform: apply T_wc per detection via batch_ids
        # (identical to FARM's transform_segmentation_to_world)
        _apply_world_transform(seg_out, poses)

        # Attach metadata for Phase 3 consumers
        seg_out["intrinsics"] = [k.cpu() for k in intrinsics]
        seg_out["poses_world"] = [p.cpu() for p in poses]
        seg_out["face_meta"] = face_meta
        seg_out["kf_id"] = kf_id
        seg_out["vocab"] = segmenter.names

        # Move heavy tensors to CPU before saving
        for key in ("means", "cov6", "features", "scores", "class_ids", "batch_ids",
                    "boxes_xyxy", "num_pixels", "det_points_flat", "det_points_offsets"):
            if key in seg_out and isinstance(seg_out[key], torch.Tensor):
                seg_out[key] = seg_out[key].cpu()
        if "masks" in seg_out and isinstance(seg_out["masks"], (list, tuple)):
            seg_out["masks"] = [m.cpu() if isinstance(m, torch.Tensor) else m
                                for m in seg_out["masks"]]

        torch.save(seg_out, out_path)

        n_det = int(seg_out.get("means", torch.empty(0)).shape[0])
        stats["keyframes"] += 1
        stats["detections_total"] += n_det
        log.info("%s: %d detections → %s", kf_id, n_det, out_path.name)

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    summary = {
        **stats,
        "frames_json": str(frames_json),
        "output_dir": str(output_dir),
        "conf": conf,
        "stride": stride,
        "vocab_size": len(segmenter.names),
        "feature_dim": segmenter.feature_dim,
        "detector": detector,
    }
    summary_path = output_dir / "phase2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(
        "done — %d keyframes, %d total detections",
        stats["keyframes"], stats["detections_total"],
    )
    log.info("summary → %s", summary_path)
    log_elapsed(log, t0, "Phase 2")
    return summary_path


# ---------------------------------------------------------------------------
# World transform helper
# (mirrors FARM utils/geometry.transform_segmentation_to_world inline
#  so the runner has no hard dependency on the FARM package at import time)
# ---------------------------------------------------------------------------

def _apply_world_transform(seg_out: dict, poses: List[torch.Tensor]) -> None:
    """In-place world transform of means and cov6 using batch_ids + poses list."""
    means = seg_out.get("means")
    cov6 = seg_out.get("cov6")
    batch_ids = seg_out.get("batch_ids")

    if means is None or cov6 is None or batch_ids is None:
        return
    if means.numel() == 0 or not poses:
        return

    from geometry import cov6_to_matrix, matrix_to_cov6  # noqa: E402 (local)

    device, dtype = means.device, means.dtype
    batch_ids_long = batch_ids.long()
    means_w = means.clone()
    cov6_w = cov6.clone()

    for idx in torch.unique(batch_ids_long).tolist():
        if idx < 0 or idx >= len(poses):
            continue
        mask = batch_ids_long == idx
        if not mask.any():
            continue
        pose = poses[idx].to(device=device, dtype=dtype)
        R, t = pose[:3, :3], pose[:3, 3]

        m_sub = means[mask]
        if m_sub.numel() > 0:
            means_w[mask] = m_sub @ R.T + t

        c_sub = cov6[mask]
        if c_sub.numel() > 0:
            cov_mats = cov6_to_matrix(c_sub)
            rot = R.unsqueeze(0).expand(cov_mats.shape[0], -1, -1)
            cov_w = rot @ cov_mats @ rot.transpose(1, 2)
            cov6_w[mask] = matrix_to_cov6(cov_w)

    seg_out["means"] = means_w
    seg_out["cov6"] = cov6_w

    flat = seg_out.get("det_points_flat")
    offs = seg_out.get("det_points_offsets")
    if (
        isinstance(flat, torch.Tensor)
        and flat.numel() > 0
        and isinstance(offs, torch.Tensor)
        and offs.numel() > 1
    ):
        points_w = flat.clone()
        for det_i in range(int(offs.numel()) - 1):
            s = int(offs[det_i].item())
            e = int(offs[det_i + 1].item())
            if e <= s:
                continue
            b = int(batch_ids_long[det_i].item()) if det_i < batch_ids_long.numel() else -1
            if b < 0 or b >= len(poses):
                continue
            pose = poses[b].to(device=device, dtype=dtype)
            R, t = pose[:3, :3], pose[:3, 3]
            pts = flat[s:e].to(device=device, dtype=dtype)
            points_w[s:e] = pts @ R.T + t
        seg_out["det_points_flat"] = points_w


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2: Detect-Segment-Embed for construction-site 360 imagery."
    )
    p.add_argument(
        "--frames-json", required=True, type=Path,
        help="Path to frames.json from DA3 depth-reconstr pipeline.",
    )
    p.add_argument(
        "--data-root", type=Path, default=None,
        help=(
            "Root directory for resolving relative rgb/depth paths. "
            "Defaults to the directory containing frames.json."
        ),
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent / "output",
        help="Where to write detections_kf*.pt files (default: ./output/).",
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--conf", type=float, default=0.35,
        help="YOLOE confidence threshold (default 0.35, matches FARM live path).",
    )
    p.add_argument(
        "--batch-size", type=int, default=4,
        help=(
            "Number of face tiles per segmentation call. "
            "4 = one full keyframe (all faces). Reduce to 1 if OOM."
        ),
    )
    p.add_argument(
        "--stride", type=int, default=1,
        help="Process every Nth keyframe (default 1 = all).",
    )
    p.add_argument(
        "--vocab", type=Path, default=None,
        help="Override vocabulary file (default: vocab/construction_vocab.txt).",
    )
    p.add_argument(
        "--detector", default="yoloe", choices=["yoloe", "sam3"],
        help="Phase 2 detector: yoloe (default) or sam3. Cuboid faces are used in both cases.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    frames_json = args.frames_json.resolve()
    data_root = args.data_root.resolve() if args.data_root else frames_json.parent.parent

    run_phase2(
        frames_json=frames_json,
        data_root=data_root,
        output_dir=args.output_dir.resolve(),
        device=args.device,
        conf=args.conf,
        batch_size=args.batch_size,
        stride=args.stride,
        vocab_file=args.vocab,
        detector=args.detector,
    )
