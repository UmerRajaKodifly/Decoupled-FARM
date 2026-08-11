#!/usr/bin/env python3
"""
Empirical sanity-check for DA3 prediction.conf.

DA3 docs/export use *percentile* thresholds (not calibrated probabilities).
This script visualizes conf on face images and reports whether the signal has
useful dynamic range / spatial structure vs looking flat.

Usage:
  # From saved face conf maps (after a pipeline run):
  python colmap_depth_pipeline/scripts/inspect_da3_conf.py \\
    --face_conf_dir path/to/output/face_conf \\
    --face_image_dir path/to/project/images \\
    --out_dir path/to/conf_inspect

  # Or run a tiny DA3 forward on a few face images:
  python colmap_depth_pipeline/scripts/inspect_da3_conf.py \\
    --face_images path/to/face0.jpg path/to/face1.jpg \\
    --model_name depth-anything/DA3-LARGE-1.1 \\
    --out_dir path/to/conf_inspect
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_REPO = _ROOT.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from io_utils import conf_to_vis, write_json  # noqa: E402
from scale_align import summarize_conf_map  # noqa: E402


def _texture_proxy(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return np.abs(lap)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 16:
        return float("nan")
    aa, bb = a[mask].ravel(), b[mask].ravel()
    if aa.std() < 1e-12 or bb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def analyze_pair(image_bgr: np.ndarray | None, conf: np.ndarray, name: str, out_dir: Path) -> dict:
    stats = summarize_conf_map(conf)
    stats["name"] = name
    if image_bgr is not None:
        H, W = conf.shape
        img = cv2.resize(image_bgr, (W, H), interpolation=cv2.INTER_AREA)
        tex = _texture_proxy(img)
        stats["corr_with_laplacian"] = _corr(conf, tex)
        # Side-by-side: image | conf
        conf_vis = conf_to_vis(conf)
        side = np.concatenate([img, conf_vis], axis=1)
        cv2.imwrite(str(out_dir / f"{name}_rgb_conf.png"), side)
        # Low-conf overlay (bottom 20% within this map)
        thr = np.percentile(conf[np.isfinite(conf)], 20)
        overlay = img.copy()
        low = conf <= thr
        overlay[low] = (0.4 * overlay[low] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(str(out_dir / f"{name}_lowconf_overlay.png"), overlay)
        stats["p20_thr"] = float(thr)
    else:
        cv2.imwrite(str(out_dir / f"{name}_conf.png"), conf_to_vis(conf))
        stats["corr_with_laplacian"] = None

    # Recommendation hint
    dr = stats.get("dynamic_range_ratio", 0.0) or 0.0
    corr = stats.get("corr_with_laplacian")
    if dr < 1e-3:
        stats["hint"] = "flat/uninformative — ignore conf (conf_mode=none)"
    elif corr is not None and corr > 0.15:
        stats["hint"] = (
            "conf varies and correlates with texture — soft weight OK "
            "(conf_mode=weight); avoid absolute thresholds"
        )
    else:
        stats["hint"] = (
            "conf has range but weak texture correlation — prefer soft weight "
            "or per-image percentile_drop; do not use a fixed absolute cutoff"
        )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face_conf_dir", type=Path, default=None)
    parser.add_argument("--face_image_dir", type=Path, default=None)
    parser.add_argument("--face_images", type=Path, nargs="*", default=None)
    parser.add_argument("--model_name", default="depth-anything/DA3-LARGE-1.1")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_faces", type=int, default=8)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    if args.face_conf_dir is not None:
        conf_paths = sorted(Path(args.face_conf_dir).glob("*_conf.npy"))[: args.max_faces]
        for cp in conf_paths:
            conf = np.load(cp).astype(np.float64)
            name = cp.stem
            img = None
            if args.face_image_dir is not None:
                # Best-effort: stem like frame_face0_conf -> look for pano_camera0/...
                # Just try same stem jpg/png under image dir recursively.
                candidates = list(Path(args.face_image_dir).rglob(f"{name.replace('_conf','')}*"))
                if not candidates:
                    # Try matching face index from name
                    candidates = []
                for c in Path(args.face_image_dir).rglob("*"):
                    if c.is_file() and c.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        candidates.append(c)
                        break
                if candidates:
                    img = cv2.imread(str(candidates[0]))
            reports.append(analyze_pair(img, conf, name, out_dir))
    elif args.face_images:
        from depth_anything_3.api import DepthAnything3
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = DepthAnything3.from_pretrained(args.model_name).to(device)
        paths = list(args.face_images)[: args.max_faces]
        pred = model.inference(image=[str(p) for p in paths])
        if pred.conf is None:
            print("Model returned no conf — nothing to inspect.")
            sys.exit(2)
        for i, p in enumerate(paths):
            conf = np.asarray(pred.conf[i], dtype=np.float64)
            img = cv2.imread(str(p))
            # Resize image to conf resolution for overlay
            reports.append(analyze_pair(img, conf, Path(p).stem, out_dir))
            np.save(out_dir / f"{Path(p).stem}_conf.npy", conf.astype(np.float32))
    else:
        parser.error("Provide --face_conf_dir or --face_images")

    write_json(out_dir / "conf_report.json", reports)
    print(json.dumps(reports, indent=2))
    print(
        "\nRecommendation: DA3 export code uses percentile cutoffs → treat conf as "
        "ordinal. Default pipeline uses conf_mode=weight and seam_mode=conf_weight."
    )


if __name__ == "__main__":
    main()
