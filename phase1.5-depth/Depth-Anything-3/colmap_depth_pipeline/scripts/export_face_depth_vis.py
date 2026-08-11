#!/usr/bin/env python3
"""Write colorized PNG previews for per-face depth maps.

Layout (new):
  face_depth/face0/{stem}.npy → face_depth_vis/face0/{stem}.png

Also accepts legacy flat files: face_depth/{stem}_face{id}.npy

  python colmap_depth_pipeline/scripts/export_face_depth_vis.py \
    --out_dir indoor_output/depth_recon
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from io_utils import (  # noqa: E402
    face_folder,
    face_vis_path,
    load_depth,
    save_depth_vis,
)

logger = logging.getLogger(__name__)
_LEGACY_RE = re.compile(r"^(?P<stem>.+)_face(?P<fid>\d+)(?P<raw>_raw)?\.npy$")


def _iter_face_depths(depth_dir: Path, include_raw: bool):
    # New layout: face_depth/face{id}/{stem}.npy
    subdirs = sorted(p for p in depth_dir.glob("face*") if p.is_dir())
    if subdirs:
        for sub in subdirs:
            m = re.fullmatch(r"face(\d+)", sub.name)
            if not m:
                continue
            fid = int(m.group(1))
            for p in sorted(sub.glob("*.npy")):
                if p.name.endswith("_raw.npy") and not include_raw:
                    continue
                stem = p.stem[:-4] if p.name.endswith("_raw.npy") else p.stem
                if p.name.endswith("_raw.npy"):
                    stem = p.name[: -len("_raw.npy")]
                yield fid, stem, p
        return

    # Legacy flat layout
    for p in sorted(depth_dir.glob("*.npy")):
        m = _LEGACY_RE.match(p.name)
        if not m:
            continue
        if m.group("raw") and not include_raw:
            continue
        yield int(m.group("fid")), m.group("stem"), p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Pipeline depth output dir (contains face_depth/)",
    )
    parser.add_argument(
        "--face_depth_dir",
        type=Path,
        default=None,
        help="Override face_depth directory",
    )
    parser.add_argument(
        "-o",
        "--vis_dir",
        type=Path,
        default=None,
        help="Output dir for PNGs (default: <out_dir>/face_depth_vis)",
    )
    parser.add_argument(
        "--include_raw",
        action="store_true",
        help="Also convert *_raw.npy (default: skip raw)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    depth_dir = args.face_depth_dir or (args.out_dir / "face_depth")
    vis_dir = args.vis_dir or (args.out_dir / "face_depth_vis")
    if not depth_dir.is_dir():
        raise SystemExit(f"Missing face depth dir: {depth_dir}")

    items = list(_iter_face_depths(depth_dir, include_raw=args.include_raw))
    if not items:
        raise SystemExit(f"No .npy depth maps in {depth_dir}")

    n = 0
    for fid, stem, src in items:
        face_folder(vis_dir, fid).mkdir(parents=True, exist_ok=True)
        depth = load_depth(src)
        out = face_vis_path(vis_dir, stem, fid)
        save_depth_vis(out, depth)
        n += 1
        if n % 50 == 0 or n == len(items):
            logger.info("[%d/%d] %s", n, len(items), out.relative_to(vis_dir))
    print(f"Wrote {n} PNGs → {vis_dir}/face{{id}}/")


if __name__ == "__main__":
    main()
