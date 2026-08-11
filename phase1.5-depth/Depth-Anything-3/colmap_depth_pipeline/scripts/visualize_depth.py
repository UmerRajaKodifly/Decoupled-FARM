#!/usr/bin/env python3
"""Quick depth visualization for equirect or face-level .npy maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from io_utils import load_depth, save_depth_vis  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("depth_path", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args()
    depth = load_depth(args.depth_path)
    out = args.output or args.depth_path.with_suffix(".vis.png")
    save_depth_vis(out, depth)
    print(f"Wrote {out} shape={depth.shape}")


if __name__ == "__main__":
    main()
