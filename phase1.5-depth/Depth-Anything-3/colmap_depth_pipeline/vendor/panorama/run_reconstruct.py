"""Entrypoint executed inside the COLMAP docker image to run panorama SfM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# When mounted at /pipeline/vendor/panorama, make panorama importable.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402

from panorama import (  # noqa: E402
    Mapper,
    Matcher,
    PanoramaReconstructionOptions,
    PanoRenderType,
    get_virtual_rotations,
    reconstruct,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Panorama SfM (runs inside COLMAP docker).")
    parser.add_argument("--input_image_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--pano_render_type",
        type=str,
        default="perspective_non_overlapping",
        choices=[e.value for e in PanoRenderType],
    )
    parser.add_argument("--matcher", type=str, default="sequential")
    parser.add_argument("--mapper", type=str, default="incremental")
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--use_cpu", dest="use_gpu", action="store_false")
    parser.add_argument("--gpu_index", default="-1")
    parser.add_argument(
        "--ba_backend",
        default="caspar",
        choices=["caspar", "ceres"],
        help="Bundle adjustment backend (default: caspar; falls back to GPU Ceres)",
    )
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=-1)
    args = parser.parse_args()

    render_type = PanoRenderType(args.pano_render_type)
    options = PanoramaReconstructionOptions(
        matcher=Matcher(args.matcher),
        mapper=Mapper(args.mapper),
        render_type=render_type,
        random_seed=args.random_seed,
        num_threads=args.num_threads,
        gpu_index=args.gpu_index,
        use_gpu=args.use_gpu,
        ba_backend=args.ba_backend,
    )
    args.output_path.mkdir(parents=True, exist_ok=True)
    # Ensure top-level dirs exist before multi-threaded rendering (Docker mounts).
    (args.output_path / "images").mkdir(parents=True, exist_ok=True)
    (args.output_path / "masks").mkdir(parents=True, exist_ok=True)
    reconstruct(args.input_image_path, args.output_path, options)

    # Persist face_meta for downstream fusion (NON_OVERLAPPING defaults).
    from panorama import PANO_RENDER_OPTIONS

    render_opts = PANO_RENDER_OPTIONS.get(render_type)
    if render_opts is not None:
        rotations = get_virtual_rotations(render_opts.num_steps_yaw, render_opts.pitches_deg)
        meta = {
            "render_type": render_type.value,
            "num_faces": len(rotations),
            "num_steps_yaw": render_opts.num_steps_yaw,
            "pitches_deg": list(render_opts.pitches_deg),
            "hfov_deg": render_opts.hfov_deg,
            "vfov_deg": render_opts.vfov_deg,
            "face_prefixes": [f"pano_camera{i}/" for i in range(len(rotations))],
            "cam_from_pano": [np.asarray(R).tolist() for R in rotations],
        }
        (args.output_path / "face_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"Wrote face_meta.json with {meta['num_faces']} faces")


if __name__ == "__main__":
    main()
