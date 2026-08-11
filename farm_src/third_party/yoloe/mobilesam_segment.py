#!/usr/bin/env python3
"""
MobileSAMv2 segmentation + visualization pipeline.

Example:
  python mobilesam_segment.py \
    --images_dir /path/to/images \
    --weights_path /path/to/mobilesamv2/weights \
    --out_dir mobilesam_vis \
    --device cuda:0 --imgsz 640 --conf 0.45 --iou 0.90
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterator, Sequence

import numpy as np
import torch
from PIL import Image
import supervision as sv

# Ensure the bbq.* modules are importable when invoked from arbitrary cwd.
ROS_SRC = Path(__file__).resolve().parent.parent  # .../ros2_ws/src
BBQ_PY = ROS_SRC / "bbq"
for candidate in (ROS_SRC, BBQ_PY):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.append(candidate_str)

try:
    from bbq.models.masks.masks_generator import ClassAgnosticMaskGenerator
except ModuleNotFoundError as exc:
    missing = exc.name or ""
    hint = ""
    if "mobilesamv2" in missing:
        hint = (
            "\nIt looks like the MobileSAMv2 python package is not installed or not on PYTHONPATH.\n"
            "Clone https://github.com/ChaoningZhang/MobileSAM (or your internal fork) and either\n"
            "  * run `pip install -e /path/to/mobilesamv2`, or\n"
            "  * add that directory to PYTHONPATH before launching this script.\n"
            "The weights directory passed via --weights_path must contain the *.pt files referenced there."
        )
    elif "bbq" in missing:
        hint = (
            "\nMake sure you've installed the BBQ package (pip install -e ros/ros2_ws/src/bbq)\n"
            "or added ros/ros2_ws/src to PYTHONPATH before running this script."
        )
    raise ModuleNotFoundError(f"{exc}{hint}") from exc


def _chunk(iterable: Sequence[Path], size: int) -> Iterator[list[Path]]:
    idx = 0
    total = len(iterable)
    while idx < total:
        yield list(iterable[idx : idx + size])
        idx += size


def _to_uint8_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.uint8)
    return torch.from_numpy(arr)


def _to_supervision_dets(det_dict: dict) -> sv.Detections:
    xyxy = det_dict["xyxy"].astype(np.float32, copy=False)
    masks = det_dict["mask"].astype(bool, copy=False)
    conf = det_dict["confidence"].astype(np.float32, copy=False)
    class_ids = np.zeros_like(conf, dtype=np.int32)
    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=class_ids, mask=masks)


def _render_overlay(
    pil_img: Image.Image,
    detections: sv.Detections,
    mask_annotator: sv.MaskAnnotator,
    box_annotator: sv.BoxAnnotator | None,
    label_annotator: sv.LabelAnnotator | None,
) -> np.ndarray:
    if len(detections) == 0:
        return np.asarray(pil_img)

    annotated = pil_img.copy()
    annotated = mask_annotator.annotate(scene=annotated, detections=detections)

    if box_annotator is not None:
        annotated = box_annotator.annotate(scene=annotated, detections=detections)

    if label_annotator is not None:
        labels = [f"{conf:.2f}" for conf in detections.confidence]
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    return np.asarray(annotated)


def _render_mask_composite(
    resolution_wh: tuple[int, int],
    detections: sv.Detections,
    mask_annotator: sv.MaskAnnotator,
) -> np.ndarray:
    width, height = resolution_wh
    blank = Image.new("RGB", (width, height), (0, 0, 0))
    base = np.asarray(blank)
    if len(detections) == 0:
        return base
    return mask_annotator.annotate(scene=base, detections=detections)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment images with MobileSAMv2 and save visualizations.")
    parser.add_argument(
        "--images_dir",
        "--images",
        required=True,
        dest="images_dir",
        help="Directory containing input images.",
    )
    parser.add_argument("--weights_path", required=True, help="Directory with MobileSAMv2 weights.")
    parser.add_argument("--out_dir", default="mobilesam_vis", help="Directory to store visualizations.")

    parser.add_argument("--device", default=None, help='Torch device (e.g. "cuda:0" or "cpu").')
    parser.add_argument("--imgsz", type=int, default=640, help="Object detector input size.")
    parser.add_argument("--conf", type=float, default=0.45, help="Detector confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.9, help="Detector NMS IoU threshold.")
    parser.add_argument("--sam-batch-size", type=int, default=192, help="Box micro-batch for MobileSAMv2 decoder.")
    parser.add_argument("--low-vram", action="store_true", help="Reduce memory usage at the cost of speed.")

    parser.add_argument("--images-per-call", type=int, default=4, help="How many images to process per generator call.")
    parser.add_argument("--mask-alpha", type=float, default=0.6, help="Opacity for overlays.")
    parser.add_argument("--draw-boxes", action="store_true", help="Also render bounding boxes.")
    parser.add_argument("--draw-labels", action="store_true", help="Render confidence labels.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    images_dir = Path(args.images_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(p for p in images_dir.glob("*") if p.is_file())
    if not img_paths:
        raise FileNotFoundError(f"No files found under {images_dir}")

    weights_dir = Path(args.weights_path).expanduser()
    if weights_dir.is_file():
        raise ValueError(
            f"--weights_path must point to the MobileSAMv2 weights directory, got file: {weights_dir}"
        )

    generator = ClassAgnosticMaskGenerator(
        model="MobileSAM",
        weights_path=str(weights_dir),
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        batch_size=args.sam_batch_size,
        low_vram=args.low_vram,
    )

    mask_overlay_annot = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=args.mask_alpha)
    mask_composite_annot = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=1.0)

    total_saved = 0
    for batch in _chunk(img_paths, max(1, args.images_per_call)):
        pil_images = [Image.open(path).convert("RGB") for path in batch]
        tensors = [_to_uint8_tensor(pil) for pil in pil_images]

        det_out = generator(tensors)
        if isinstance(det_out, dict):
            det_dicts = [det_out]
        else:
            det_dicts = list(det_out)
        if len(det_dicts) != len(batch):
            raise RuntimeError(
                f"Generator returned {len(det_dicts)} results for {len(batch)} inputs."
            )

        for path, pil_img, det_dict in zip(batch, pil_images, det_dicts):
            detections = _to_supervision_dets(det_dict)

            w, h = pil_img.size
            thickness = sv.calculate_optimal_line_thickness(resolution_wh=(w, h))
            box_annotator = (
                sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=thickness)
                if args.draw_boxes
                else None
            )
            label_annotator = (
                sv.LabelAnnotator(
                    color_lookup=sv.ColorLookup.INDEX,
                    text_scale=sv.calculate_optimal_text_scale(resolution_wh=(w, h)) * 0.4,
                    text_thickness=max(1, thickness // 2),
                )
                if args.draw_labels
                else None
            )

            overlay_np = _render_overlay(
                pil_img,
                detections,
                mask_overlay_annot,
                box_annotator,
                label_annotator,
            )
            overlay_path = out_dir / f"{path.stem}_mobilesam_overlay.jpg"
            Image.fromarray(overlay_np).save(overlay_path, quality=95)

            mask_np = _render_mask_composite(
                pil_img.size,
                detections,
                mask_composite_annot,
            )
            mask_path = out_dir / f"{path.stem}_mobilesam_masks.png"
            Image.fromarray(mask_np).save(mask_path)

            total_saved += 2
            print(f"Saved overlays for {path.name} -> {overlay_path.name}, {mask_path.name}")

        for pil in pil_images:
            pil.close()

    print(f"Done. Wrote {total_saved} visualization files to {out_dir}")


if __name__ == "__main__":
    main()
