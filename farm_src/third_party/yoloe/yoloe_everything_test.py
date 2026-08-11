"""
Prompt-free segmentation demo with extra \"unknown\" masks using a second-pass NMS filter.

This script runs YOLOE in prompt-free mode, collects the standard vocabulary-based detections,
and then performs a secondary scan of the raw predictions to recover low-confidence objects
that likely belong to categories outside the vocabulary list. The unknown proposals are
filtered with IoU against the known detections and rendered with their own label.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import supervision as sv
import torch
from PIL import Image

from ultralytics import YOLOE
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.utils import DEFAULT_CFG, ops


# Holds CLI-configured thresholds for unknown-mask mining. Populated in main().
UNKNOWN_CFG = {}


@dataclass
class UnknownDetections:
    """Container for the extra 'unknown' detections."""

    boxes: torch.Tensor
    scores: torch.Tensor
    masks: Optional[torch.Tensor]

    @staticmethod
    def empty(device: torch.device | None = None) -> "UnknownDetections":
        dev = device or torch.device("cpu")
        return UnknownDetections(
            boxes=torch.zeros((0, 4), device=dev),
            scores=torch.zeros(0, device=dev),
            masks=None,
        )


class TwoPassSegPredictor(SegmentationPredictor):
    """
    Custom predictor that keeps the default segmentation outputs and, in addition,
    extracts a set of unknown detections using a relaxed confidence threshold.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.unknown_results: List[UnknownDetections] = []
        self.unknown_cfg = UNKNOWN_CFG.copy()

    def postprocess(self, preds, img, orig_imgs):
        results = super().postprocess(preds, img, orig_imgs)
        self.unknown_results = self._collect_unknown(preds, img, orig_imgs, results)
        return results

    def _collect_unknown(self, preds, img, orig_imgs, results: List) -> List[UnknownDetections]:
        cfg = self.unknown_cfg
        conf_low = cfg.get("conf_low")
        if conf_low is None or conf_low <= 0:
            return [UnknownDetections.empty() for _ in results]

        conf_high = cfg.get("conf_high", 0.3)
        overlap_thresh = cfg.get("iou_with_known", 0.5)
        max_det = cfg.get("max_det", 200)

        pred_logits = preds[0].clone()
        unknown_raw = ops.non_max_suppression(
            pred_logits,
            conf_thres=conf_low,
            iou_thres=self.args.iou,
            classes=None,
            agnostic=self.args.agnostic_nms,
            max_det=max_det,
            nc=len(self.model.names),
        )

        orig_list = (
            orig_imgs if isinstance(orig_imgs, list) else ops.convert_torch2numpy_batch(orig_imgs)
        )
        proto = preds[1][-1] if isinstance(preds[1], (list, tuple)) else preds[1]

        collected: List[UnknownDetections] = []
        for i, (raw_det, orig_img, known_res) in enumerate(zip(unknown_raw, orig_list, results)):
            if raw_det is None or not len(raw_det):
                collected.append(UnknownDetections.empty(device=pred_logits.device))
                continue

            masks = None
            det = raw_det.clone()

            if self.args.retina_masks:
                det[:, :4] = ops.scale_boxes(img.shape[2:], det[:, :4], orig_img.shape)
                masks = ops.process_mask_native(proto[i], det[:, 6:], det[:, :4], orig_img.shape[:2])
            else:
                masks = ops.process_mask(proto[i], det[:, 6:], det[:, :4], img.shape[2:], upsample=True)
                det[:, :4] = ops.scale_boxes(img.shape[2:], det[:, :4], orig_img.shape)
                scaled = ops.scale_masks(masks[:, None].float(), orig_img.shape[:2]).squeeze(1)
                masks = scaled.gt_(0.0)

            keep_mask = torch.ones(det.shape[0], dtype=torch.bool, device=det.device)

            if conf_high is not None:
                keep_mask &= det[:, 4] < conf_high

            known_boxes = getattr(known_res, "boxes", None)
            if (
                keep_mask.any()
                and known_boxes is not None
                and hasattr(known_boxes, "xyxy")
                and len(known_boxes)
                and overlap_thresh < 1.0
            ):
                kb = torch.as_tensor(known_boxes.xyxy, device=det.device, dtype=det.dtype)
                overlaps = ops.box_iou(det[:, :4], kb)
                keep_mask &= overlaps.max(dim=1).values < overlap_thresh

            if not keep_mask.any():
                collected.append(UnknownDetections.empty(device=pred_logits.device))
                continue

            det = det[keep_mask]
            masks = masks[keep_mask] if masks is not None else None

            collected.append(
                UnknownDetections(
                    boxes=det[:, :4].detach().cpu(),
                    scores=det[:, 4].detach().cpu(),
                    masks=None if masks is None else masks.detach().cpu(),
                )
            )

        return collected


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOE prompt-free everything segmentation test.")
    parser.add_argument("--source", type=str, required=True, help="Path to an input image.")
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the annotated image. Defaults to <source>_everything.png",
    )
    parser.add_argument("--pf-checkpoint", type=str, default="pretrain/yoloe-v8l-seg-pf.pt")
    parser.add_argument("--vocab-config", type=str, default="yoloe-v8l.yaml")
    parser.add_argument("--vocab-weights", type=str, default="pretrain/yoloe-v8l-seg.pt")
    parser.add_argument("--names-file", type=str, default="tools/ram_tag_list.txt")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence for known detections.")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold for known detections.")
    parser.add_argument("--head-conf", type=float, default=0.001, help="LRPC head confidence gate.")
    parser.add_argument("--max-det", type=int, default=1000, help="Max detections for known path.")
    parser.add_argument("--retina-masks", action="store_true", help="Use retina mask rendering.")
    parser.add_argument("--unknown-conf-low", type=float, default=0.25, help="Low pass conf for unknown search.")
    parser.add_argument(
        "--unknown-conf-high",
        type=float,
        default=0.35,
        help="Upper bound for class confidence to remain unknown (set None to disable).",
    )
    parser.add_argument(
        "--unknown-iou",
        type=float,
        default=0.5,
        help="IoU threshold to drop overlaps with known detections.",
    )
    parser.add_argument("--unknown-max-det", type=int, default=200, help="Max proposals to keep as unknown.")
    parser.add_argument("--unknown-label", type=str, default="unknown", help="Label for unknown masks.")
    return parser.parse_args()


def read_vocab(names_path: str) -> List[str]:
    with open(names_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_prompt_free_model(args, names: List[str], device: torch.device) -> YOLOE:
    unfused = YOLOE(args.vocab_config)
    if args.vocab_weights:
        unfused.load(args.vocab_weights)
    unfused.to(device)
    unfused.eval()
    vocab = unfused.get_vocab(names)

    model = YOLOE(args.pf_checkpoint)
    model.to(device)
    model.set_vocab(vocab, names=names)
    head = model.model.model[-1]
    head.is_fused = True
    head.conf = args.head_conf
    head.max_det = args.max_det
    model.eval()
    return model


def annotate(image: Image.Image, known: sv.Detections, unknown: UnknownDetections, args) -> Image.Image:
    annotated = np.array(image.copy())
    resolution_wh = image.size
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.4)
    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        color_lookup=sv.ColorLookup.INDEX,
        text_scale=text_scale * 0.5,
        text_thickness=max(1, thickness - 1),
        smart_position=True,
    )

    if len(known):
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(known["class_name"], known.confidence)
        ]
    else:
        labels = []

    annotated = mask_annotator.annotate(scene=annotated, detections=known)
    # annotated = box_annotator.annotate(scene=annotated, detections=known)
    # annotated = label_annotator.annotate(scene=annotated, detections=known, labels=labels)

    if unknown.boxes.numel():
        unknown_det = sv.Detections(
            xyxy=unknown.boxes.numpy(),
            mask=None if unknown.masks is None else unknown.masks.numpy().astype(bool),
            confidence=unknown.scores.numpy(),
            class_id=np.full(unknown.boxes.shape[0], fill_value=-1),
        )
        unknown_mask_annotator = sv.MaskAnnotator(color=sv.Color.from_hex("#FFFFFF"), opacity=0.25)
        unknown_box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#FFFFFF"), thickness=thickness)
        unknown_label_annotator = sv.LabelAnnotator(
            color=sv.Color.from_hex("#FFFFFF"),
            text_scale=text_scale * 0.5,
            text_thickness=max(1, thickness - 1),
            smart_position=True,
        )
        unknown_labels = [f"{args.unknown_label} {float(score):.2f}" for score in unknown.scores.numpy()]
        annotated = unknown_mask_annotator.annotate(scene=annotated, detections=unknown_det)
        annotated = unknown_box_annotator.annotate(scene=annotated, detections=unknown_det)
        annotated = unknown_label_annotator.annotate(scene=annotated, detections=unknown_det, labels=unknown_labels)

    return Image.fromarray(annotated)


def main():
    args = parse_args()
    if not os.path.exists(args.source):
        raise FileNotFoundError(f"Source image {args.source} not found.")

    global UNKNOWN_CFG
    UNKNOWN_CFG = {
        "conf_low": args.unknown_conf_low,
        "conf_high": args.unknown_conf_high,
        "iou_with_known": args.unknown_iou,
        "max_det": args.unknown_max_det,
        "label": args.unknown_label,
    }

    names = read_vocab(args.names_file)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_prompt_free_model(args, names, device)

    predictor_args = dict(
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        retina_masks=args.retina_masks,
        verbose=False,
        predictor=TwoPassSegPredictor,
    )

    results = model.predict(source=args.source, **predictor_args)
    unknown = getattr(model.predictor, "unknown_results", [UnknownDetections.empty() for _ in results])

    image = Image.open(args.source).convert("RGB")
    known_det = sv.Detections.from_ultralytics(results[0])
    annotated = annotate(image, known_det, unknown[0], args)

    output_path = args.output
    if not output_path:
        stem = Path(args.source).stem
        output_path = f"{stem}_everything.png"
    annotated.save(output_path)

    print(
        f"Known detections: {len(known_det)} | "
        f"Unknown proposals: {unknown[0].boxes.shape[0]} | "
        f"Saved visualization to {output_path}"
    )


if __name__ == "__main__":
    main()
