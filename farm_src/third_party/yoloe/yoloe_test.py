#!/usr/bin/env python3
"""
YOLOE-v8-L Prompt-Free segmentation over frame*.jpg with Supervision visualization,
plus (optional) proto tiling and per-detection image dumps.

Example:
  python yoloe_pf_dir_vis.py \
    --images_dir /path/to/frames \
    --out_dir vis_pf \
    --model_id yoloe-v8l \
    --vocab_file tools/ram_tag_list.txt \
    --imgsz 640 --conf 0.25 --iou 0.70 --device cuda:0 \
    --save-protos --save-detections
"""

import argparse
from pathlib import Path
import math

import torch
import numpy as np
from PIL import Image

import supervision as sv
from ultralytics import YOLOE
from ultralytics.utils.torch_utils import smart_inference_mode
from huggingface_hub import hf_hub_download


def init_model(model_id: str, is_pf: bool = False, device: str | None = None) -> YOLOE:
    filename = f"{model_id}-seg.pt" if not is_pf else f"{model_id}-seg-pf.pt"
    path = hf_hub_download(repo_id="jameslahm/yoloe", filename=filename)
    model = YOLOE(path)
    model.eval()
    model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return model


def load_vocab_list(vocab_file: Path) -> list[str]:
    with open(vocab_file, "r") as f:
        names = [x.strip() for x in f.readlines() if x.strip()]
    return names


def render_scene_masks(pil_img: Image.Image, detections: sv.Detections) -> np.ndarray:
    """Scene-level composite: masks + (optional) boxes/labels."""
    resolution_wh = pil_img.size  # (W,H)
    mask_annot = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=0.45)
    annotated = mask_annot.annotate(scene=pil_img.copy(), detections=detections)
    return np.asarray(annotated)


def tile_protos_to_image(proto: np.ndarray, normalize=True) -> Image.Image:
    """
    proto: (C, H, W) ndarray. Tile each channel as a small grayscale tile.
    Returns a PIL image (grayscale) with a grid of C tiles.
    """
    assert proto.ndim == 3, f"Proto must be (C,H,W), got {proto.shape}"
    C, H, W = proto.shape

    # Normalize each channel to [0,255] for visualization
    if normalize:
        p = proto.copy()
        for i in range(C):
            ch = p[i]
            mn, mx = ch.min(), ch.max()
            if mx > mn:
                p[i] = (ch - mn) / (mx - mn)
            else:
                p[i] = np.zeros_like(ch)
        p = (p * 255.0).clip(0, 255).astype(np.uint8)
    else:
        # Just scale to 0..255 by global min/max
        mn, mx = proto.min(), proto.max()
        p = ((proto - mn) / (mx - mn + 1e-12) * 255.0).clip(0, 255).astype(np.uint8)

    # Grid size: square-ish
    cols = math.ceil(math.sqrt(C))
    rows = math.ceil(C / cols)

    canvas = np.zeros((rows * H, cols * W), dtype=np.uint8)
    for i in range(C):
        r = i // cols
        c = i % cols
        canvas[r * H : (r + 1) * H, c * W : (c + 1) * W] = p[i]

    return Image.fromarray(canvas, mode="L")


def save_detection_images(
    pil_img: Image.Image,
    detections: sv.Detections,
    class_names: list[str],
    out_dir: Path,
    stem: str,
    opacity: float = 0.6,
):
    """
    Save one image per detection with mask overlay + label. Filenames include index and class.
    Ensures mask is shaped (1, H, W) for Supervision.
    """
    det_dir = out_dir / "detections"
    det_dir.mkdir(parents=True, exist_ok=True)

    img_np = np.asarray(pil_img)
    H, W = img_np.shape[:2]
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(W, H)) * 0.8

    for i in range(len(detections)):
        this = detections[i]

        # ---- Normalize mask to (1, H, W) or None ----
        mm = None
        m = getattr(this, "mask", None)
        if m is not None:
            if isinstance(m, torch.Tensor):
                m = m.detach().cpu().numpy()
            # m could be (H,W), (1,H,W), or (N,H,W). We want (1,H,W).
            if m.ndim == 2:
                mm = m[None, ...]
            elif m.ndim == 3:
                if m.shape[0] == 1:
                    mm = m
                else:
                    # some slices still carry (N,H,W); pick the first channel
                    mm = m[:1]
            else:
                raise ValueError(f"Unexpected mask ndim={m.ndim}, shape={m.shape}")
            # ensure uint8/bool for visualization
            if mm.dtype != np.bool_ and mm.dtype != np.uint8:
                mm = (mm > 0.5).astype(np.uint8)

        # ---- Build single-detection Detections ----
        single = sv.Detections(
            xyxy=this.xyxy,                # expected shape (1,4)
            mask=mm,                       # shape (1,H,W) or None
            class_id=this.class_id,        # shape (1,) or None
            confidence=this.confidence,    # shape (1,)
        )

        # ---- Draw ----
        canvas = Image.fromarray(img_np.copy())
        if mm is not None:
            mask_annot = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX, opacity=opacity)
            canvas = mask_annot.annotate(scene=canvas, detections=single)

        box_annot = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX, thickness=2)
        label_annot = sv.LabelAnnotator(
            text_padding=4, text_thickness=1, text_scale=text_scale, color_lookup=sv.ColorLookup.INDEX
        )
        canvas = box_annot.annotate(scene=canvas, detections=single)

        # label
        if this.class_id is not None:
            cls_idx = int(this.class_id[0])
            cls_name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else f"class{cls_idx}"
        else:
            cls_name = "cls"
        conf = float(this.confidence[0]) if this.confidence is not None else 0.0
        canvas = label_annot.annotate(scene=canvas, detections=single, labels=[f"{cls_name} {conf:.2f}"])

        # ---- Save ----
        out_path = (out_dir / "detections") / f"{stem}_det{i:03d}_{cls_name}.jpg"
        canvas.save(out_path, quality=95)


@smart_inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, type=str, help="Directory with frame*.jpg")
    ap.add_argument("--out_dir", type=str, default="yoloe_pf_vis", help="Output directory")
    ap.add_argument("--model_id", type=str, default="yoloe-v8l",
                    choices=["yoloe-v8s", "yoloe-v8m", "yoloe-v8l", "yoloe-11s", "yoloe-11m", "yoloe-11l"])
    ap.add_argument("--vocab_file", type=str, default="../kept_final.txt",
                    help="Newline-separated class names for PF vocab")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", type=str, default=None, help='e.g. "cuda:0" or "cpu"')
    ap.add_argument("--save-protos", action="store_true", help="Also tile and save the mask prototypes if exposed")
    ap.add_argument("--save-detections", action="store_true", help="Save one image per detection")
    args = ap.parse_args()

    images_dir = Path(args.images_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(images_dir.glob("*.*"))
    if not img_paths:
        print(f"No images in {images_dir}")
        return

    # 1) Build vocab using the UNFUSED seg checkpoint
    unfused = init_model(args.model_id, is_pf=False, device=args.device)
    names = load_vocab_list(Path(args.vocab_file))
    vocab = unfused.get_vocab(names)

    # 2) Load PF weights, inject vocab, fuse head
    model = init_model(args.model_id, is_pf=True, device=args.device)
    model.set_vocab(vocab, names=names)
    model.model.model[-1].is_fused = True
    model.model.model[-1].conf = 0.001
    model.model.model[-1].max_det = 1000

    # 3) Predict per image and visualize
    for p in img_paths:
        pil_img = Image.open(p).convert("RGB")
        results = model.predict(
            source=pil_img,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False
        )

        r = results[0]
        det = sv.Detections.from_ultralytics(r)
        # Scene-level composite
        annotated_np = render_scene_masks(pil_img, det)
        out_scene = out_dir / f"{p.stem}_pf_seg.jpg"
        Image.fromarray(annotated_np).save(out_scene, quality=95)
        print(f"Saved: {out_scene}")

        # Per-detection dump (optional)
        if args.save_detections and len(det) > 0:
            save_detection_images(pil_img, det, class_names=names, out_dir=out_dir, stem=p.stem)

        # Proto tiling (optional; best-effort — only if exposed by the build)
        if args.save_protos:
            proto = None
            # Ultralytics sometimes stashes it under r.masks.data (final masks) and keeps proto internally.
            # Some builds expose r.masks and r.masks.data but not proto; others add r.protos or r.masks.protos.
            # We try a few common places:
            for key in ("protos",):
                if hasattr(r, key):
                    obj = getattr(r, key)
                    if isinstance(obj, torch.Tensor):
                        proto = obj.detach().cpu().numpy()
                    elif isinstance(obj, np.ndarray):
                        proto = obj
                    break
            if proto is None and hasattr(r, "masks") and r.masks is not None:
                # Some forks expose r.masks.data (N,H,W) and r.masks.protos (C,H,W)
                if hasattr(r.masks, "protos"):
                    obj = getattr(r.masks, "protos")
                    if isinstance(obj, torch.Tensor):
                        proto = obj.detach().cpu().numpy()
                    elif isinstance(obj, np.ndarray):
                        proto = obj

            if proto is not None and proto.ndim == 3:
                proto_img = tile_protos_to_image(proto)  # (C,H,W) -> grid image
                out_proto = out_dir / f"{p.stem}_protos_tiled.jpg"
                proto_img.save(out_proto, quality=95)
                print(f"Saved protos: {out_proto}")
            else:
                print("[info] Protos not exposed by this Ultralytics/YOLOE build; skipping proto tile.")


if __name__ == "__main__":
    main()
