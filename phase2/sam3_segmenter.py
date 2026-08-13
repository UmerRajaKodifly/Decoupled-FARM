"""SAM3 open-vocabulary segmenter with the same Phase 2 output schema as YOLOE.

Runs Meta SAM 3 on each 360 cuboid face (not the equirectangular image).
Every vocabulary class is prompted on every face; GPU / KV state is dropped
after each face so VRAM stays bounded.

Downstream geometry is unchanged: mask erosion, depth unprojection, 3-D
Gaussians, DINOv3 mask-pooled features — same keys Phase 3 already consumes.
"""

from __future__ import annotations

import gc
import os
from contextlib import contextmanager
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

from dino_extractor import Phase2DinoExtractor
from geometry import (
    build_mask_weights,
    compute_mask_medians,
    compute_weighted_stats,
    depth_mode_mad_filter,
    erode_masks,
    mahalanobis_reject,
    unproject_depth,
)


def _load_vocab(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


@contextmanager
def _sam3_autocast():
    """SAM3 ViT fused ops cast activations to bfloat16; autocast keeps weights in sync."""
    if torch.cuda.is_available():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def _color_to_pil(color: torch.Tensor) -> Image.Image:
    t = color.detach().cpu()
    if t.ndim == 3 and t.shape[0] == 3 and t.shape[-1] != 3:
        t = t.permute(1, 2, 0)
    arr = t.numpy()
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _boxes_to_xyxy_px(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """SAM3 may emit normalised [0,1] xyxy; convert to pixel xyxy."""
    out = boxes.detach().float().clone()
    if out.ndim == 1:
        out = out.unsqueeze(0)
    if out.numel() == 0:
        return torch.empty((0, 4), dtype=torch.float32, device=boxes.device)
    if out.shape[-1] > 4:
        out = out[..., :4]
    mx = float(out.max().item()) if out.numel() else 0.0
    if mx <= 1.5:
        scale = torch.tensor([width, height, width, height], device=out.device, dtype=out.dtype)
        out = out * scale
    return out


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    try:
        from torchvision.ops import nms
    except ImportError:
        order = torch.argsort(scores, descending=True)
        keep: list[int] = []
        suppressed = torch.zeros(boxes.shape[0], dtype=torch.bool, device=boxes.device)
        for i in order.tolist():
            if suppressed[i]:
                continue
            keep.append(i)
            xx1 = torch.maximum(boxes[i, 0], boxes[:, 0])
            yy1 = torch.maximum(boxes[i, 1], boxes[:, 1])
            xx2 = torch.minimum(boxes[i, 2], boxes[:, 2])
            yy2 = torch.minimum(boxes[i, 3], boxes[:, 3])
            inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
            area_i = (boxes[i, 2] - boxes[i, 0]).clamp(min=0) * (boxes[i, 3] - boxes[i, 1]).clamp(min=0)
            area_j = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
            iou_t = inter / (area_i + area_j - inter).clamp(min=1e-6)
            suppressed |= iou_t > iou
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)
    return nms(boxes, scores, iou)


class SAM3Segmenter:
    """Drop-in replacement for ConstructionSegmenter using SAM3 text prompts."""

    def __init__(
        self,
        vocab_file: Path | str,
        device: str = "cuda",
        conf: float = 0.35,
        iou: float = 0.5,
        checkpoint: Path | str | None = None,
        mask_erosion_px: int = 3,
        mahalanobis_thresh: float = 2.0,
        min_mask_pixels: int = 50,
        min_depth_points: int = 50,
    ) -> None:
        vocab_file = Path(vocab_file)
        if not vocab_file.is_file():
            raise FileNotFoundError(f"Vocab file not found: {vocab_file}")
        self.names = _load_vocab(vocab_file)
        if not self.names:
            raise ValueError(f"Empty vocabulary: {vocab_file}")

        self.device = torch.device(device)
        self.conf = float(conf)
        self.iou = float(iou)
        self.mask_erosion_px = int(mask_erosion_px)
        self.mahalanobis_thresh = float(mahalanobis_thresh)
        self.min_mask_pixels = int(min_mask_pixels)
        self.min_depth_points = int(min_depth_points)

        ckpt = Path(
            checkpoint
            or os.environ.get("SAM3_CHECKPOINT", "")
            or "/opt/sam3/sam3.pt"
        )
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"SAM3 checkpoint not found: {ckpt}. "
                "Place sam3.pt at models/sam3/sam3.pt (gated facebook/sam3; needs HF_TOKEN)."
            )

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        print(f"[sam3] loading {ckpt} on {self.device} (vocab={len(self.names)} classes)")
        self._model = build_sam3_image_model(
            checkpoint_path=str(ckpt),
            load_from_HF=False,
            device=str(self.device),
            eval_mode=True,
        )
        self._processor = Sam3Processor(
            self._model,
            device=str(self.device),
            confidence_threshold=self.conf,
        )
        self._dino = Phase2DinoExtractor(device=self.device)
        self.feature_dim = int(self._dino.hidden_size)

    def _release_state(self, state: dict | None) -> None:
        if state is None:
            return
        try:
            self._processor.reset_all_prompts(state)
        except Exception:
            pass
        state.clear()
        _clear_cuda()

    def _detect_face(self, color: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prompt every vocab class on one cuboid face. Returns boxes, scores, class_ids, masks."""
        image = _color_to_pil(color)
        width, height = image.size
        state = None
        boxes_acc: list[torch.Tensor] = []
        scores_acc: list[torch.Tensor] = []
        class_acc: list[int] = []
        masks_acc: list[torch.Tensor] = []

        try:
            with torch.inference_mode(), _sam3_autocast():
                state = self._processor.set_image(image)
                for class_id, prompt in enumerate(self.names):
                    try:
                        self._processor.reset_all_prompts(state)
                    except Exception:
                        pass
                    out = self._processor.set_text_prompt(prompt=prompt, state=state)
                    if isinstance(out, dict):
                        state = out
                    masks = out.get("masks") if isinstance(out, dict) else None
                    boxes = out.get("boxes") if isinstance(out, dict) else None
                    scores = out.get("scores") if isinstance(out, dict) else None
                    if masks is None or boxes is None or scores is None:
                        continue
                    if not isinstance(masks, torch.Tensor):
                        masks = torch.as_tensor(masks)
                    if not isinstance(boxes, torch.Tensor):
                        boxes = torch.as_tensor(boxes)
                    if not isinstance(scores, torch.Tensor):
                        scores = torch.as_tensor(scores)
                    if masks.numel() == 0:
                        continue
                    if masks.ndim == 4:
                        masks = masks.squeeze(1)
                    if masks.ndim == 2:
                        masks = masks.unsqueeze(0)
                    boxes_px = _boxes_to_xyxy_px(boxes, width, height)
                    scores_t = scores.detach().float().reshape(-1)
                    n = min(int(masks.shape[0]), int(boxes_px.shape[0]), int(scores_t.numel()))
                    for i in range(n):
                        sc = float(scores_t[i].item())
                        if sc < self.conf:
                            continue
                        mask_i = masks[i].detach()
                        if mask_i.dtype != torch.bool:
                            mask_i = mask_i > 0.5
                        if mask_i.shape[-2:] != (height, width):
                            mask_i = torch.nn.functional.interpolate(
                                mask_i.float().unsqueeze(0).unsqueeze(0),
                                size=(height, width),
                                mode="nearest",
                            ).squeeze(0).squeeze(0) > 0.5
                        if int(mask_i.sum().item()) < self.min_mask_pixels:
                            continue
                        boxes_acc.append(boxes_px[i].detach().cpu().float())
                        scores_acc.append(scores_t[i].detach().cpu().float())
                        class_acc.append(class_id)
                        masks_acc.append(mask_i.cpu().bool())
        finally:
            self._release_state(state)

        if not boxes_acc:
            empty_box = torch.empty((0, 4), dtype=torch.float32)
            empty_1d = torch.empty((0,), dtype=torch.float32)
            empty_ids = torch.empty((0,), dtype=torch.long)
            empty_masks = torch.empty((0, height, width), dtype=torch.bool)
            return empty_box, empty_1d, empty_ids, empty_masks

        boxes_t = torch.stack(boxes_acc, dim=0).detach().cpu().float()
        scores_t = torch.stack(scores_acc, dim=0).detach().cpu().float()
        class_t = torch.tensor(class_acc, dtype=torch.long)
        masks_t = torch.stack(masks_acc, dim=0).cpu().bool()
        keep = _nms(boxes_t, scores_t, self.iou).cpu()
        return boxes_t[keep], scores_t[keep], class_t[keep], masks_t[keep]

    def __call__(
        self,
        colors: List[torch.Tensor] | torch.Tensor,
        depths: List[torch.Tensor] | torch.Tensor,
        intrinsics: List[torch.Tensor] | torch.Tensor,
    ) -> dict:
        if isinstance(colors, torch.Tensor):
            colors = [colors]
        if isinstance(depths, torch.Tensor):
            depths = [depths]
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = [intrinsics]

        device = self.device
        boxes_l: list[torch.Tensor] = []
        scores_l: list[torch.Tensor] = []
        class_l: list[torch.Tensor] = []
        masks_l: list[torch.Tensor] = []
        batch_l: list[torch.Tensor] = []

        for b_idx, color in enumerate(colors):
            boxes_b, scores_b, class_b, masks_b = self._detect_face(color)
            n_b = int(scores_b.numel())
            if n_b == 0:
                continue
            boxes_l.append(boxes_b)
            scores_l.append(scores_b)
            class_l.append(class_b)
            masks_l.append(masks_b)
            batch_l.append(torch.full((n_b,), b_idx, dtype=torch.int64))

        if not boxes_l:
            return self._empty_pack(device)

        # Geometry tensors are (M, H, W); with 200+ SAM3 dets that OOMs an 8 GB
        # GPU that already holds SAM3. Run unproject / Mahalanobis on CPU.
        _clear_cuda()
        geo = torch.device("cpu")

        boxes = torch.cat(boxes_l, dim=0).to(geo)
        scores = torch.cat(scores_l, dim=0).to(geo)
        class_ids = torch.cat(class_l, dim=0).to(geo)
        masks = torch.cat(masks_l, dim=0).to(geo)
        batch_ids = torch.cat(batch_l, dim=0).to(geo)

        depth_maps = []
        k_maps = []
        for d, k in zip(depths, intrinsics):
            dep = d.squeeze()
            if dep.ndim != 2:
                raise ValueError(f"Expected depth (H,W), got {tuple(dep.shape)}")
            depth_maps.append(dep.to(device=geo, dtype=torch.float32))
            kk = k[:3, :3].to(device=geo, dtype=torch.float32)
            k_maps.append(kk)
        depth_b = torch.stack(depth_maps, dim=0)
        k_b = torch.stack(k_maps, dim=0)

        XB, YB, ZB = unproject_depth(depth_b, k_b)
        depth_valid = (ZB > 0) & torch.isfinite(ZB)
        masks_eroded = erode_masks(masks, self.mask_erosion_px)
        weights = build_mask_weights(masks_eroded, depth_valid, batch_ids, dtype=torch.float32)
        batch_long = batch_ids.long()
        XB_sel = XB[batch_long]
        YB_sel = YB[batch_long]
        ZB_sel = ZB[batch_long]
        weights = depth_mode_mad_filter(ZB_sel, weights, min_depth_points=self.min_depth_points)
        n, means, cov6 = compute_weighted_stats(XB_sel, YB_sel, ZB_sel, weights)
        if self.mahalanobis_thresh > 0.0:
            weights = mahalanobis_reject(
                XB_sel, YB_sel, ZB_sel, weights, means, cov6, thresh=self.mahalanobis_thresh
            )
            n, means, cov6 = compute_weighted_stats(XB_sel, YB_sel, ZB_sel, weights)
        means = compute_mask_medians(XB_sel, YB_sel, ZB_sel, weights, min_points=self.min_depth_points)

        M = int(batch_ids.numel())
        inlier_mask = weights > 0
        det_points_flat = torch.empty((0, 3), dtype=means.dtype, device=geo)
        det_points_offsets = torch.zeros((M + 1,), dtype=torch.long, device=geo)
        if inlier_mask.any():
            det_idx_t, py_t, px_t = torch.nonzero(inlier_mask, as_tuple=True)
            pts = torch.stack(
                [XB_sel[det_idx_t, py_t, px_t], YB_sel[det_idx_t, py_t, px_t], ZB_sel[det_idx_t, py_t, px_t]],
                dim=1,
            )
            det_points_flat = pts
            counts = torch.bincount(det_idx_t, minlength=M)
            det_points_offsets[1:] = counts.cumsum(dim=0)

        feat_chunks: list[torch.Tensor] = []
        dino_chunk = 32
        for i0 in range(0, M, dino_chunk):
            sl = slice(i0, i0 + dino_chunk)
            feat_chunks.append(self._dino.extract(list(colors), masks_eroded[sl], batch_ids[sl]))
            _clear_cuda()
        features = torch.cat(feat_chunks, dim=0) if feat_chunks else self._dino.extract(
            list(colors), masks_eroded, batch_ids
        )
        _clear_cuda()

        labels = [self.names[int(c)] if 0 <= int(c) < len(self.names) else f"class_{int(c)}"
                  for c in class_ids.tolist()]
        masks_list = [m.bool() for m in masks.unbind(0)]
        return {
            "batch_ids": batch_ids,
            "boxes_xyxy": boxes,
            "num_pixels": n,
            "scores": scores,
            "class_ids": class_ids,
            "labels": labels,
            "masks": masks_list,
            "features": features,
            "means": means,
            "cov6": cov6,
            "det_points_flat": det_points_flat,
            "det_points_offsets": det_points_offsets,
        }

    def _empty_pack(self, device: torch.device) -> dict:
        return {
            "batch_ids": torch.empty((0,), dtype=torch.int64, device=device),
            "boxes_xyxy": torch.empty((0, 4), dtype=torch.float32, device=device),
            "num_pixels": torch.empty((0,), dtype=torch.float32, device=device),
            "scores": torch.empty((0,), dtype=torch.float32, device=device),
            "class_ids": torch.empty((0,), dtype=torch.long, device=device),
            "labels": [],
            "masks": [],
            "features": torch.empty((0, self.feature_dim), dtype=torch.float32, device=device),
            "means": torch.empty((0, 3), dtype=torch.float32, device=device),
            "cov6": torch.empty((0, 6), dtype=torch.float32, device=device),
            "det_points_flat": torch.empty((0, 3), dtype=torch.float32, device=device),
            "det_points_offsets": torch.zeros((1,), dtype=torch.long, device=device),
        }
