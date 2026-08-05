"""YOLOE 2D instance masks. Class / prompt set is caller-provided — never guessed."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    mask: np.ndarray  # (H, W) bool, original image resolution
    bbox_xyxy: np.ndarray  # (4,) float


class YOLOEDetector:
    def __init__(
        self,
        *,
        classes: list[str] | None,
        prompt_free: bool = False,
        model_id: str = "yoloe-v8l-seg.pt",
        conf: float = 0.25,
        iou: float = 0.5,
        device: str | None = None,
    ) -> None:
        if not prompt_free and not classes:
            raise ValueError(
                "YOLOE class/prompt list is required unless prompt_free=True. "
                "Pass --classes or set prompt_free explicitly. Do not guess a vocab."
            )
        from ultralytics import YOLOE

        self.classes = list(classes or [])
        self.prompt_free = bool(prompt_free)
        self.conf = conf
        self.iou = iou
        self.model_id = model_id
        logger.info("Loading YOLOE checkpoint %s (prompt_free=%s)", model_id, prompt_free)
        self.model = YOLOE(model_id)
        if device:
            self.model.to(device)
        if self.prompt_free:
            # Prompt-free weights use the built-in vocab of the *-seg-pf.pt ckpt.
            if not str(model_id).endswith("-pf.pt") and "yoloe" in str(model_id):
                logger.warning(
                    "prompt_free=True but checkpoint is %s; FARM uses yoloe-v8l-seg-pf.pt",
                    model_id,
                )
        else:
            self.model.set_classes(self.classes, self.model.get_text_pe(self.classes))

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            image_bgr,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )
        if not results:
            return []
        r0 = results[0]
        if r0.masks is None or r0.boxes is None:
            return []
        masks = r0.masks.data.cpu().numpy()  # (N, h, w) model space
        boxes = r0.boxes.xyxy.cpu().numpy()
        scores = r0.boxes.conf.cpu().numpy()
        cls_ids = r0.boxes.cls.cpu().numpy().astype(int)
        names = r0.names or {}
        h, w = image_bgr.shape[:2]
        out: list[Detection] = []
        for i in range(masks.shape[0]):
            mask_small = masks[i]
            mask = (
                cv2.resize(mask_small.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                > 0.5
            )
            label = str(names.get(int(cls_ids[i]), cls_ids[i]))
            out.append(
                Detection(
                    label=label,
                    score=float(scores[i]),
                    mask=mask.astype(bool),
                    bbox_xyxy=boxes[i].astype(np.float32),
                )
            )
        return out


def ensure_yoloe_checkpoint(model_id: str = "yoloe-v8l-seg.pt") -> Path:
    """Trigger ultralytics download of the checkpoint if missing."""
    from ultralytics import YOLOE

    model = YOLOE(model_id)
    ckpt = getattr(model, "ckpt_path", None) or model_id
    return Path(str(ckpt))
