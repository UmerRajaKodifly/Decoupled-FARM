"""Cross-frame association baselines.

Default production path is FARM Hellinger + DINO + union-find, implemented by
wrapping FARM source in ``farm_runtime.py`` (``association.method: farm``).

This module keeps the cheap greedy same-class IoU tracker as
``association.method: greedy_iou`` for diffs against the FARM backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detect import Detection


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    union = np.logical_or(a, b).sum(dtype=np.int64)
    if union == 0:
        return 0.0
    return float(inter) / float(union)


@dataclass
class Track:
    track_id: int
    label: str
    last_mask: np.ndarray
    last_bbox: np.ndarray
    last_frame_index: int
    hits: int = 1


@dataclass
class IoUTracker:
    iou_threshold: float = 0.3
    max_age: int = 5
    use_mask_iou: bool = False
    _next_id: int = 1
    tracks: list[Track] = field(default_factory=list)

    def update(self, frame_index: int, detections: list[Detection]) -> list[tuple[int, Detection]]:
        assigned: list[tuple[int, Detection]] = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        scores: list[tuple[float, int, int]] = []
        for ti in unmatched_tracks:
            tr = self.tracks[ti]
            for di in unmatched_dets:
                det = detections[di]
                if det.label != tr.label:
                    continue
                if self.use_mask_iou:
                    iou = mask_iou(tr.last_mask, det.mask)
                else:
                    iou = _bbox_iou(tr.last_bbox, det.bbox_xyxy)
                if iou >= self.iou_threshold:
                    scores.append((iou, ti, di))
        scores.sort(reverse=True)

        used_t: set[int] = set()
        used_d: set[int] = set()
        for _, ti, di in scores:
            if ti in used_t or di in used_d:
                continue
            tr = self.tracks[ti]
            det = detections[di]
            tr.last_mask = det.mask
            tr.last_bbox = det.bbox_xyxy
            tr.last_frame_index = frame_index
            tr.hits += 1
            assigned.append((tr.track_id, det))
            used_t.add(ti)
            used_d.add(di)

        for di, det in enumerate(detections):
            if di in used_d:
                continue
            tid = self._next_id
            self._next_id += 1
            self.tracks.append(
                Track(
                    track_id=tid,
                    label=det.label,
                    last_mask=det.mask,
                    last_bbox=det.bbox_xyxy,
                    last_frame_index=frame_index,
                )
            )
            assigned.append((tid, det))

        alive = []
        for tr in self.tracks:
            if frame_index - tr.last_frame_index <= self.max_age:
                alive.append(tr)
        self.tracks = alive
        return assigned
