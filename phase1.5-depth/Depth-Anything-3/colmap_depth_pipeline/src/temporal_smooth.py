"""Optional temporal smoothing of per-frame (a, b) scale-shift parameters."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def smooth_scale_shift(
    abs_list: Sequence[Tuple[float, float]],
    *,
    window: int = 5,
) -> List[Tuple[float, float]]:
    """
    Moving-average smooth of (a, b) along the trajectory.

    window must be odd-ish; uses centered window clipped at edges.
    """
    if window <= 1 or len(abs_list) == 0:
        return [(float(a), float(b)) for a, b in abs_list]
    arr = np.asarray(abs_list, dtype=np.float64)  # [T, 2]
    half = max(1, window // 2)
    out = []
    T = arr.shape[0]
    for i in range(T):
        lo = max(0, i - half)
        hi = min(T, i + half + 1)
        out.append((float(arr[lo:hi, 0].mean()), float(arr[lo:hi, 1].mean())))
    return out
