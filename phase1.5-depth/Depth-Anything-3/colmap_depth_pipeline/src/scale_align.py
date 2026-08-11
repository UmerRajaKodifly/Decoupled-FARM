"""Fit COLMAP→metric pose scale from DA3METRIC depths vs sparse COLMAP depth.

alpha maps COLMAP units into meters: d_metric ≈ alpha * d_sparse
Pose translations (and camera centers) are scaled by alpha; depths stay metric.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

DEFAULT_CONF_MODE = "weight"


def _sample_map_at(image: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Nearest-neighbor sample image[H,W] at pixel coords [M,2] (u,v)."""
    H, W = image.shape
    u = np.clip(np.round(coords[:, 0]).astype(np.int64), 0, W - 1)
    v = np.clip(np.round(coords[:, 1]).astype(np.int64), 0, H - 1)
    return image[v, u].astype(np.float64)


_sample_pred_at = _sample_map_at


def fit_pose_scale_alpha(
    metric_depth_at_sparse: np.ndarray,
    sparse_depth: np.ndarray,
    *,
    conf_at_sparse: np.ndarray | None = None,
    conf_mode: str = DEFAULT_CONF_MODE,
    conf_percentile: float = 20.0,
    conf_min: float | None = None,
    n_iters: int = 3,
    thresh_mult: float = 2.0,
    min_points: int = 8,
) -> tuple[float, dict]:
    """
    Fit d_metric ≈ alpha * d_sparse (scale-only, through origin).

    Returns (alpha, info). alpha converts COLMAP length → meters.
    """
    metric = np.asarray(metric_depth_at_sparse, dtype=np.float64).reshape(-1)
    sparse = np.asarray(sparse_depth, dtype=np.float64).reshape(-1)
    mask = np.isfinite(metric) & np.isfinite(sparse) & (metric > 1e-6) & (sparse > 1e-6)

    if conf_at_sparse is not None:
        conf = np.asarray(conf_at_sparse, dtype=np.float64).reshape(-1)
        if conf.shape != metric.shape:
            raise ValueError(f"conf shape {conf.shape} != metric shape {metric.shape}")
        mask &= np.isfinite(conf)
    else:
        conf = None
        conf_mode = "none"

    metric, sparse = metric[mask], sparse[mask]
    if conf is not None:
        conf = conf[mask]

    info: dict = {
        "n_inliers": 0,
        "n_total": int(metric.size),
        "n_before_conf_filter": int(metric.size),
        "residual_med": None,
        "ok": False,
        "conf_mode": conf_mode,
    }
    if metric.size < min_points:
        return 1.0, info

    weights = np.ones(metric.shape[0], dtype=np.float64)
    if conf is not None and conf_mode == "percentile_drop":
        thr = float(np.percentile(conf, conf_percentile))
        keep = conf >= thr
        info["conf_percentile_thr"] = thr
        info["n_dropped_conf"] = int((~keep).sum())
        metric, sparse, conf, weights = metric[keep], sparse[keep], conf[keep], weights[keep]
    elif conf is not None and conf_mode == "threshold":
        if conf_min is None:
            raise ValueError("conf_mode='threshold' requires conf_min")
        keep = conf >= float(conf_min)
        info["conf_min"] = float(conf_min)
        info["n_dropped_conf"] = int((~keep).sum())
        metric, sparse, conf, weights = metric[keep], sparse[keep], conf[keep], weights[keep]
    elif conf is not None and conf_mode == "weight":
        cmin, cmax = float(np.min(conf)), float(np.max(conf))
        if cmax > cmin:
            weights = 0.05 + 0.95 * (conf - cmin) / (cmax - cmin)
        info["conf_range"] = [cmin, cmax]
    elif conf_mode not in ("none", "weight", "percentile_drop", "threshold"):
        raise ValueError(f"Unknown conf_mode: {conf_mode!r}")

    info["n_total"] = int(metric.size)
    if metric.size < min_points:
        return 1.0, info

    inlier = np.ones(metric.shape[0], dtype=bool)
    alpha = 1.0
    for _ in range(max(1, n_iters)):
        m, s, w = metric[inlier], sparse[inlier], weights[inlier]
        if m.size < min_points:
            break
        # Weighted least squares through origin: alpha = sum(w m s) / sum(w s^2)
        denom = float(np.sum(w * s * s))
        if denom < 1e-18:
            break
        alpha = float(np.sum(w * m * s) / denom)
        if alpha <= 1e-12:
            alpha = 1.0
            break
        resid = np.abs(metric - alpha * sparse)
        med = float(np.median(resid[inlier])) if inlier.any() else float(np.median(resid))
        thresh = max(med * thresh_mult, 1e-6)
        inlier = resid <= thresh

    info["n_inliers"] = int(inlier.sum())
    if inlier.any():
        info["residual_med"] = float(np.median(np.abs(metric[inlier] - alpha * sparse[inlier])))
        # Also report median ratio for sanity
        info["median_ratio"] = float(np.median(metric[inlier] / sparse[inlier]))
    info["ok"] = info["n_inliers"] >= min_points
    info["alpha"] = alpha
    return alpha, info


def fit_pose_scale_from_maps(
    metric_depth: np.ndarray,
    coords: np.ndarray,
    sparse_depth: np.ndarray,
    conf_map: np.ndarray | None = None,
    **kwargs,
) -> tuple[float, dict]:
    metric_at = _sample_map_at(metric_depth, coords)
    conf_at = _sample_map_at(conf_map, coords) if conf_map is not None else None
    return fit_pose_scale_alpha(metric_at, sparse_depth, conf_at_sparse=conf_at, **kwargs)


def fit_pose_scale_pooled(
    face_metric: dict[int, np.ndarray],
    face_sparse: dict[int, Tuple[np.ndarray, np.ndarray]],
    face_confs: dict[int, np.ndarray] | None = None,
    **kwargs,
) -> tuple[float, dict]:
    """Pool sparse correspondences across faces (or whole trajectory)."""
    mets, spars, confs = [], [], []
    for fid, pred in face_metric.items():
        if fid not in face_sparse:
            continue
        coords, depths = face_sparse[fid]
        if len(depths) == 0:
            continue
        mets.append(_sample_map_at(pred, coords))
        spars.append(depths)
        if face_confs is not None and fid in face_confs:
            confs.append(_sample_map_at(face_confs[fid], coords))
    if not mets:
        return 1.0, {"ok": False, "n_total": 0, "n_inliers": 0}
    conf_at = np.concatenate(confs) if confs and len(confs) == len(mets) else None
    return fit_pose_scale_alpha(
        np.concatenate(mets),
        np.concatenate(spars),
        conf_at_sparse=conf_at,
        **kwargs,
    )


def align_depth_to_metric_sparse(
    metric_depth: np.ndarray,
    coords: np.ndarray,
    sparse_depth_colmap: np.ndarray,
    *,
    pose_alpha: float,
    conf_map: np.ndarray | None = None,
    **fit_kwargs,
) -> tuple[np.ndarray, float, dict]:
    """
    Scale a monocular metric depth map onto the shared COLMAP metric frame.

    Target: ``s * d_metric ≈ pose_alpha * d_sparse`` at sparse pixels, so every
    face agrees with the same sparse scaffolding (and thus with each other).

    Returns ``(aligned_depth, s, info)``. If the fit fails, returns the input
    depth unchanged with ``s=1``.
    """
    coords = np.asarray(coords, dtype=np.float64)
    sparse = np.asarray(sparse_depth_colmap, dtype=np.float64).reshape(-1)
    if coords.size == 0 or sparse.size == 0 or abs(pose_alpha) < 1e-12:
        return np.asarray(metric_depth, dtype=np.float64), 1.0, {"ok": False, "reason": "no_sparse"}

    metric_at = _sample_map_at(metric_depth, coords)
    conf_at = _sample_map_at(conf_map, coords) if conf_map is not None else None
    target = float(pose_alpha) * sparse
    # fit_pose_scale_alpha(A, B) → A ≈ s * B  ⇒  target ≈ s * metric_at
    s, info = fit_pose_scale_alpha(
        target,
        metric_at,
        conf_at_sparse=conf_at,
        **fit_kwargs,
    )
    if not info.get("ok") or not np.isfinite(s) or s <= 1e-12:
        return np.asarray(metric_depth, dtype=np.float64), 1.0, info
    aligned = np.asarray(metric_depth, dtype=np.float64) * float(s)
    info["s"] = float(s)
    info["pose_alpha"] = float(pose_alpha)
    return aligned, float(s), info


def scale_extrinsics(w2c: np.ndarray, alpha: float) -> np.ndarray:
    """Scale world-to-camera translation by alpha (R unchanged). C' = alpha * C."""
    out = np.asarray(w2c, dtype=np.float64).copy()
    if out.shape == (3, 4):
        out[:3, 3] *= alpha
    elif out.shape == (4, 4):
        out[:3, 3] *= alpha
    else:
        raise ValueError(f"Unexpected extrinsics shape {out.shape}")
    return out


def camera_center_from_w2c(w2c: np.ndarray) -> np.ndarray:
    """World-space camera center from OpenCV/COLMAP w2c: C = -R^T t."""
    E = np.asarray(w2c, dtype=np.float64)
    R, t = E[:3, :3], E[:3, 3]
    return (-R.T @ t).astype(np.float64)


def summarize_conf_map(conf: np.ndarray) -> dict:
    """Quick stats for empirical conf sanity checks."""
    c = np.asarray(conf, dtype=np.float64)
    valid = np.isfinite(c)
    if not np.any(valid):
        return {"ok": False}
    v = c[valid]
    return {
        "ok": True,
        "min": float(v.min()),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "p10": float(np.percentile(v, 10)),
        "p50": float(np.percentile(v, 50)),
        "p90": float(np.percentile(v, 90)),
        "dynamic_range_ratio": float((v.max() - v.min()) / (v.mean() + 1e-12)),
        "recommendation": "weight" if (v.max() - v.min()) > 1e-6 else "none",
    }


# --- Legacy helpers kept for unit tests / optional depth affine experiments ---


def _weighted_lstsq(pred: np.ndarray, sparse: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = np.clip(w, 0.0, None)
    if not np.any(w > 0):
        w = np.ones_like(pred)
    sw = np.sqrt(w)
    A = np.column_stack([pred * sw, sw])
    s = sparse * sw
    sol, _, _, _ = np.linalg.lstsq(A, s, rcond=None)
    return float(sol[0]), float(sol[1])


def fit_scale_shift(
    pred_depth_at_sparse: np.ndarray,
    sparse_depth: np.ndarray,
    *,
    conf_at_sparse: np.ndarray | None = None,
    conf_mode: str = DEFAULT_CONF_MODE,
    conf_percentile: float = 20.0,
    conf_min: float | None = None,
    n_iters: int = 3,
    thresh_mult: float = 2.0,
    min_points: int = 8,
) -> Tuple[float, float, dict]:
    """Legacy: sparse ≈ a * pred + b (depth→COLMAP). Prefer fit_pose_scale_alpha."""
    pred = np.asarray(pred_depth_at_sparse, dtype=np.float64).reshape(-1)
    sparse = np.asarray(sparse_depth, dtype=np.float64).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(sparse) & (pred > 1e-6) & (sparse > 1e-6)
    if conf_at_sparse is not None:
        conf = np.asarray(conf_at_sparse, dtype=np.float64).reshape(-1)
        mask &= np.isfinite(conf)
    else:
        conf = None
        conf_mode = "none"
    pred, sparse = pred[mask], sparse[mask]
    if conf is not None:
        conf = conf[mask]
    info = {"n_inliers": 0, "n_total": int(pred.size), "ok": False, "conf_mode": conf_mode}
    if pred.size < min_points:
        return 1.0, 0.0, info
    weights = np.ones(pred.shape[0], dtype=np.float64)
    if conf is not None and conf_mode == "percentile_drop":
        thr = float(np.percentile(conf, conf_percentile))
        keep = conf >= thr
        pred, sparse, conf, weights = pred[keep], sparse[keep], conf[keep], weights[keep]
        info["n_dropped_conf"] = int((~keep).sum())
    elif conf is not None and conf_mode == "threshold":
        keep = conf >= float(conf_min)
        pred, sparse, conf, weights = pred[keep], sparse[keep], conf[keep], weights[keep]
    elif conf is not None and conf_mode == "weight":
        cmin, cmax = float(np.min(conf)), float(np.max(conf))
        if cmax > cmin:
            weights = 0.05 + 0.95 * (conf - cmin) / (cmax - cmin)
    info["n_total"] = int(pred.size)
    if pred.size < min_points:
        return 1.0, 0.0, info
    inlier = np.ones(pred.shape[0], dtype=bool)
    a, b = 1.0, 0.0
    for _ in range(max(1, n_iters)):
        p, s, w = pred[inlier], sparse[inlier], weights[inlier]
        if p.size < min_points:
            break
        a, b = _weighted_lstsq(p, s, w)
        resid = np.abs(a * pred + b - sparse)
        med = float(np.median(resid[inlier])) if inlier.any() else float(np.median(resid))
        inlier = resid <= max(med * thresh_mult, 1e-6)
    info["n_inliers"] = int(inlier.sum())
    info["ok"] = info["n_inliers"] >= min_points
    return a, b, info


def apply_scale_shift(depth: np.ndarray, a: float, b: float) -> np.ndarray:
    out = a * depth.astype(np.float64) + b
    out[~np.isfinite(depth)] = np.nan
    return out
