"""Unit tests for COLMAP→metric pose scale (alpha) and legacy helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from scale_align import (  # noqa: E402
    apply_scale_shift,
    camera_center_from_w2c,
    fit_pose_scale_alpha,
    fit_scale_shift,
    scale_extrinsics,
    summarize_conf_map,
)


def test_fit_pose_scale_alpha_clean():
    rng = np.random.default_rng(0)
    sparse = rng.uniform(1.0, 10.0, size=200)
    alpha_gt = 1.25
    metric = alpha_gt * sparse
    alpha, info = fit_pose_scale_alpha(metric, sparse, n_iters=3, min_points=20, conf_mode="none")
    assert info["ok"]
    assert abs(alpha - alpha_gt) < 0.02


def test_fit_pose_scale_alpha_outliers():
    rng = np.random.default_rng(1)
    sparse = rng.uniform(1.0, 10.0, size=200)
    alpha_gt = 0.9
    metric = alpha_gt * sparse
    metric[:30] = rng.uniform(50, 100, size=30)
    alpha, info = fit_pose_scale_alpha(
        metric, sparse, n_iters=5, thresh_mult=2.0, min_points=20, conf_mode="none"
    )
    assert info["ok"]
    assert abs(alpha - alpha_gt) < 0.08


def test_scale_extrinsics():
    E = np.eye(4)
    E[:3, 3] = [1.0, 2.0, 3.0]
    out = scale_extrinsics(E, 2.0)
    assert np.allclose(out[:3, 3], [2.0, 4.0, 6.0])
    assert np.allclose(out[:3, :3], np.eye(3))


def test_camera_center_scales_with_alpha():
    E = np.eye(4)
    E[:3, 3] = [1.0, 0.0, 0.0]  # C = -t = (-1,0,0) for R=I
    c0 = camera_center_from_w2c(E)
    c1 = camera_center_from_w2c(scale_extrinsics(E, 2.5))
    assert np.allclose(c0, [-1.0, 0.0, 0.0])
    assert np.allclose(c1, [-2.5, 0.0, 0.0])


def test_fit_scale_shift_clean():
    rng = np.random.default_rng(0)
    pred = rng.uniform(1.0, 10.0, size=200)
    a_gt, b_gt = 1.05, -0.2
    sparse = a_gt * pred + b_gt
    a, b, info = fit_scale_shift(pred, sparse, n_iters=3, min_points=20, conf_mode="none")
    assert info["ok"]
    assert abs(a - a_gt) < 0.02
    assert abs(b - b_gt) < 0.05


def test_apply_scale_shift():
    d = np.array([[1.0, 2.0], [3.0, np.nan]])
    out = apply_scale_shift(d, 2.0, 1.0)
    assert out[0, 0] == 3.0
    assert np.isnan(out[1, 1])


def test_too_few_points():
    alpha, info = fit_pose_scale_alpha(np.array([1.0, 2.0]), np.array([1.0, 2.0]), min_points=10)
    assert not info["ok"]
    assert alpha == 1.0


def test_summarize_conf_map():
    c = np.linspace(0.1, 2.0, 100).reshape(10, 10)
    s = summarize_conf_map(c)
    assert s["ok"]
    assert s["recommendation"] == "weight"
