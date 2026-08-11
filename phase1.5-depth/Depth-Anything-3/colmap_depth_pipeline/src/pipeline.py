"""COLMAP relative poses + DA3METRIC dense depth (meters).

SfM owns relative poses/structure. Dense depth comes from DA3METRIC.
Global scale alpha upgrades COLMAP translations into meters via
``d_metric ≈ alpha · d_sparse``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import yaml

from colmap_io import default_face_meta, load_colmap_model, write_face_meta
from cubemap import (
    faces_to_equirect_depth,
    get_virtual_rotations,
    planar_to_ray_distance,
)
from da3_runner import DA3Runner, iter_frame_windows
from io_utils import (
    face_conf_path,
    face_depth_path,
    face_folder,
    face_sky_path,
    face_vis_path,
    resolve_face_conf_path,
    resolve_face_depth_path,
    resolve_face_sky_path,
    save_conf,
    save_conf_vis,
    save_depth,
    save_depth_vis,
    write_json,
)
from scale_align import (
    align_depth_to_metric_sparse,
    fit_pose_scale_from_maps,
    fit_pose_scale_pooled,
    summarize_conf_map,
)
from sparse_depth import get_sparse_depth

logger = logging.getLogger(__name__)


def load_config(path: Path | None) -> dict:
    default_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    with open(default_path) as f:
        cfg = yaml.safe_load(f) or {}
    if path is None:
        return cfg
    cfg_path = Path(path)
    if cfg_path.resolve() == default_path.resolve():
        return cfg
    with open(cfg_path) as f:
        overlay = yaml.safe_load(f) or {}
    return _deep_merge(cfg, overlay)


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resize_map(arr: np.ndarray, hw: tuple[int, int], nearest: bool = True) -> np.ndarray:
    H, W = hw
    if arr.shape == (H, W):
        return arr.astype(np.float64)
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(arr.astype(np.float32), (W, H), interpolation=interp).astype(np.float64)


def _face_rotations_from_meta(meta: dict) -> Dict[int, np.ndarray]:
    if "cam_from_pano" in meta and meta["cam_from_pano"]:
        return {i: np.asarray(R, dtype=np.float64) for i, R in enumerate(meta["cam_from_pano"])}
    rots = get_virtual_rotations(
        int(meta.get("num_steps_yaw", 4)),
        tuple(meta.get("pitches_deg", [0.0])),
    )
    return {i: R for i, R in enumerate(rots)}


def run_pipeline(
    colmap_dir: Path,
    pano_dir: Path,
    out_dir: Path,
    *,
    config_path: Path | None = None,
    model_name: str | None = None,
    window_size: int | None = None,
    overlap: int | None = None,
    save_format: str | None = None,
    skip_da3: bool = False,
    device: str | None = None,
    export_ply: bool = False,
) -> dict:
    cfg = load_config(config_path)
    model_name = model_name or cfg.get("model_name", "depth-anything/DA3METRIC-LARGE")
    window_size = int(window_size if window_size is not None else cfg.get("window_size", 4))
    overlap = int(overlap if overlap is not None else cfg.get("overlap", 1))
    save_format = save_format or cfg.get("save_format", "npy")
    scale_cfg = cfg.get("scale_align", {})
    sparse_cfg = cfg.get("sparse_filter", {})
    fusion_cfg = cfg.get("fusion", {}) or {}
    fusion_enabled = bool(fusion_cfg.get("enabled", False))
    agree_cfg = cfg.get("depth_agree", {}) or {}
    agree_enabled = bool(agree_cfg.get("enabled", True))
    per_face_scale = bool(agree_cfg.get("per_face_scale", True)) and agree_enabled
    sampling_number = int(agree_cfg.get("sampling_number", 5_000_000))
    sky_cfg = cfg.get("sky_seg", {}) or {}
    sky_enabled = bool(sky_cfg.get("enabled", True))
    expected_faces = int(cfg.get("faces_per_frame", 4))
    pose_condition = bool(cfg.get("pose_condition", False))
    align_to_input_ext_scale = bool(cfg.get("align_to_input_ext_scale", False))
    use_ray_pose = bool(cfg.get("use_ray_pose", False))
    # global: one alpha for whole trajectory (default); per_frame: per-frame alphas
    alpha_mode = str(scale_cfg.get("mode", "global"))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    face_depth_dir = out_dir / "face_depth"
    face_conf_dir = out_dir / "face_conf"
    face_sky_dir = out_dir / "face_sky"
    face_vis_dir = out_dir / "face_depth_vis"
    save_face_vis = bool(cfg.get("save_face_depth_vis", True))
    for d in (face_depth_dir, face_conf_dir, face_sky_dir):
        d.mkdir(exist_ok=True)
    for fid in range(expected_faces):
        face_folder(face_depth_dir, fid).mkdir(parents=True, exist_ok=True)
        face_folder(face_conf_dir, fid).mkdir(parents=True, exist_ok=True)
        face_folder(face_sky_dir, fid).mkdir(parents=True, exist_ok=True)
        if save_face_vis:
            face_folder(face_vis_dir, fid).mkdir(parents=True, exist_ok=True)

    depth_dir = conf_dir = vis_dir = conf_vis_dir = None
    if fusion_enabled:
        depth_dir = out_dir / "equirect_depth"
        conf_dir = out_dir / "equirect_conf"
        vis_dir = out_dir / "equirect_vis"
        conf_vis_dir = out_dir / "equirect_conf_vis"
        for d in (depth_dir, conf_dir, vis_dir, conf_vis_dir):
            d.mkdir(exist_ok=True)

    colmap_dir = Path(colmap_dir)
    meta_path = colmap_dir / "face_meta.json"
    if not meta_path.is_file():
        meta = default_face_meta(
            num_faces=expected_faces,
            num_steps_yaw=int(cfg.get("num_steps_yaw", 4)),
            pitches_deg=tuple(cfg.get("pitches_deg", [0.0])),
            hfov_deg=float(cfg.get("hfov_deg", 90.0)),
            vfov_deg=float(cfg.get("vfov_deg", 90.0)),
        )
        write_face_meta(meta_path, meta)

    model = load_colmap_model(colmap_dir, expected_faces=expected_faces)
    rotations = _face_rotations_from_meta(model.face_meta)
    frame_ids = model.frame_ids
    logger.info("Loaded %d frames, %d faces", len(frame_ids), len(model.faces))

    sparse_by_face: Dict[tuple, tuple] = {}
    for face in model.faces:
        coords, depths = get_sparse_depth(
            model,
            face,
            min_track_length=int(sparse_cfg.get("min_track_length", 3)),
            max_reproj_error=float(sparse_cfg.get("max_reproj_error", 4.0)),
        )
        sparse_by_face[(face.frame_id, face.face_id)] = (coords, depths)

    pred_depth: Dict[tuple, np.ndarray] = {}  # metric meters
    pred_conf: Dict[tuple, np.ndarray] = {}
    pred_sky: Dict[tuple, np.ndarray] = {}

    skip_infer = skip_da3  # reuse saved depths from out_dir
    if not skip_infer:
        runner = DA3Runner(
            model_name=model_name,
            device=device,
            process_res=int(cfg.get("process_res", 504)),
            ref_view_strategy=str(cfg.get("ref_view_strategy", "saddle_balanced")),
            use_ray_pose=use_ray_pose,
        )
        windows = iter_frame_windows(frame_ids, window_size, overlap)
        logger.info(
            "Running DA3 over %d windows (window_size=%d, pose_condition=%s, model=%s)",
            len(windows),
            window_size,
            pose_condition and runner.pose_ok,
            runner.model_name,
        )
        logged_conf = False
        logged_sky = False
        n_blended = 0
        for wi, window in enumerate(windows):
            faces_batch = [face for fid in window for face in model.frames[fid]]
            if not faces_batch:
                continue
            logger.info(
                "Window %d/%d: %d faces (%d frames)",
                wi + 1,
                len(windows),
                len(faces_batch),
                len(window),
            )
            raws, confs, skys = runner.infer_faces(
                faces_batch,
                pose_condition=pose_condition,
                align_to_input_ext_scale=align_to_input_ext_scale,
            )
            for i, face in enumerate(faces_batch):
                key = (face.frame_id, face.face_id)
                raw_r = _resize_map(raws[i], (face.height, face.width), nearest=True)
                d_m = runner.to_metric_face_depth(raw_r, face.intrinsics).astype(np.float64)
                c = None
                if confs is not None:
                    c = _resize_map(confs[i], (face.height, face.width), nearest=False)

                if key in pred_depth:
                    # Overlap: keep higher-confidence pixel (not first-write-wins)
                    if c is not None and key in pred_conf:
                        take = c > pred_conf[key]
                        pred_depth[key] = np.where(take, d_m, pred_depth[key])
                        pred_conf[key] = np.where(take, c, pred_conf[key])
                        n_blended += 1
                        cp = face_conf_path(
                            face_conf_dir, Path(key[0]).stem, key[1]
                        )
                        cp.parent.mkdir(parents=True, exist_ok=True)
                        np.save(cp, pred_conf[key].astype(np.float32))
                        if skys is not None and key in pred_sky:
                            sky_new = (
                                _resize_map(
                                    skys[i].astype(np.float32),
                                    (face.height, face.width),
                                    nearest=True,
                                )
                                >= 0.5
                            )
                            pred_sky[key] = np.where(take, sky_new, pred_sky[key])
                            sp = face_sky_path(
                                face_sky_dir, Path(key[0]).stem, key[1]
                            )
                            sp.parent.mkdir(parents=True, exist_ok=True)
                            np.save(sp, pred_sky[key].astype(np.uint8))
                    continue

                pred_depth[key] = d_m
                raw_p = face_depth_path(
                    face_depth_dir, Path(key[0]).stem, key[1], raw=True
                )
                raw_p.parent.mkdir(parents=True, exist_ok=True)
                np.save(raw_p, raw_r.astype(np.float32))
                if c is not None:
                    pred_conf[key] = c
                    cp = face_conf_path(
                        face_conf_dir, Path(key[0]).stem, key[1]
                    )
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    np.save(cp, c.astype(np.float32))
                    if not logged_conf:
                        logger.info("DA3 conf stats (first face): %s", summarize_conf_map(c))
                        logged_conf = True
                if sky_enabled and skys is not None:
                    sky = (
                        _resize_map(
                            skys[i].astype(np.float32),
                            (face.height, face.width),
                            nearest=True,
                        )
                        >= 0.5
                    )
                    pred_sky[key] = sky
                    sp = face_sky_path(face_sky_dir, Path(key[0]).stem, key[1])
                    sp.parent.mkdir(parents=True, exist_ok=True)
                    np.save(sp, sky.astype(np.uint8))
                    if not logged_sky:
                        logger.info(
                            "DA3 sky mask frac (first face): %.3f",
                            float(sky.mean()),
                        )
                        logged_sky = True
                elif sky_enabled and skys is None and not logged_sky:
                    logger.warning("sky_seg.enabled but model returned no sky masks")
                    logged_sky = True

        if n_blended:
            logger.info("Confidence-blended %d overlapping face updates across windows", n_blended)
    else:
        # Reuse written metric depths; if only raw exists, convert with face K.
        from da3_runner import is_metric_model, raw_to_metric_depth

        use_focal = (
            is_metric_model(model_name) and "nested" not in model_name.lower()
        )
        for face in model.faces:
            stem = Path(face.frame_id).stem
            metric_p = resolve_face_depth_path(face_depth_dir, stem, face.face_id)
            raw_p = resolve_face_depth_path(
                face_depth_dir, stem, face.face_id, raw=True
            )
            if metric_p is not None:
                pred_depth[(face.frame_id, face.face_id)] = np.load(metric_p).astype(np.float64)
            elif raw_p is not None:
                raw = np.load(raw_p).astype(np.float64)
                if use_focal:
                    pred_depth[(face.frame_id, face.face_id)] = raw_to_metric_depth(
                        raw, face.intrinsics
                    ).astype(np.float64)
                else:
                    pred_depth[(face.frame_id, face.face_id)] = raw
            cp = resolve_face_conf_path(face_conf_dir, stem, face.face_id)
            if cp is not None:
                pred_conf[(face.frame_id, face.face_id)] = np.load(cp).astype(np.float64)
            sp = resolve_face_sky_path(face_sky_dir, stem, face.face_id)
            if sp is not None:
                pred_sky[(face.frame_id, face.face_id)] = np.load(sp).astype(bool)

    fit_kwargs = dict(
        n_iters=int(scale_cfg.get("n_iters", 3)),
        thresh_mult=float(scale_cfg.get("thresh_mult", 2.0)),
        min_points=int(scale_cfg.get("min_points", 20)),
        conf_mode=str(scale_cfg.get("conf_mode", "weight")),
        conf_percentile=float(scale_cfg.get("conf_percentile", 20.0)),
        conf_min=scale_cfg.get("conf_min", None),
    )

    # Collect metric depths + sparse for alpha fit
    all_metric: dict = {}
    all_sparse: dict = {}
    all_conf: dict = {}
    frame_alphas: Dict[str, tuple] = {}
    scale_details: List[dict] = []

    for fi, face in enumerate(model.faces):
        key = (face.frame_id, face.face_id)
        if key not in pred_depth:
            continue
        # Use unique int keys for pooled dict across trajectory
        all_metric[fi] = pred_depth[key]
        all_sparse[fi] = sparse_by_face.get(key, (np.zeros((0, 2)), np.zeros((0,))))
        if key in pred_conf:
            all_conf[fi] = pred_conf[key]

    if alpha_mode == "none":
        # Relative reconstruction: depths already match COLMAP via pose-cond
        # (align_to_input_ext_scale) or we keep COLMAP units as-is.
        alpha_g = 1.0
        info_g = {"ok": True, "mode": "none", "n_inliers": 0}
        logger.info(
            "scale_align.mode=none → alpha=1.0 (relative / COLMAP units, no metric fit)"
        )
        for frame_id in frame_ids:
            frame_alphas[frame_id] = (alpha_g, info_g)
            scale_details.append(
                {"frame_id": frame_id, "alpha": alpha_g, "fit": info_g, "mode": "none"}
            )
        alpha_global = alpha_g
    elif alpha_mode == "global":
        alpha_g, info_g = fit_pose_scale_pooled(
            all_metric,
            all_sparse,
            face_confs=all_conf if all_conf else None,
            **fit_kwargs,
        )
        if not info_g.get("ok"):
            logger.warning("Global alpha fit failed (%s); using alpha=1.0", info_g)
            alpha_g = 1.0
        logger.info(
            "Global COLMAP→depth alpha=%.4f (n_inliers=%s residual_med=%s)",
            alpha_g,
            info_g.get("n_inliers"),
            info_g.get("residual_med"),
        )
        for frame_id in frame_ids:
            frame_alphas[frame_id] = (alpha_g, info_g)
            scale_details.append(
                {"frame_id": frame_id, "alpha": alpha_g, "fit": info_g, "mode": "global"}
            )
        alpha_global = alpha_g
    else:
        alphas = []
        for frame_id in frame_ids:
            face_preds, face_sparse, face_confs = {}, {}, {}
            per_face = {}
            for face in model.frames[frame_id]:
                key = (frame_id, face.face_id)
                if key not in pred_depth:
                    continue
                face_preds[face.face_id] = pred_depth[key]
                face_sparse[face.face_id] = sparse_by_face.get(
                    key, (np.zeros((0, 2)), np.zeros((0,)))
                )
                if key in pred_conf:
                    face_confs[face.face_id] = pred_conf[key]
                a_f, info_f = fit_pose_scale_from_maps(
                    pred_depth[key],
                    face_sparse[face.face_id][0],
                    face_sparse[face.face_id][1],
                    conf_map=pred_conf.get(key),
                    **fit_kwargs,
                )
                per_face[face.face_id] = {"alpha": a_f, "info": info_f}
            a_f, info_f = fit_pose_scale_pooled(
                face_preds,
                face_sparse,
                face_confs=face_confs if face_confs else None,
                **fit_kwargs,
            )
            if not info_f.get("ok"):
                a_f = 1.0
            frame_alphas[frame_id] = (a_f, info_f)
            alphas.append(a_f)
            scale_details.append(
                {
                    "frame_id": frame_id,
                    "alpha": a_f,
                    "fit": info_f,
                    "faces": per_face,
                    "mode": "per_frame",
                }
            )
            logger.info("Frame %s alpha=%.4f", frame_id, a_f)
        alpha_global = float(np.median(alphas)) if alphas else 1.0

    # Per-face scale onto shared sparse scaffolding (pose_alpha * d_colmap).
    # With scale_align.mode=none and align_to_input_ext_scale, pose_alpha=1
    # and this only cleans residual per-face drift.
    face_scale_s: Dict[tuple, float] = {}
    if per_face_scale:
        logger.info(
            "Depth agreement: per-face scale onto alpha*sparse (alpha=%.4f)",
            alpha_global,
        )
        n_ok = 0
        for face in model.faces:
            key = (face.frame_id, face.face_id)
            if key not in pred_depth:
                continue
            coords, sparse_d = sparse_by_face.get(key, (np.zeros((0, 2)), np.zeros((0,))))
            aligned, s_f, info_f = align_depth_to_metric_sparse(
                pred_depth[key],
                coords,
                sparse_d,
                pose_alpha=frame_alphas[face.frame_id][0],
                conf_map=pred_conf.get(key),
                **fit_kwargs,
            )
            pred_depth[key] = aligned
            face_scale_s[key] = s_f
            if info_f.get("ok"):
                n_ok += 1
        logger.info(
            "Per-face depth alignment: %d/%d faces ok (median s=%.4f)",
            n_ok,
            len(face_scale_s),
            float(np.median(list(face_scale_s.values()))) if face_scale_s else 1.0,
        )

    pano_dir = Path(pano_dir)
    seam_mode = str(fusion_cfg.get("seam_mode", "conf_weight"))
    results = []
    for frame_id in frame_ids:
        faces = model.frames[frame_id]
        alpha, fit_info = frame_alphas[frame_id]
        stem = Path(frame_id).stem
        face_entries = []
        face_depths_ray = {}
        face_Ks = {}
        face_confs_fuse = {}

        for face in faces:
            key = (frame_id, face.face_id)
            if key not in pred_depth:
                continue
            d_m = pred_depth[key]
            # Optionally zero sky for export cleanliness
            if key in pred_sky:
                d_m = d_m.copy()
                d_m[pred_sky[key]] = np.nan
            out_path = save_depth(
                face_depth_path(face_depth_dir, stem, face.face_id).with_suffix(""),
                d_m,
                fmt=save_format,
            )
            vis_path = None
            if save_face_vis:
                vis_path = str(
                    save_depth_vis(face_vis_path(face_vis_dir, stem, face.face_id), d_m)
                )
            conf_path = None
            if key in pred_conf:
                conf_path = str(face_conf_path(face_conf_dir, stem, face.face_id))
                face_confs_fuse[face.face_id] = pred_conf[key]
            sky_path = None
            if key in pred_sky:
                sky_path = str(face_sky_path(face_sky_dir, stem, face.face_id))
            face_entries.append(
                {
                    "face_id": face.face_id,
                    "depth_path": str(out_path),
                    "vis_path": vis_path,
                    "conf_path": conf_path,
                    "sky_path": sky_path,
                    "raw_path": str(
                        face_depth_path(face_depth_dir, stem, face.face_id, raw=True)
                    ),
                    "image_path": str(face.image_path),
                    "face_depth_scale_s": face_scale_s.get(key),
                }
            )
            if fusion_enabled:
                face_depths_ray[face.face_id] = planar_to_ray_distance(d_m, face.intrinsics)
                face_Ks[face.face_id] = face.intrinsics

        frame_result: dict = {
            "frame_id": frame_id,
            "alpha": alpha,
            "fit": fit_info,
            "faces": face_entries,
        }

        if fusion_enabled and face_depths_ray:
            assert depth_dir and conf_dir and vis_dir and conf_vis_dir
            pano_path = pano_dir / frame_id
            if pano_path.is_file():
                pano = cv2.imread(str(pano_path), cv2.IMREAD_COLOR)
                H_eq, W_eq = pano.shape[:2]
            else:
                W_eq = int(faces[0].width * 360 / 90)
                H_eq = int(faces[0].height * 180 / 90)
            eq_depth, valid, eq_conf = faces_to_equirect_depth(
                face_depths_ray,
                rotations,
                (H_eq, W_eq),
                face_Ks=face_Ks,
                face_confs=face_confs_fuse if face_confs_fuse else None,
                hfov_deg=float(model.face_meta.get("hfov_deg", 90.0)),
                seam_mode=seam_mode,
            )
            out_path = save_depth(depth_dir / stem, eq_depth, fmt=save_format)
            conf_path = save_conf(conf_dir / stem, eq_conf, fmt=save_format)
            save_depth_vis(vis_dir / f"{stem}.png", eq_depth)
            save_conf_vis(conf_vis_dir / f"{stem}.png", eq_conf)
            frame_result.update(
                {
                    "equirect_depth_path": str(out_path),
                    "equirect_conf_path": str(conf_path),
                    "invalid_fraction": float((~valid).mean()),
                }
            )

        logger.info(
            "Wrote %d metric face depths for %s (alpha=%.4f)",
            len(face_entries),
            frame_id,
            alpha,
        )
        results.append(frame_result)

    manifest = {
        "colmap_dir": str(colmap_dir),
        "pano_dir": str(pano_dir),
        "out_dir": str(out_dir),
        "model_name": model_name,
        "window_size": window_size,
        "overlap": overlap,
        "pose_condition": pose_condition,
        "align_to_input_ext_scale": align_to_input_ext_scale,
        "use_ray_pose": use_ray_pose,
        "alpha_mode": alpha_mode,
        "alpha": alpha_global,
        "conf_mode": fit_kwargs["conf_mode"],
        "fusion_enabled": fusion_enabled,
        "sky_seg": {"enabled": sky_enabled},
        "depth_agree": {
            "enabled": agree_enabled,
            "per_face_scale": per_face_scale,
            "sampling_number": sampling_number,
            "n_faces_scaled": len(face_scale_s),
            "median_face_s": (
                float(np.median(list(face_scale_s.values()))) if face_scale_s else None
            ),
        },
        "frames": results,
        "scale_details": scale_details,
        "roles": {
            "sfm": "relative poses + sparse structure (COLMAP units)",
            "dense_depth": (
                "pose-conditioned relative depth aligned to COLMAP"
                if alpha_mode == "none"
                else "DA3METRIC dense depth (meters); alpha scales COLMAP t"
            ),
            "pointcloud": "unproject face depths with COLMAP K + poses",
            "primary_outputs": "face_depth/ (DA3), pointcloud.ply",
        },
        "units": "colmap" if alpha_mode == "none" else "meters",
    }
    write_json(out_dir / "manifest.json", manifest)

    if export_ply:
        from pointcloud_export import export_face_pointcloud

        ply_path = export_face_pointcloud(
            colmap_dir=colmap_dir,
            out_dir=out_dir,
            stride=4,
            max_points=sampling_number,
            mask_sky=True,
            show_cameras=True,
            camera_sphere_radius=0.08,
        )
        manifest["pointcloud_path"] = str(ply_path)
        write_json(out_dir / "manifest.json", manifest)

    return manifest
