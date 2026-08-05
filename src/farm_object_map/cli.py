"""CLI for the FARM-style monocular object-mapping pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from .cloud import ply_to_cloud_npz
from .colmap_dense import ColmapMvsDepthSource, export_mvs_depth_npz, undistort_and_mvs
from .compare_flows import compare_object_summaries
from .cubemap import (
    DEFAULT_RENDER_TYPE,
    render_cubemap_faces,
    rig_config_for_render_type,
    run_panorama_sfm,
)
from .depth import probe_dl_depth_v1
from .detect import YOLOEDetector, ensure_yoloe_checkpoint
from .dl_depth_v1 import build_dl_depth_v1_source, npz_dir_depth_source
from .frames import extract_frames, probe_video
from .gpu_verify import probe_colmap_build
from .mapper import map_detections_to_objects, reproject_mean_into_mask, save_object
from .poses import FramePose, export_frame_poses, save_poses_json, verify_pose_convention
from .sfm import export_sparse_cloud, run_sfm
from .vocab import DEFAULT_SPATIALGPT_VOCAB, write_farm_vocab_txt

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_SS3DGS_PARAMS = REPO / "configs" / "ss3dgs_sfm_only.yaml"
DEFAULT_EXTRACT_FPS = 2.0  # configs/ss3dgs_sfm_only.yaml general.extract_fps


def _load_image(image_dir: Path):
    def loader(frame_name: str):
        path = image_dir / frame_name
        if not path.exists():
            matches = list(image_dir.rglob(Path(frame_name).name))
            if not matches:
                raise FileNotFoundError(frame_name)
            path = matches[0]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read {path}")
        return image

    return loader


def _load_poses(path: str | Path) -> list[FramePose]:
    poses_payload = json.loads(Path(path).read_text())
    return [
        FramePose(
            frame_name=e["frame_name"],
            K=np.asarray(e["K"], dtype=np.float64),
            T_world_cam=np.asarray(e["T_world_cam"], dtype=np.float32),
            T_cam_world=np.asarray(e["T_cam_world"], dtype=np.float32),
            width=int(e["width"]),
            height=int(e["height"]),
            camera_model=e["camera_model"],
            image_id=int(e["image_id"]),
        )
        for e in poses_payload
    ]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    probe = sub.add_parser("probe-video", help="Print video metadata; does not extract.")
    probe.add_argument("--video", required=True)

    extract = sub.add_parser("extract-frames")
    extract.add_argument("--video", required=True)
    extract.add_argument("--out", required=True)
    extract.add_argument("--fps", type=float, default=DEFAULT_EXTRACT_FPS)
    extract.add_argument("--max-frames", type=int, default=None)
    extract.add_argument("--duration-s", type=float, default=None, help="Only decode the first N seconds.")

    sfm = sub.add_parser("sfm", help="COLMAP SfM via ss-3dgs helpers (no 3DGS).")
    sfm.add_argument("--images", required=True)
    sfm.add_argument("--workspace", required=True)
    sfm.add_argument("--params", default=str(DEFAULT_SS3DGS_PARAMS))

    cubemap = sub.add_parser("render-cubemap", help="Equirect → ss-3dgs perspective cubemap faces.")
    cubemap.add_argument("--images", required=True, help="Directory of equirectangular frames.")
    cubemap.add_argument("--out", required=True, help="Output directory for pano_camera*/ faces.")
    cubemap.add_argument("--masks", required=True, help="COLMAP mask tree output.")
    cubemap.add_argument("--render-type", default=DEFAULT_RENDER_TYPE)
    cubemap.add_argument("--edge-margin-px", type=int, default=0)

    pano_sfm = sub.add_parser("sfm-panorama", help="COLMAP rig SfM on cubemap faces.")
    pano_sfm.add_argument("--faces", required=True)
    pano_sfm.add_argument("--masks", required=True)
    pano_sfm.add_argument("--workspace", required=True)
    pano_sfm.add_argument("--render-type", default=DEFAULT_RENDER_TYPE)
    pano_sfm.add_argument("--params", default=str(DEFAULT_SS3DGS_PARAMS))

    poses = sub.add_parser("export-poses", help="Export K + T_world_cam and verify convention.")
    poses.add_argument("--model", required=True)
    poses.add_argument("--out", required=True)

    dense = sub.add_parser(
        "dense-depth",
        help="Explicit COLMAP MVS alternate depth (not the default mapping source).",
    )
    dense.add_argument("--images", required=True)
    dense.add_argument("--model", required=True)
    dense.add_argument("--dense-workspace", required=True)
    dense.add_argument("--gpu-index", default="0")
    dense.add_argument("--npz-out", default=None, help="Write DepthMap .npz files here.")

    cloud = sub.add_parser("export-cloud", help="Sparse COLMAP PLY → cloud.npz (viz only).")
    cloud.add_argument("--model", required=True)
    cloud.add_argument("--out-dir", required=True)

    ckpt = sub.add_parser("download-yoloe")
    ckpt.add_argument("--model-id", default="yoloe-v8l-seg.pt")

    smoke = sub.add_parser("smoke-yoloe", help="Load FARM YOLOE + vocab; infer masks on one frame.")
    smoke.add_argument("--image", required=True)
    smoke.add_argument("--out", required=True)
    smoke.add_argument("--vocab-json", default=str(DEFAULT_SPATIALGPT_VOCAB))
    smoke.add_argument("--no-dino", action="store_true")

    mapping = sub.add_parser("map-objects", help="Phase 3: masks + depth → Gaussians + voxels.")
    mapping.add_argument("--images", required=True)
    mapping.add_argument("--poses", required=True, help="poses.json from export-poses")
    mapping.add_argument("--out", required=True)
    mapping.add_argument("--association", choices=("farm", "greedy_iou"), default="farm")
    mapping.add_argument(
        "--depth-source",
        choices=("dl", "colmap_mvs"),
        default="dl",
        help="dl is the primary/default source. colmap_mvs is an explicit alternate flow.",
    )
    mapping.add_argument("--vocab-json", default=str(DEFAULT_SPATIALGPT_VOCAB))
    mapping.add_argument("--dense-workspace", default=None, help="Required when --depth-source colmap_mvs.")
    mapping.add_argument("--classes", nargs="*", default=None)
    mapping.add_argument("--prompt-free", action="store_true")
    mapping.add_argument("--yoloe-model", default="yoloe-v8l-seg.pt")
    mapping.add_argument("--voxel-size", type=float, default=0.05, help="Default 5 cm (indoor).")
    mapping.add_argument("--iou-threshold", type=float, default=0.3)
    mapping.add_argument("--min-depth-points", type=int, default=30)
    mapping.add_argument("--conf", type=float, default=0.25)
    mapping.add_argument("--no-dino", action="store_true")
    mapping.add_argument(
        "--allow-mvs-fallback",
        action="store_true",
        help="Explicit opt-in to COLMAP MVS. Default is to refuse if DL depth is missing.",
    )
    mapping.add_argument(
        "--dl-depth-npz-dir",
        default=None,
        help="Directory of precomputed dl_depth_v1 DepthMap .npz files (one per frame).",
    )

    e2e = sub.add_parser("e2e", help="Phase 1→4 on one video (stops if DL depth is missing).")
    e2e.add_argument("--video", required=True)
    e2e.add_argument("--work", default=None, help="Run directory. Optional when --outputs-root is set.")
    e2e.add_argument("--fps", type=float, default=DEFAULT_EXTRACT_FPS)
    e2e.add_argument("--max-frames", type=int, default=None)
    e2e.add_argument("--duration-s", type=float, default=None)
    e2e.add_argument("--vocab-json", default=str(DEFAULT_SPATIALGPT_VOCAB))
    e2e.add_argument("--params", default=str(DEFAULT_SS3DGS_PARAMS))
    e2e.add_argument("--association", choices=("farm", "greedy_iou"), default="farm")
    e2e.add_argument("--no-dino", action="store_true")
    e2e.add_argument(
        "--depth-source",
        choices=("dl", "colmap_mvs"),
        default="dl",
        help="Primary default is dl. Use colmap_mvs only for the explicit comparison flow.",
    )
    e2e.add_argument("--gpu-index", default="0", help="GPU index for Caspar BA logs + patch_match_stereo.")
    e2e.add_argument(
        "--outputs-root",
        default=None,
        help="If set, work dir becomes <outputs-root>/<video_stem>/<depth-source>/.",
    )
    e2e.add_argument("--reuse-sfm", action="store_true", help="Reuse work/sfm/sparse/0 when present.")
    e2e.add_argument("--allow-mvs-fallback", action="store_true")
    e2e.add_argument(
        "--image-type",
        choices=("panorama", "standard"),
        default="panorama",
        help="panorama = ss-3dgs cubemap then FARM pinhole on faces. standard = raw frames as pinhole.",
    )
    e2e.add_argument("--render-type", default=DEFAULT_RENDER_TYPE)
    e2e.add_argument("--dl-depth-npz-dir", default=None)

    dl = sub.add_parser("check-dl-depth", help="Report whether dl_depth_v1 is deployable.")

    cmp_ = sub.add_parser("compare-flows", help="Diff dl vs colmap_mvs object summaries.")
    cmp_.add_argument("--dl-summary", required=True)
    cmp_.add_argument("--mvs-summary", required=True)
    cmp_.add_argument("--out", required=True)
    cmp_.add_argument("--max-mean-dist", type=float, default=2.0)
    return p


def _subset_frames(src_dir: Path, dest_dir: Path, max_frames: int | None) -> Path:
    frames = sorted(src_dir.glob("frame_*.jpg")) + sorted(src_dir.glob("frame_*.png"))
    if max_frames is not None:
        frames = frames[: max_frames]
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in frames:
        target = dest_dir / f.name
        if not target.exists():
            shutil.copy2(f, target)
    return dest_dir


def _run_smoke_yoloe(args) -> int:
    from .farm_runtime import AssociationConfig, build_farm_segmenter, smoke_yoloe_masks

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vocab_txt, report = write_farm_vocab_txt(out / "construction_vocab.txt", json_path=args.vocab_json)
    (out / "vocab_adapter_report.json").write_text(
        json.dumps(
            {
                "source": report.source,
                "n_objects": report.n_objects,
                "n_aliases_ignored": report.n_aliases_ignored,
                "prompt_names": report.prompt_names,
                "unmapped_top_level_keys": report.unmapped_top_level_keys,
                "notes": report.notes,
            },
            indent=2,
        )
    )
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        print(f"ERROR: cannot read {args.image}", file=sys.stderr)
        return 2
    h, w = image.shape[:2]
    dummy_depth = np.full((h, w), 2.0, dtype=np.float32)
    K = np.array([[float(w), 0.0, w / 2.0], [0.0, float(w), h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    cfg = AssociationConfig(use_dino_features=not args.no_dino)
    segmenter = build_farm_segmenter(vocab_txt, cfg, device="cuda" if __import__("torch").cuda.is_available() else "cpu")
    result = smoke_yoloe_masks(segmenter, image, dummy_depth, K)
    result["vocab"] = {"n_prompts": report.n_objects, "txt": str(vocab_txt)}
    (out / "yoloe_smoke.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result["n_masks"] > 0 else 3


def _video_stem(path: str | Path) -> str:
    stem = Path(path).stem
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
    return "_".join(part for part in cleaned.split("_") if part)


def _resolve_depth_source(args, image_dir: Path):
    """Select a DepthSource. Downstream mapping only sees depth_for_frame()."""
    requested = getattr(args, "depth_source", "dl")
    loader = _load_image(image_dir)
    if requested == "colmap_mvs":
        if not getattr(args, "dense_workspace", None):
            return None, {"error": "colmap_mvs_requires_dense_workspace"}
        return ColmapMvsDepthSource(args.dense_workspace), {
            "source": "colmap_mvs",
            "units": "sfm",
            "dense_workspace": args.dense_workspace,
        }
    if getattr(args, "dl_depth_npz_dir", None):
        return npz_dir_depth_source(args.dl_depth_npz_dir), {"source": "dl_depth_v1_npz_dir", "units": "m"}
    live = build_dl_depth_v1_source(image_loader=loader)
    if live is not None:
        return live, {"source": "dl_depth_v1_infer", "units": "m"}
    probe = probe_dl_depth_v1()
    if getattr(args, "allow_mvs_fallback", False) and getattr(args, "dense_workspace", None):
        logger.warning("Using --allow-mvs-fallback; prefer explicit --depth-source colmap_mvs")
        return ColmapMvsDepthSource(args.dense_workspace), {
            "source": "colmap_mvs_opt_in_fallback",
            "units": "sfm",
            "probe": probe,
        }
    return None, {"error": "dl_depth_v1_missing", **probe}


def _run_map_objects(args) -> int:
    frames = _load_poses(args.poses)
    image_dir = Path(args.images)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    depth_source, depth_meta = _resolve_depth_source(args, image_dir)
    if depth_source is None:
        print(json.dumps(depth_meta, indent=2))
        return 3
    if hasattr(depth_source, "has_depth"):
        kept = [f for f in frames if depth_source.has_depth(f.frame_name)]
        depth_meta["frames_with_depth"] = len(kept)
        depth_meta["frames_skipped_no_depth"] = len(frames) - len(kept)
        if not kept:
            print(json.dumps({"error": "no_frames_with_depth", **depth_meta}, indent=2))
            return 3
        frames = kept
    (out / "depth_source.json").write_text(json.dumps(depth_meta, indent=2))

    if args.association == "farm":
        from .farm_runtime import AssociationConfig, run_farm_association_mapping

        vocab_txt, report = write_farm_vocab_txt(out / "construction_vocab.txt", json_path=args.vocab_json)
        (out / "vocab_adapter_report.json").write_text(
            json.dumps(
                {
                    "source": report.source,
                    "n_objects": report.n_objects,
                    "n_aliases_ignored": report.n_aliases_ignored,
                    "prompt_names": report.prompt_names,
                    "unmapped_top_level_keys": report.unmapped_top_level_keys,
                    "notes": report.notes,
                },
                indent=2,
            )
        )
        result = run_farm_association_mapping(
            frames,
            _load_image(image_dir),
            depth_source.depth_for_frame,
            vocab_txt,
            out,
            cfg=AssociationConfig(use_dino_features=not args.no_dino, iou_threshold=args.iou_threshold),
        )
        print(json.dumps(result.summary, indent=2))
        return 0

    if not args.prompt_free and not args.classes:
        vocab_txt, report = write_farm_vocab_txt(out / "construction_vocab.txt", json_path=args.vocab_json)
        classes = report.prompt_names
    else:
        classes = args.classes
    detector = YOLOEDetector(
        classes=classes,
        prompt_free=args.prompt_free,
        model_id=args.yoloe_model,
        conf=args.conf,
    )
    objects, drop_log = map_detections_to_objects(
        frames,
        _load_image(image_dir),
        detector,
        depth_source,
        voxel_size=args.voxel_size,
        iou_threshold=args.iou_threshold,
        min_depth_points=args.min_depth_points,
    )
    (out / "drop_log.json").write_text(json.dumps(drop_log, indent=2))
    for obj in objects.values():
        save_object(obj, out / f"object_{obj.object_id:04d}.npz")
    checks = []
    frames_by_name = {f.frame_name: f for f in frames}
    loader = _load_image(image_dir)
    for obj in objects.values():
        if not obj.observations or obj.gaussian is None:
            continue
        frame = frames_by_name.get(obj.observations[0].frame_name)
        if frame is None:
            continue
        dets = detector.detect(loader(frame.frame_name))
        same = [d for d in dets if d.label == obj.label]
        if not same:
            checks.append({"object_id": obj.object_id, "ok": False, "reason": "no_mask"})
            continue
        mask = max(same, key=lambda d: int(d.mask.sum())).mask
        checks.append(reproject_mean_into_mask(obj, frame, mask))
    summary = {
        "num_objects": len(objects),
        "voxel_size_m_or_sfm": args.voxel_size,
        "avg_points_per_view": float(
            np.mean([o.n_points for obj in objects.values() for o in obj.observations] or [0])
        ),
        "low_depth_masks": sum(1 for s in drop_log if s.get("skipped_few_points")),
        "reprojection_checks": checks,
        "association": "greedy_iou",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def _run_e2e(args) -> int:
    if getattr(args, "outputs_root", None):
        args.work = str(Path(args.outputs_root) / _video_stem(args.video) / args.depth_source)
        logger.info("outputs-root layout → work=%s", args.work)
    if not args.work:
        print("ERROR: pass --work or --outputs-root", file=sys.stderr)
        return 2
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    report = {
        "fps_used": args.fps,
        "fps_source": "configs/ss3dgs_sfm_only.yaml general.extract_fps (default 2.0)",
        "video": args.video,
        "video_probe": probe_video(args.video),
        "depth_source_requested": args.depth_source,
        "gpu_index": str(args.gpu_index),
        "colmap_build": probe_colmap_build(),
        "dl_depth": probe_dl_depth_v1(),
        "phases": {},
    }
    (work / "e2e_report.json").write_text(json.dumps(report, indent=2))

    frames_dir = work / "frames"
    extract_kwargs = {"fps": args.fps}
    logger.info("Extracting frames at %.4g fps (ss3dgs_sfm_only.yaml default unless overridden)", args.fps)
    # Duration truncation for cheap smoke extracts.
    if args.duration_s is not None:
        import subprocess

        frames_dir.mkdir(parents=True, exist_ok=True)
        pattern = frames_dir / "frame_%06d.jpg"
        cmd = [
            "ffmpeg",
            "-y",
            "-t",
            str(args.duration_s),
            "-i",
            str(args.video),
            "-vf",
            f"fps={args.fps}",
            "-qscale:v",
            "2",
            str(pattern),
        ]
        subprocess.run(cmd, check=True)
        extracted = sorted(frames_dir.glob("frame_*.jpg"))
    else:
        extracted = extract_frames(args.video, frames_dir, fps=args.fps)
    if args.max_frames is not None:
        frames_dir = _subset_frames(frames_dir, work / f"frames_{args.max_frames}", args.max_frames)
        extracted = sorted(frames_dir.glob("frame_*.jpg"))
    report["phases"]["extract"] = {"n_frames": len(extracted), "dir": str(frames_dir)}
    (work / "e2e_report.json").write_text(json.dumps(report, indent=2))

    sfm_image_dir = frames_dir
    report["image_type"] = args.image_type
    if args.image_type == "panorama":
        faces_dir = work / "cubemap_faces"
        masks_dir = work / "cubemap_masks"
        try:
            rig = render_cubemap_faces(
                frames_dir,
                faces_dir,
                masks_dir,
                render_type=args.render_type,
            )
            report["phases"]["cubemap"] = {
                "faces_dir": str(faces_dir),
                "masks_dir": str(masks_dir),
                "render_type": args.render_type,
                "n_rig_cameras": len(rig.cameras),
            }
        except Exception as exc:
            report["phases"]["cubemap"] = {"error": repr(exc)}
            (work / "e2e_report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
            return 2
        sfm_image_dir = faces_dir
        (work / "e2e_report.json").write_text(json.dumps(report, indent=2))

    sfm_ws = work / "sfm"
    cached_model = sfm_ws / "sparse" / "0"
    reused_sfm = bool(getattr(args, "reuse_sfm", False) and (cached_model / "cameras.bin").exists())
    try:
        if reused_sfm:
            model = cached_model
            logger.info("Reusing cached SfM model %s", model)
        elif args.image_type == "panorama":
            model = run_panorama_sfm(
                sfm_image_dir,
                sfm_ws,
                rig,
                masks_dir,
                params_yaml=args.params,
            )
        else:
            model = run_sfm(frames_dir, sfm_ws, params_yaml=args.params)
        poses_path = work / "poses.json"
        verify = verify_pose_convention(model)
        frames = export_frame_poses(model)
        save_poses_json(frames, poses_path)
        ba_report_path = sfm_ws / "logs" / "ba_gpu_report.json"
        report["phases"]["sfm"] = {
            "model": str(model),
            "n_registered": len(frames),
            "verify": verify,
            "poses": str(poses_path),
            "reused": reused_sfm,
            "ba_gpu": json.loads(ba_report_path.read_text()) if ba_report_path.exists() else None,
            "ba_log": str(sfm_ws / "logs" / "bundle_adjustment.log"),
        }
    except Exception as exc:
        report["phases"]["sfm"] = {"error": repr(exc)}
        (work / "e2e_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 2
    (work / "e2e_report.json").write_text(json.dumps(report, indent=2))

    live_depth = build_dl_depth_v1_source(image_loader=_load_image(sfm_image_dir))
    npz_depth = args.dl_depth_npz_dir
    depth_ready = live_depth is not None or bool(npz_depth)
    explicit_mvs = args.depth_source == "colmap_mvs"
    if not explicit_mvs and not depth_ready and not args.allow_mvs_fallback:
        report["phases"]["depth"] = {
            "status": "blocked",
            "reason": report["dl_depth"]["blocker"],
            "drop_in": (
                "Register infer via farm_object_map.dl_depth_v1.register_infer_fn "
                "or FARM_DL_DEPTH_INFER=module:fn, or pass --dl-depth-npz-dir. "
                "For the comparison flow use --depth-source colmap_mvs."
            ),
        }
        report["phases"]["map"] = {
            "status": "not_run",
            "reason": "dl_depth_v1 not deployable; refused silent COLMAP MVS fallback",
        }
        (work / "e2e_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 3

    dense_ws = None
    map_images = str(sfm_image_dir)
    map_poses = work / "poses.json"
    if explicit_mvs or (not depth_ready and args.allow_mvs_fallback):
        dense_ws = work / "dense"
        mvs_report = undistort_and_mvs(
            sfm_image_dir,
            model,
            dense_ws,
            gpu_index=args.gpu_index,
        )
        npz_dir = work / "depth_npz"
        npz_paths = export_mvs_depth_npz(dense_ws, npz_dir)
        undist_model = dense_ws / "sparse"
        if (undist_model / "cameras.bin").exists() or (undist_model / "cameras.txt").exists():
            verify_u = verify_pose_convention(undist_model)
            frames_u = export_frame_poses(undist_model)
            map_poses = work / "poses_undistorted.json"
            save_poses_json(frames_u, map_poses)
            report["phases"]["poses_undistorted"] = {
                "n_frames": len(frames_u),
                "verify": verify_u,
                "poses": str(map_poses),
            }
        if (dense_ws / "images").is_dir():
            map_images = str(dense_ws / "images")
        report["phases"]["depth"] = {
            "status": "colmap_mvs",
            "units": "sfm",
            "dense_workspace": str(dense_ws),
            "depth_npz_dir": str(npz_dir),
            "n_depth_npz": len(npz_paths),
            "mvs": mvs_report,
        }
    else:
        report["phases"]["depth"] = {
            "status": "dl_depth_v1",
            "units": "m",
            "npz_dir": npz_depth,
            "live_infer": live_depth is not None,
        }
    (work / "e2e_report.json").write_text(json.dumps(report, indent=2))
    map_ns = argparse.Namespace(
        images=map_images,
        poses=str(map_poses),
        out=str(work / "objects"),
        association=args.association,
        depth_source="colmap_mvs" if dense_ws is not None else "dl",
        vocab_json=args.vocab_json,
        dense_workspace=str(dense_ws) if dense_ws else None,
        classes=None,
        prompt_free=False,
        yoloe_model="yoloe-v8l-seg.pt",
        voxel_size=0.05,
        iou_threshold=0.3,
        min_depth_points=30,
        conf=0.25,
        no_dino=args.no_dino,
        allow_mvs_fallback=bool(args.allow_mvs_fallback),
        dl_depth_npz_dir=None if dense_ws is not None else npz_depth,
    )
    rc = _run_map_objects(map_ns)
    report["phases"]["map"] = {"status": "ran", "returncode": rc, "out": str(work / "objects")}
    (work / "e2e_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return rc


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    if args.cmd == "probe-video":
        print(json.dumps(probe_video(args.video), indent=2))
        return 0

    if args.cmd == "extract-frames":
        info = probe_video(args.video)
        logger.info("Video probe: %s", json.dumps(info, indent=2))
        logger.info("Extract FPS=%.4g (default from ss3dgs_sfm_only.yaml is %.4g)", args.fps, DEFAULT_EXTRACT_FPS)
        if args.duration_s is not None:
            import subprocess

            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            pattern = out / "frame_%06d.jpg"
            cmd = [
                "ffmpeg",
                "-y",
                "-t",
                str(args.duration_s),
                "-i",
                str(args.video),
                "-vf",
                f"fps={args.fps}",
                "-qscale:v",
                "2",
                str(pattern),
            ]
            subprocess.run(cmd, check=True)
            frames = sorted(out.glob("frame_*.jpg"))
        else:
            frames = extract_frames(args.video, args.out, fps=args.fps)
        if args.max_frames is not None:
            for extra in frames[args.max_frames :]:
                extra.unlink()
            frames = frames[: args.max_frames]
        print(json.dumps({"extracted": len(frames), "fps": args.fps, "out": args.out}, indent=2))
        return 0

    if args.cmd == "sfm":
        model = run_sfm(args.images, args.workspace, params_yaml=args.params)
        print(json.dumps({"model": str(model)}, indent=2))
        return 0

    if args.cmd == "render-cubemap":
        rig = render_cubemap_faces(
            args.images,
            args.out,
            args.masks,
            render_type=args.render_type,
            edge_margin_px=args.edge_margin_px,
        )
        print(json.dumps({"faces": args.out, "masks": args.masks, "n_rig_cameras": len(rig.cameras)}, indent=2))
        return 0

    if args.cmd == "sfm-panorama":
        model = run_panorama_sfm(
            args.faces,
            args.workspace,
            rig_config_for_render_type(args.render_type),
            args.masks,
            params_yaml=args.params,
        )
        print(json.dumps({"model": str(model)}, indent=2))
        return 0

    if args.cmd == "export-poses":
        report = verify_pose_convention(args.model)
        frames = export_frame_poses(args.model)
        save_poses_json(frames, args.out)
        print(json.dumps({"n_frames": len(frames), "poses": args.out, "verify": report}, indent=2))
        return 0 if report["ok"] else 2

    if args.cmd == "dense-depth":
        report = undistort_and_mvs(
            args.images,
            args.model,
            args.dense_workspace,
            gpu_index=args.gpu_index,
        )
        npz_out = args.npz_out or str(Path(args.dense_workspace) / "depth_npz")
        written = export_mvs_depth_npz(args.dense_workspace, npz_out)
        print(json.dumps({"dense_workspace": args.dense_workspace, "n_npz": len(written), "mvs": report}, indent=2))
        return 0

    if args.cmd == "export-cloud":
        out_dir = Path(args.out_dir)
        ply = export_sparse_cloud(args.model, out_dir)
        npz = ply_to_cloud_npz(ply, out_dir / "cloud.npz")
        print(json.dumps({"ply": str(ply), "cloud_npz": str(npz)}, indent=2))
        return 0

    if args.cmd == "download-yoloe":
        path = ensure_yoloe_checkpoint(args.model_id)
        print(json.dumps({"checkpoint": str(path)}, indent=2))
        return 0

    if args.cmd == "smoke-yoloe":
        return _run_smoke_yoloe(args)

    if args.cmd == "map-objects":
        return _run_map_objects(args)

    if args.cmd == "e2e":
        return _run_e2e(args)

    if args.cmd == "check-dl-depth":
        print(json.dumps(probe_dl_depth_v1(), indent=2))
        return 0

    if args.cmd == "compare-flows":
        diff = compare_object_summaries(
            args.dl_summary,
            args.mvs_summary,
            max_mean_dist=args.max_mean_dist,
        )
        Path(args.out).write_text(json.dumps(diff, indent=2))
        print(json.dumps(diff, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
