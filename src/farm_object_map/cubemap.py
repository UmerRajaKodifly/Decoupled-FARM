"""Equirect → perspective cubemap faces via ss-3dgs (then FARM pinhole unprojection).

FARM itself never sees a 360 image. This module only produces pinhole RGB faces
(and COLMAP rig geometry) using ss-3dgs ``PanoProcessor`` math. Downstream
mapping calls FARM's YOLOESegmenter on those faces unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pycolmap

from .paths import ensure_ss3dgs_on_path

logger = logging.getLogger(__name__)

DEFAULT_RENDER_TYPE = "cubemap-nosfm-top-and-bottom"


def _import_ss3dgs_pano():
    ensure_ss3dgs_on_path()
    from src.pano_processing import (
        PanoProcessor,
        apply_cubemap_edge_margin_to_mask_trees,
        apply_sfm_only_camera_exclusions,
    )
    from utils.pano_utils import PANO_RENDER_OPTIONS

    return (
        PanoProcessor,
        apply_cubemap_edge_margin_to_mask_trees,
        apply_sfm_only_camera_exclusions,
        PANO_RENDER_OPTIONS,
    )


def rig_config_for_render_type(render_type: str = DEFAULT_RENDER_TYPE):
    """Rebuild the ss-3dgs virtual-camera rig without re-rendering faces."""
    _import_ss3dgs_pano()
    from src.pano_processing import create_pano_rig_config
    from utils.pano_utils import PANO_RENDER_OPTIONS, get_virtual_rotations

    if render_type not in PANO_RENDER_OPTIONS:
        raise ValueError(f"Unknown render_type {render_type!r}")
    opts = PANO_RENDER_OPTIONS[render_type]
    rots = get_virtual_rotations(opts.num_steps_yaw, opts.pitches_deg)
    return create_pano_rig_config(rots)


def render_cubemap_faces(
    pano_image_dir: str | Path,
    output_image_dir: str | Path,
    mask_dir: str | Path,
    *,
    render_type: str = DEFAULT_RENDER_TYPE,
    camera_model: str = "SIMPLE_PINHOLE",
    image_scale: float = 1.0,
    edge_margin_px: int = 0,
) -> object:
    """Render ss-3dgs cubemap faces. Returns the ``RigConfig`` used for SfM."""
    (
        PanoProcessor,
        apply_edge,
        apply_sfm_excl,
        options_table,
    ) = _import_ss3dgs_pano()
    if render_type not in options_table:
        raise ValueError(f"Unknown render_type {render_type!r}; known={sorted(options_table)}")
    render_options = options_table[render_type]
    pano_image_dir = Path(pano_image_dir)
    output_image_dir = Path(output_image_dir)
    mask_dir = Path(mask_dir)
    output_image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    names = sorted(
        p.name
        for p in pano_image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not names:
        raise FileNotFoundError(f"No panorama images in {pano_image_dir}")

    processor = PanoProcessor(
        pano_image_dir,
        output_image_dir,
        mask_dir,
        render_options,
        camera_model=camera_model,
        image_scale=image_scale,
    )
    logger.info(
        "Rendering %d panoramas as %s (%d virtual cameras)",
        len(names),
        render_type,
        len(processor.cams_from_pano_rotation),
    )
    for name in names:
        processor.process(name)

    apply_edge(mask_dir, mask_dir, edge_margin_px)
    apply_sfm_excl(mask_dir, render_options.sfm_only_mask_camera_indices)
    return processor.rig_config


def run_panorama_sfm(
    face_image_dir: str | Path,
    workspace: str | Path,
    rig_config,
    mask_dir: str | Path,
    *,
    params_yaml: str | Path,
) -> Path:
    """COLMAP SfM on cubemap faces with ss-3dgs rig + panorama matcher options."""
    from .sfm import _import_ss3dgs

    colmap_sfm, _, load_pipeline_params = _import_ss3dgs()
    params = load_pipeline_params(params_yaml)
    colmap_sfm.apply_pipeline_params(params)

    face_image_dir = Path(face_image_dir)
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    database_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    colmap_sfm.clean_sfm_outputs(database_path, sparse_dir)

    reader = pycolmap.ImageReaderOptions()
    reader.mask_path = str(mask_dir)
    reader.camera_model = "SIMPLE_PINHOLE"

    logger.info("Panorama SfM feature extraction on %s", face_image_dir)
    colmap_sfm.extract_features(
        database_path,
        face_image_dir,
        reader_options=reader,
        camera_mode=pycolmap.CameraMode.PER_FOLDER,
    )
    with pycolmap.Database.open(database_path) as db:
        pycolmap.apply_rig_config([rig_config], db)

    matcher = colmap_sfm.normalize_sfm_matcher(params["sfm"]["matcher"])
    matching_options = colmap_sfm.build_panorama_matching_options(
        gpu_index=params["sfm"]["gpu_index"],
        matcher_type=params["sfm"]["matcher_type"],
        match_preset=params["sfm"]["match_preset"],
        guided_matching=bool(params["sfm"]["guided_matching"]),
        feature_extractor=params["sfm"]["feature_extractor"],
    )
    logger.info("Panorama SfM matching: %s (loop_closure=%s)", matcher, params["sfm"]["loop_closure"])
    colmap_sfm.run_matcher(
        database_path,
        matcher,
        matching_options=matching_options,
        loop_detection=bool(params["sfm"]["loop_closure"]),
    )

    ba_backend = params["sfm"]["ba_backend"]
    logger.info("Panorama SfM mapping (ba_backend=%s)", ba_backend)
    colmap_sfm.run_mapping(
        database_path,
        face_image_dir,
        sparse_dir,
        sfm_mapper=params["sfm"]["mapper"],
        ba_backend=ba_backend,
        incremental_pipeline_options=colmap_sfm.build_panorama_incremental_pipeline_options(
            ba_backend=ba_backend
        ),
        global_pipeline_options=colmap_sfm.build_panorama_global_pipeline_options(
            ba_backend=ba_backend
        ),
        calibrate_view_graph_before_global=True,
    )
    colmap_sfm.export_model(sparse_dir / "0")
    model_path = sparse_dir / "0"
    if not (model_path / "cameras.bin").exists() and not (model_path / "cameras.txt").exists():
        raise RuntimeError(f"Panorama SfM produced no model at {model_path}")
    return model_path
