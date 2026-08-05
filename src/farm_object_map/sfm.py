"""Run COLMAP SfM via ss-3dgs pycolmap helpers (3DGS never invoked)."""

from __future__ import annotations

import logging
from pathlib import Path

import pycolmap

from .paths import ensure_ss3dgs_on_path

logger = logging.getLogger(__name__)


def _import_ss3dgs():
    ensure_ss3dgs_on_path()
    from utils import colmap_sfm
    from utils.colmap_export import export_colmap_sfm_artifacts
    from utils.pipeline_params import load_pipeline_params

    return colmap_sfm, export_colmap_sfm_artifacts, load_pipeline_params


def run_sfm(
    image_dir: str | Path,
    workspace: str | Path,
    *,
    params_yaml: str | Path,
    camera_model: str = "PINHOLE",
) -> Path:
    """Feature extract → match → incremental map with Caspar when available.

    Returns the sparse model directory (``workspace/sparse/0``).
    """
    colmap_sfm, _, load_pipeline_params = _import_ss3dgs()
    params = load_pipeline_params(params_yaml)
    colmap_sfm.apply_pipeline_params(params)

    image_dir = Path(image_dir)
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    database_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    colmap_sfm.clean_sfm_outputs(database_path, sparse_dir)

    reader = pycolmap.ImageReaderOptions()
    if hasattr(reader, "camera_model"):
        reader.camera_model = camera_model

    logger.info("SfM feature extraction on %s (camera_model=%s)", image_dir, camera_model)
    colmap_sfm.extract_features(
        database_path,
        image_dir,
        reader_options=reader,
        camera_mode=pycolmap.CameraMode.SINGLE,
    )

    matcher = colmap_sfm.normalize_sfm_matcher(params["sfm"]["matcher"])
    logger.info("SfM matching: %s", matcher)
    colmap_sfm.run_matcher(
        database_path,
        matcher,
        loop_detection=bool(params["sfm"]["loop_closure"]),
    )

    ba_backend = params["sfm"]["ba_backend"]
    logger.info("SfM mapping (requested ba_backend=%s)", ba_backend)
    colmap_sfm.run_incremental_mapping(
        database_path,
        image_dir,
        sparse_dir,
        pipeline_options=colmap_sfm.build_incremental_pipeline_options(
            ba_backend=ba_backend
        ),
        ba_backend=ba_backend,
    )
    colmap_sfm.export_model(sparse_dir / "0")
    model_path = sparse_dir / "0"
    if not (model_path / "cameras.bin").exists() and not (model_path / "cameras.txt").exists():
        raise RuntimeError(f"SfM produced no model at {model_path}")
    return model_path


def export_sparse_cloud(model_dir: str | Path, exports_dir: str | Path) -> Path:
    _, export_colmap_sfm_artifacts, _ = _import_ss3dgs()
    model_dir = Path(model_dir)
    exports_dir = Path(exports_dir)
    processed = model_dir.parent.parent  # .../colmap or workspace
    # export_colmap_sfm_artifacts searches processed_data_dir/colmap/sparse and /sparse
    # Pass the workspace that contains sparse/0.
    workspace = model_dir.parent.parent if model_dir.parent.name == "sparse" else model_dir.parent
    export_colmap_sfm_artifacts(
        workspace,
        exports_dir,
        model_path=model_dir,
        colmap_cmd=["colmap", "model_converter"],
        export_trajectory=True,
    )
    ply = exports_dir / "sparse_pointcloud.ply"
    return ply
