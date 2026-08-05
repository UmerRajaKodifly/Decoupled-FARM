"""COLMAP dense stereo depth behind the DepthSource interface."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .depth import DepthMap, DepthSource
from .gpu_verify import parse_mvs_gpu_log, probe_colmap_build, resolve_colmap_bin

logger = logging.getLogger(__name__)


def _run_logged(cmd: list[str], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Running: %s", " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("CMD: " + " ".join(cmd) + "\n")
        fh.write(f"argv0_resolved: {Path(cmd[0]).resolve()}\n\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        chunks: list[str] = []
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            chunks.append(line.rstrip("\n"))
            logger.info("colmap: %s", line.rstrip("\n"))
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}; see {log_path}")
    return "\n".join(chunks)


def read_colmap_depth_bin(path: str | Path) -> np.ndarray:
    """Read COLMAP ``*.geometric.bin`` / ``*.photometric.bin`` depth maps.

    Format (little-endian): text header ``width&height&channels&`` then
    ``width * height * channels`` float32 values, row-major.
    """
    data = Path(path).read_bytes()
    header_end = 0
    amps = 0
    for i, byte in enumerate(data):
        if byte == ord("&"):
            amps += 1
            if amps == 3:
                header_end = i + 1
                break
    header = data[: header_end - 1].decode("ascii")
    width_s, height_s, channels_s = header.split("&")
    width, height, channels = int(width_s), int(height_s), int(channels_s)
    payload = data[header_end:]
    expected = width * height * channels * 4
    if len(payload) < expected:
        raise ValueError(f"{path}: expected {expected} bytes of floats, got {len(payload)}")
    arr = np.frombuffer(payload[:expected], dtype=np.float32).reshape(height, width, channels)
    return arr[:, :, 0]


def undistort_and_mvs(
    image_dir: str | Path,
    sparse_model: str | Path,
    dense_workspace: str | Path,
    *,
    max_image_size: int = 2000,
    gpu_index: str | int = "0",
    skip_if_present: bool = True,
) -> dict:
    """Run ``image_undistorter`` + ``patch_match_stereo``. Returns a GPU report."""
    image_dir = Path(image_dir)
    sparse_model = Path(sparse_model)
    dense_workspace = Path(dense_workspace)
    dense_workspace.mkdir(parents=True, exist_ok=True)
    gpu_index = str(gpu_index)
    colmap_bin = str(resolve_colmap_bin())
    build = probe_colmap_build(colmap_bin)
    smi = subprocess.run(["nvidia-smi", "-L"], check=False, capture_output=True, text=True)
    build["nvidia_smi_L"] = (smi.stdout or smi.stderr or "").strip().splitlines()
    if not build["ok"]:
        raise RuntimeError(
            "Refusing dense stereo: need COLMAP 4.1+CUDA, got "
            f"{build['version_line']} at {build['bin_resolved']}"
        )

    logs_dir = dense_workspace / "logs"
    depth_dir = dense_workspace / "stereo" / "depth_maps"
    already = skip_if_present and any(depth_dir.glob("*.geometric.bin"))
    mvs_log_text = ""
    undistort_log = ""
    if already:
        logger.info("Reusing existing geometric depth maps under %s", depth_dir)
        cached_log = logs_dir / "patch_match_stereo.log"
        mvs_log_text = cached_log.read_text(encoding="utf-8") if cached_log.exists() else ""
    else:
        undistort_log = _run_logged(
            [
                colmap_bin,
                "image_undistorter",
                "--image_path",
                str(image_dir),
                "--input_path",
                str(sparse_model),
                "--output_path",
                str(dense_workspace),
                "--output_type",
                "COLMAP",
                "--max_image_size",
                str(max_image_size),
            ],
            logs_dir / "image_undistorter.log",
        )
        mvs_log_text = _run_logged(
            [
                colmap_bin,
                "patch_match_stereo",
                "--workspace_path",
                str(dense_workspace),
                "--workspace_format",
                "COLMAP",
                "--PatchMatchStereo.geom_consistency",
                "1",
                "--PatchMatchStereo.gpu_index",
                gpu_index,
            ],
            logs_dir / "patch_match_stereo.log",
        )

    mvs_gpu = parse_mvs_gpu_log(mvs_log_text, requested_gpu_index=gpu_index)
    report = {
        "dense_workspace": str(dense_workspace),
        "colmap": build,
        "gpu_index": gpu_index,
        "reused_existing_depth": already,
        "undistorter_log": str(logs_dir / "image_undistorter.log"),
        "patch_match_log": str(logs_dir / "patch_match_stereo.log"),
        "mvs_gpu": mvs_gpu,
        "undistorter_tail": undistort_log.splitlines()[-20:] if undistort_log else [],
    }
    (logs_dir / "gpu_report.json").write_text(
        __import__("json").dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def export_mvs_depth_npz(dense_workspace: str | Path, out_dir: str | Path) -> list[Path]:
    """Convert every geometric (else photometric) depth map into the DepthMap npz contract."""
    source = ColmapMvsDepthSource(dense_workspace)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bins = sorted(source.depth_dir.glob("*.geometric.bin"))
    if not bins:
        bins = sorted(source.depth_dir.rglob("*.geometric.bin"))
    if not bins:
        bins = sorted(source.depth_dir.rglob("*.photometric.bin"))
    written: list[Path] = []
    for bin_path in bins:
        rel = bin_path.relative_to(source.depth_dir)
        stem = str(rel).removesuffix(".geometric.bin").removesuffix(".photometric.bin")
        depth_map = source.depth_for_frame(stem)
        dest = out_dir / (stem.replace("/", "__").replace("\\", "__") + ".npz")
        dest.parent.mkdir(parents=True, exist_ok=True)
        depth_map.save_npz(dest)
        written.append(dest)
    if not written:
        raise FileNotFoundError(f"No COLMAP depth maps under {source.depth_dir}")
    return written


class ColmapMvsDepthSource:
    """Load per-frame geometric depth written by ``patch_match_stereo``."""

    source_id = "colmap_mvs"
    units = "sfm"

    def __init__(self, dense_workspace: str | Path, *, rgb_hw: tuple[int, int] | None = None):
        self.dense_workspace = Path(dense_workspace)
        self.depth_dir = self.dense_workspace / "stereo" / "depth_maps"
        self.undistorted_images = self.dense_workspace / "images"
        self.rgb_hw = rgb_hw

    def has_depth(self, frame_name: str) -> bool:
        try:
            self._depth_path(frame_name)
            return True
        except FileNotFoundError:
            return False

    def _depth_path(self, frame_name: str) -> Path:
        dotted = frame_name.replace("/", ".").replace("\\", ".")
        candidates = [
            self.depth_dir / f"{frame_name}.geometric.bin",
            self.depth_dir / f"{frame_name}.photometric.bin",
            self.depth_dir / f"{dotted}.geometric.bin",
            self.depth_dir / f"{dotted}.photometric.bin",
        ]
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(f"No COLMAP depth map for {frame_name} under {self.depth_dir}")

    def depth_for_frame(self, frame_name: str) -> DepthMap:
        depth = read_colmap_depth_bin(self._depth_path(frame_name))
        if self.rgb_hw is not None and depth.shape != self.rgb_hw:
            depth = cv2.resize(
                depth,
                (self.rgb_hw[1], self.rgb_hw[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        valid = np.isfinite(depth) & (depth > 0)
        h, w = depth.shape
        return DepthMap(
            depth_m=depth.astype(np.float32),
            valid_mask=valid,
            frame_hw=(h, w),
            units=self.units,
            source=self.source_id,
        )


_: type[DepthSource] = ColmapMvsDepthSource
