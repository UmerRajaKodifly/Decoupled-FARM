"""Capture and parse COLMAP GPU evidence for BA and dense stereo."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


def resolve_colmap_bin() -> Path:
    env_bin = os.environ.get("COLMAP_BIN") or os.environ.get("COLMAP_EXE")
    if env_bin:
        return Path(env_bin).expanduser().resolve()
    root = os.environ.get("COLMAP_ROOT")
    if root:
        candidate = Path(root).expanduser() / "bin" / "colmap"
        if candidate.is_file():
            return candidate.resolve()
    which = shutil.which("colmap")
    if which:
        return Path(which).resolve()
    raise FileNotFoundError(
        "colmap not on PATH. Source env.sh or set COLMAP_ROOT / COLMAP_BIN."
    )


def probe_colmap_build(colmap_bin: str | Path | None = None) -> dict:
    bin_path = Path(colmap_bin) if colmap_bin else resolve_colmap_bin()
    proc = subprocess.run(
        [str(bin_path), "help"],
        check=False,
        capture_output=True,
        text=True,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    version_line = next((ln for ln in lines if "COLMAP" in ln), lines[0] if lines else "")
    has_cuda = "with CUDA" in text
    looks_37 = bool(re.search(r"\b3\.7\b", version_line)) and "4.1" not in version_line
    return {
        "bin": str(bin_path),
        "bin_resolved": str(bin_path.resolve()),
        "version_line": version_line,
        "has_cuda": has_cuda,
        "looks_like_system_3_7": looks_37,
        "ok": has_cuda and not looks_37,
    }


def parse_mvs_gpu_log(log_text: str, *, requested_gpu_index: str) -> dict:
    lines = log_text.splitlines()
    cuda_lines = [
        ln
        for ln in lines
        if re.search(r"cuda|gpu", ln, re.I)
        and not re.search(r"PatchMatchStereo\.gpu_index", ln)
    ]
    fallback_lines = [
        ln
        for ln in lines
        if re.search(
            r"no cuda|without cuda|cpu only|falling back|fallback|cuda initialization failed|"
            r"invalid cuda|switching to cpu",
            ln,
            re.I,
        )
    ]
    device_hits = re.findall(
        r"(?:CUDA device|GPU(?: index)?|gpu_index)\s*[:=]?\s*(\d+)",
        log_text,
        flags=re.I,
    )
    return {
        "requested_gpu_index": str(requested_gpu_index),
        "observed_device_indices": device_hits,
        "cuda_log_lines": cuda_lines[:40],
        "fallback_warnings": fallback_lines,
        "used_cuda": bool(cuda_lines) and not fallback_lines,
    }


def parse_ba_gpu_log(log_text: str, *, requested_gpu_index: str) -> dict:
    lines = log_text.splitlines()
    caspar_lines = [ln for ln in lines if re.search(r"caspar", ln, re.I)]
    ceres_fallback = [
        ln
        for ln in lines
        if re.search(r"caspar requested but|ceres fallback", ln, re.I)
    ]
    gpu_lines = [ln for ln in lines if re.search(r"gpu_index\s*=\s*\d+", ln, re.I)]
    indices = re.findall(r"gpu_index\s*=\s*(\d+)", log_text, flags=re.I)
    caspar_selected = any(re.search(r"✓.*caspar|global ba backend:\s*caspar", ln, re.I) for ln in lines)
    return {
        "requested_gpu_index": str(requested_gpu_index),
        "observed_gpu_indices": indices,
        "caspar_selected": caspar_selected or (
            any(re.search(r"caspar", ln, re.I) for ln in caspar_lines) and not ceres_fallback
        ),
        "caspar_log_lines": caspar_lines[:40],
        "gpu_index_log_lines": gpu_lines[:40],
        "fallback_warnings": ceres_fallback,
    }


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return False


class _ListHandler(logging.Handler):
    def __init__(self, bucket: list[str]) -> None:
        super().__init__(level=logging.DEBUG)
        self.bucket = bucket
        self.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self.bucket.append(self.format(record))


@contextmanager
def capture_process_logs(log_path: str | Path, *logger_names: str) -> Iterator[list[str]]:
    """Tee stderr + selected Python loggers into ``log_path``."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    bucket: list[str] = []
    handlers: list[tuple[logging.Logger, logging.Handler]] = []
    for name in ("", *logger_names):
        logger = logging.getLogger(name)
        handler = _ListHandler(bucket)
        logger.addHandler(handler)
        handlers.append((logger, handler))
    with log_path.open("w", encoding="utf-8") as fh:
        old_err, old_out = sys.stderr, sys.stdout
        sys.stderr = _Tee(old_err, fh)  # type: ignore[assignment]
        sys.stdout = _Tee(old_out, fh)  # type: ignore[assignment]
        try:
            yield bucket
            if bucket:
                fh.write("\n--- python logging ---\n")
                fh.write("\n".join(bucket) + "\n")
        finally:
            sys.stderr = old_err
            sys.stdout = old_out
            for logger, handler in handlers:
                logger.removeHandler(handler)
