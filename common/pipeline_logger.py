"""Run-scoped logging helpers with tqdm-friendly output.

Usage
-----
from pipeline_logger import setup_logger, log_stage_banner, log_elapsed, tqdm_kwargs

log = setup_logger("phase2", stage="phase2")
log_stage_banner(log, "Phase 2 — Detect / Segment / Embed")
t0 = time.time()
for item in tqdm(items, **tqdm_kwargs(desc="Phase 2 detect")):
    ...
log_elapsed(log, t0, "Phase 2")
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _run_id() -> str:
    return os.environ.get("PIPELINE_RUN_ID", "").strip() or "run_local"


def _log_root() -> Path:
    env = os.environ.get("PIPELINE_LOG_DIR", "").strip()
    if env:
        return Path(env)
    # Prefer /outputs/logs when in container; else repo/outputs/logs
    for candidate in (
        Path("/outputs/logs"),
        Path(__file__).resolve().parent.parent / "outputs" / "logs",
    ):
        if candidate.parent.exists() or candidate.exists():
            return candidate
    return Path("./outputs/logs")


def run_log_dir(run_id: Optional[str] = None) -> Path:
    rid = run_id or _run_id()
    d = _log_root() / rid
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup_logger(
    name: str,
    *,
    stage: str,
    level: int = logging.INFO,
    run_id: Optional[str] = None,
) -> logging.Logger:
    """Configure logger that writes to stdout and stage log file."""
    rid = run_id or _run_id()
    log_dir = run_log_dir(rid)
    log_path = log_dir / f"{stage}.log"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        fmt=f"%(asctime)s [{stage}] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Use UTC-ish ISO feel via local wall time; docker hosts use host TZ
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(level)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    # Mirror to merged pipeline.log
    merged = log_dir / "pipeline.log"
    mh = logging.FileHandler(merged, mode="a", encoding="utf-8")
    mh.setFormatter(fmt)
    mh.setLevel(level)
    logger.addHandler(mh)

    logger.debug("logger ready → %s", log_path)
    return logger


def log_stage_banner(logger: logging.Logger, title: str) -> None:
    bar = "=" * 64
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    logger.info(bar)
    logger.info("%s", title)
    logger.info("run_id=%s  wall=%s", _run_id(), ts)
    logger.info(bar)


def log_elapsed(logger: logging.Logger, t0: float, label: str) -> float:
    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    hrs, mins = divmod(mins, 60)
    if hrs:
        human = f"{hrs}h {mins}m {secs}s"
    elif mins:
        human = f"{mins}m {secs}s"
    else:
        human = f"{secs}s"
    logger.info("%s finished in %s (%.2fs)", label, human, elapsed)
    return elapsed


def tqdm_kwargs(desc: str = "", **extra: Any) -> Dict[str, Any]:
    """Standard tqdm kwargs: leave bars + play nice with logging."""
    kwargs: Dict[str, Any] = {
        "desc": desc,
        "dynamic_ncols": True,
        "leave": True,
        "file": sys.stdout,
        "mininterval": 0.5,
    }
    kwargs.update(extra)
    return kwargs
