#!/usr/bin/env python3
"""Background experiment runner for the compare-viewer UI."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

_REPO = Path(__file__).resolve().parents[2]
_STATUS_PATH = _REPO / "outputs" / "compare" / "job_status.json"
_LOCK = threading.Lock()
_PROC: Optional[subprocess.Popen] = None


@dataclass
class ExperimentParams:
    yoloe_conf: float = 0.35
    label_min_score: float = 0.25
    label_margin: float = 1.15
    vocab: str = "construction_vocab.txt"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentParams":
        return cls(
            yoloe_conf=float(d.get("yoloe_conf", 0.35)),
            label_min_score=float(d.get("label_min_score", 0.25)),
            label_margin=float(d.get("label_margin", 1.15)),
            vocab=str(d.get("vocab", "construction_vocab.txt")),
        )


@dataclass
class JobStatus:
    status: str = "idle"  # idle | running | complete | error
    baseline_run_id: str = ""
    experiment_run_id: str = ""
    manifest_path: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    log_path: str = ""
    message: str = ""
    started_at: str = ""
    finished_at: str = ""
    exit_code: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _read_baseline_run_id() -> str:
    manifest = _REPO / "outputs" / "baselines" / "manifest.json"
    if manifest.is_file():
        return json.loads(manifest.read_text())["baseline_run_id"]
    latest = _REPO / "outputs" / "baseline"
    if latest.is_symlink() or latest.exists():
        return Path(os.readlink(latest) if latest.is_symlink() else latest.name).name
    return ""


def _load_status() -> JobStatus:
    if _STATUS_PATH.is_file():
        data = json.loads(_STATUS_PATH.read_text())
        return JobStatus(**{k: data[k] for k in JobStatus.__dataclass_fields__ if k in data})
    return JobStatus(baseline_run_id=_read_baseline_run_id())


def _save_status(st: JobStatus) -> None:
    _STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATUS_PATH.write_text(json.dumps(st.to_dict(), indent=2))


def get_status() -> Dict[str, Any]:
    with _LOCK:
        return _load_status().to_dict()


def list_vocabs() -> list[Dict[str, str]]:
    vocab_dir = _REPO / "vocab"
    out = []
    for p in sorted(vocab_dir.glob("*.txt")):
        n = len([ln for ln in p.read_text().splitlines() if ln.strip()])
        out.append({"file": p.name, "path": str(p), "n_classes": n})
    return out


def get_config() -> Dict[str, Any]:
    st = _load_status()
    last_params = ExperimentParams()
    last_run = _REPO / "outputs" / "compare" / "last_run.json"
    if last_run.is_file():
        last_params = ExperimentParams.from_dict(
            json.loads(last_run.read_text()).get("params", {})
        )
    elif st.params:
        last_params = ExperimentParams.from_dict(st.params)

    manifest_path = _REPO / "outputs" / "compare" / "latest" / "manifest.json"
    current_manifest = str(manifest_path) if manifest_path.is_file() else ""

    return {
        "baseline_run_id": _read_baseline_run_id(),
        "defaults": asdict(ExperimentParams()),
        "current_params": asdict(last_params),
        "vocabs": list_vocabs(),
        "manifest_path": current_manifest,
        "job": st.to_dict(),
    }


def _tail_log(path: Path, n: int = 30) -> str:
    if not path.is_file():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def _watch_process(proc: subprocess.Popen, log_path: Path, params: ExperimentParams) -> None:
    global _PROC
    exit_code = proc.wait()
    _PROC = None
    st = _load_status()
    st.exit_code = exit_code
    st.finished_at = datetime.now(timezone.utc).isoformat()
    st.message = _tail_log(log_path, 40)

    if exit_code != 0:
        st.status = "error"
        st.message = f"Pipeline failed (exit {exit_code}).\n" + st.message
    else:
        last_run = _REPO / "outputs" / "compare" / "last_run.json"
        if last_run.is_file():
            meta = json.loads(last_run.read_text())
            st.status = "complete"
            st.experiment_run_id = meta.get("experiment_run_id", "")
            st.manifest_path = meta.get("manifest", "")
            st.message = f"Run {st.experiment_run_id} complete."
            params = meta.get("params", {})
            try:
                from ablation_registry import register_experiment

                mf = Path(meta.get("manifest", ""))
                if not mf.is_file():
                    aid = __import__("ablation_registry", fromlist=["ablation_id"]).ablation_id(
                        float(params.get("yoloe_conf", 0.35)),
                        float(params.get("label_min_score", 0.25)),
                        float(params.get("label_margin", 1.15)),
                    )
                    mf = _REPO / "outputs" / "ablation" / "manifests" / f"{aid}.json"
                register_experiment(
                    conf=float(params.get("yoloe_conf", 0.35)),
                    vote=float(params.get("label_min_score", 0.25)),
                    margin=float(params.get("label_margin", 1.15)),
                    run_id=st.experiment_run_id,
                    manifest_path=mf,
                    baseline_run_id=st.baseline_run_id,
                    vocab=Path(params.get("vocab", "construction_vocab.txt")).name,
                    note="manual rerun",
                )
            except Exception:
                pass
        else:
            st.status = "error"
            st.message = "Pipeline finished but last_run.json missing."

    _save_status(st)


def start_experiment(params: ExperimentParams) -> Dict[str, Any]:
    global _PROC
    with _LOCK:
        if _PROC is not None and _PROC.poll() is None:
            return {"ok": False, "error": "A run is already in progress."}

        baseline = _read_baseline_run_id()
        if not baseline:
            return {"ok": False, "error": "No baseline pinned. Run scripts/snapshot_baseline.sh first."}

        vocab_path = _REPO / "vocab" / Path(params.vocab).name
        if not vocab_path.is_file():
            return {"ok": False, "error": f"Vocab not found: {params.vocab}"}

        for name, val, lo, hi in [
            ("yoloe_conf", params.yoloe_conf, 0.05, 0.95),
            ("label_min_score", params.label_min_score, 0.0, 0.95),
            ("label_margin", params.label_margin, 1.0, 3.0),
        ]:
            if not (lo <= val <= hi):
                return {"ok": False, "error": f"{name}={val} out of range [{lo}, {hi}]"}

        log_path = _REPO / "outputs" / "logs" / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.update({
            "BASELINE_RUN_ID": baseline,
            "CONSTRUCTION_VOCAB": str(vocab_path),
            "YOLOE_CONF": str(params.yoloe_conf),
            "LABEL_MIN_SCORE": str(params.label_min_score),
            "LABEL_MARGIN": str(params.label_margin),
            "EXPERIMENT_LABEL": (
                f"Experiment (conf={params.yoloe_conf}, vote={params.label_min_score})"
            ),
        })

        script = _REPO / "scripts" / "run_experiment.sh"
        log_f = open(log_path, "w")
        proc = subprocess.Popen(
            ["bash", str(script)],
            cwd=str(_REPO),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        _PROC = proc

        st = JobStatus(
            status="running",
            baseline_run_id=baseline,
            params=asdict(params),
            log_path=str(log_path),
            message="Pipeline started…",
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        _save_status(st)
        threading.Thread(
            target=_watch_process, args=(proc, log_path, params), daemon=True
        ).start()
        return {"ok": True, "job": st.to_dict()}


def refresh_running_status() -> Dict[str, Any]:
    """If process died without watcher, update status."""
    global _PROC
    with _LOCK:
        st = _load_status()
        if st.status == "running" and (_PROC is None or _PROC.poll() is not None):
            if _PROC is not None:
                _watch_process(_PROC, Path(st.log_path), ExperimentParams.from_dict(st.params))
            else:
                st.status = "error"
                st.message = "Run process lost."
                _save_status(st)
        return st.to_dict()
