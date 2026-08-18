#!/usr/bin/env python3
"""Tiny CORS-safe HTTP server for the Three.js 3D viewer.

Serves two directory trees merged at the URL root:
  1. <data-dir>/   — viewer data (bg_cloud.bin, objects.json, metadata.json, crops/)
  2. <static-dir>/ — HTML/JS assets (index.html, …)

Optional fragmentation experiment switcher:
  GET /api/experiments              → index.json experiments list
  GET /api/switch?experiment_id=…   → point data_dir at that experiment's viewer

Usage
-----
    python 3d-viewer/serve.py \\
        --data-dir outputs/fragmentation/baseline/3d-viewer \\
        --experiments-index outputs/fragmentation/index.json \\
        --port 8091 --no-open
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

_HERE = Path(__file__).resolve().parent
_STATIC_DIR = _HERE / "static"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


class _Handler(BaseHTTPRequestHandler):
    """Merge data_dir + static_dir, add CORS + cache headers."""

    data_dir: Path = Path(".")
    static_dir: Path = _STATIC_DIR
    experiments_index: Path | None = None
    current_experiment_id: str = ""

    def log_message(self, fmt: str, *args: object) -> None:
        if args and str(args[1]) not in ("200", "304", "204"):
            super().log_message(fmt, *args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path).lstrip("/") or "index.html"
        if path == "api/experiments":
            self._api_experiments()
            return
        if path == "api/switch":
            self._api_switch(parse_qs(parsed.query))
            return
        if path == "api/query":
            self._api_query_get(parse_qs(parsed.query))
            return
        self._serve(path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.lstrip("/") == "api/query":
            self._api_query_post()
            return
        self.send_error(404)

    def _api_query_get(self, qs: dict) -> None:
        q = (qs.get("q") or qs.get("query") or [""])[0].strip()
        if not q:
            self._json(400, {"ok": False, "error": "missing q"})
            return
        top_k = int((qs.get("top_k") or ["15"])[0])
        mock = (qs.get("mock") or ["0"])[0] in {"1", "true", "yes"}
        try:
            from query_api import search

            payload = search(self.data_dir, q, top_k=top_k, mock=mock)
            payload["ok"] = True
            self._json(200, payload)
        except Exception as exc:
            self._json(500, {"ok": False, "error": str(exc)})

    def _api_query_post(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self._json(400, {"ok": False, "error": "invalid JSON"})
            return
        q = str(data.get("query") or data.get("q") or "").strip()
        if not q:
            self._json(400, {"ok": False, "error": "missing query"})
            return
        top_k = int(data.get("top_k") or 15)
        mock = bool(data.get("mock"))
        try:
            from query_api import search

            payload = search(self.data_dir, q, top_k=top_k, mock=mock)
            payload["ok"] = True
            self._json(200, payload)
        except Exception as exc:
            self._json(500, {"ok": False, "error": str(exc)})

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path).lstrip("/") or "index.html"
        if path.startswith("api/"):
            self.send_error(405)
            return
        self._serve(path, head_only=True)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _load_index(self) -> dict:
        idx = self.experiments_index
        if idx is None or not idx.is_file():
            return {"experiments": [], "baseline_run_id": ""}
        return json.loads(idx.read_text(encoding="utf-8"))

    def _api_experiments(self) -> None:
        data = self._load_index()
        exps = []
        for e in data.get("experiments", []):
            exps.append(
                {
                    "id": e.get("id"),
                    "label": e.get("label") or e.get("id"),
                    "note": e.get("note") or "",
                    "kind": e.get("kind") or "experiment",
                    "n_active": e.get("n_active"),
                    "n_total": e.get("n_total"),
                    "params": e.get("params") or {},
                    "viewer_dir": e.get("viewer_dir") or "",
                }
            )
        self._json(
            200,
            {
                "baseline_run_id": data.get("baseline_run_id") or "",
                "current_experiment_id": self.current_experiment_id,
                "experiments": exps,
            },
        )

    def _api_switch(self, qs: dict) -> None:
        exp_id = (qs.get("experiment_id") or [""])[0]
        if not exp_id:
            self._json(400, {"ok": False, "error": "missing experiment_id"})
            return
        data = self._load_index()
        match = None
        for e in data.get("experiments", []):
            if e.get("id") == exp_id:
                match = e
                break
        if match is None:
            self._json(404, {"ok": False, "error": f"unknown experiment_id: {exp_id}"})
            return
        viewer = Path(match.get("viewer_dir") or "")
        if not viewer.is_dir():
            self._json(404, {"ok": False, "error": f"viewer_dir missing: {viewer}"})
            return
        type(self).data_dir = viewer.resolve()
        type(self).current_experiment_id = exp_id
        meta_path = viewer / "metadata.json"
        meta = {}
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"[serve] Switched → {exp_id} ({viewer})")
        self._json(
            200,
            {
                "ok": True,
                "experiment_id": exp_id,
                "label": match.get("label"),
                "viewer_dir": str(viewer),
                "n_active": match.get("n_active") or meta.get("n_objects_active"),
                "metadata": meta,
            },
        )

    def _serve(self, raw_path: str, head_only: bool = False) -> None:
        for segment in raw_path.split("/"):
            if segment in ("..", "~"):
                self.send_error(403)
                return

        candidate: Path | None = None
        for base in (self.data_dir, self.static_dir):
            unresolved = (base / raw_path).absolute()
            try:
                unresolved.relative_to(base.resolve())
            except ValueError:
                norm = Path(os.path.normpath(unresolved))
                try:
                    norm.relative_to(base.resolve())
                except ValueError:
                    continue
            p = unresolved.resolve()
            if p.is_file():
                candidate = p
                break

        if candidate is None:
            self.send_error(404, f"Not found: {raw_path}")
            return

        suffix = candidate.suffix.lower()
        mime = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".bin": "application/octet-stream",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".css": "text/css; charset=utf-8",
            ".txt": "text/plain; charset=utf-8",
            ".ply": "application/octet-stream",
        }.get(suffix, "application/octet-stream")

        size = candidate.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(size))
        self._cors_headers()
        if suffix in (".jpg", ".jpeg", ".png"):
            self.send_header("Cache-Control", "public, max-age=3600")
        else:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head_only:
            with open(candidate, "rb") as f:
                self.wfile.write(f.read())

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Range")


def _auto_data_dir() -> Path:
    repo = _HERE.parent
    frag = repo / "outputs" / "fragmentation" / "baseline" / "3d-viewer"
    if frag.is_dir():
        return frag
    runs_dir = repo / "outputs" / "runs"
    if runs_dir.is_dir():
        runs = sorted(p for p in runs_dir.glob("run_*/") if p.is_dir())
        if runs:
            cand = runs[-1] / "validation" / "3d-viewer"
            if cand.is_dir():
                return cand
    return Path(".")


def main() -> int:
    p = argparse.ArgumentParser(description="3D viewer HTTP server")
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Viewer data directory (default: fragmentation baseline or latest run)",
    )
    p.add_argument(
        "--experiments-index",
        type=Path,
        default=None,
        help="outputs/fragmentation/index.json — enables Run dropdown switching",
    )
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-open", action="store_true", help="Skip auto browser open")
    args = p.parse_args()

    if args.data_dir is None:
        args.data_dir = _auto_data_dir()
        print(f"[serve] Auto-selected data dir: {args.data_dir}")

    if args.experiments_index is None:
        cand = _HERE.parent / "outputs" / "fragmentation" / "index.json"
        if cand.is_file():
            args.experiments_index = cand

    if not args.data_dir.is_dir():
        print(f"[serve] ERROR: data dir does not exist: {args.data_dir}")
        print("[serve] Run:  python 3d-viewer/build_viewer_data.py  first")
        return 2

    if not _STATIC_DIR.is_dir():
        print(f"[serve] ERROR: static dir not found: {_STATIC_DIR}")
        return 2

    meta = args.data_dir / "metadata.json"
    if meta.is_file():
        m = json.loads(meta.read_text())
        print(
            f"[serve] Scene: {m.get('n_objects_active')} active objects, "
            f"{m.get('n_bg_pts', 0):,} background pts"
        )

    _Handler.data_dir = args.data_dir.resolve()
    _Handler.static_dir = _STATIC_DIR.resolve()
    _Handler.experiments_index = (
        args.experiments_index.resolve() if args.experiments_index else None
    )
    if _Handler.experiments_index and _Handler.experiments_index.is_file():
        idx = json.loads(_Handler.experiments_index.read_text(encoding="utf-8"))
        for e in idx.get("experiments", []):
            if Path(e.get("viewer_dir") or "").resolve() == _Handler.data_dir:
                _Handler.current_experiment_id = e.get("id") or ""
                break
        if not _Handler.current_experiment_id and idx.get("experiments"):
            _Handler.current_experiment_id = idx["experiments"][0].get("id") or ""

    server = HTTPServer((args.host, args.port), _Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[serve] Serving at  {url}")
    print(f"[serve] Data:       {args.data_dir}")
    print(f"[serve] Static:     {_STATIC_DIR}")
    if _Handler.experiments_index:
        print(f"[serve] Experiments: {_Handler.experiments_index}")
    print("[serve] Ctrl+C to stop.")

    if not args.no_open:
        def _open() -> None:
            time.sleep(0.6)
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[serve] Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
