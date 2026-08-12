#!/usr/bin/env python3
"""Tiny CORS-safe HTTP server for the Three.js 3D viewer.

Serves two directory trees merged at the URL root:
  1. <data-dir>/   — viewer data (bg_cloud.bin, objects.json, metadata.json, crops/)
  2. <static-dir>/ — HTML/JS assets (index.html, …)

Data files take precedence; static files fill in the rest (so index.html is at /).

Usage
-----
    conda activate farm-phase2   # or any env with Python 3.8+
    cd /home/kodifly/Desktop/farm-git/repo

    python 3d-viewer/serve.py
        --data-dir outputs/runs/run_XXXX/validation/3d-viewer
        --port     8090
        --no-open  (skip auto browser open)
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote

_HERE = Path(__file__).resolve().parent
_STATIC_DIR = _HERE / "static"


class _Handler(BaseHTTPRequestHandler):
    """Merge data_dir + static_dir, add CORS + cache headers."""

    data_dir: Path = Path(".")
    static_dir: Path = _STATIC_DIR

    def log_message(self, fmt: str, *args: object) -> None:
        # Suppress per-request noise; only errors shown
        if args and str(args[1]) not in ("200", "304"):
            super().log_message(fmt, *args)

    def do_GET(self) -> None:
        self._serve()

    def do_HEAD(self) -> None:
        self._serve(head_only=True)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def _serve(self, head_only: bool = False) -> None:
        raw_path = unquote(self.path.split("?", 1)[0]).lstrip("/") or "index.html"

        # Security: reject path traversal
        for segment in raw_path.split("/"):
            if segment in ("..", "~"):
                self.send_error(403)
                return

        # Resolve: data_dir first, then static_dir.
        # We check the *unresolved* path stays under base to support symlinked
        # subdirs (e.g. crops/ symlinked to a different run directory).
        candidate: Path | None = None
        for base in (self.data_dir, self.static_dir):
            unresolved = (base / raw_path).absolute()
            try:
                unresolved.relative_to(base.resolve())
            except ValueError:
                # The unresolved path escapes the base — check one more time
                # with normpath in case of ../ tricks without symlinks
                import os
                norm = Path(os.path.normpath(unresolved))
                try:
                    norm.relative_to(base.resolve())
                except ValueError:
                    continue  # genuine path escape attempt
            p = unresolved.resolve()  # follow symlinks for actual file check
            if p.is_file():
                candidate = p
                break

        if candidate is None:
            self.send_error(404, f"Not found: {raw_path}")
            return

        # MIME
        suffix = candidate.suffix.lower()
        mime = {
            ".html": "text/html; charset=utf-8",
            ".js":   "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".bin":  "application/octet-stream",
            ".jpg":  "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png":  "image/png",
            ".css":  "text/css; charset=utf-8",
            ".txt":  "text/plain; charset=utf-8",
            ".ply":  "application/octet-stream",
        }.get(suffix, "application/octet-stream")

        size = candidate.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(size))
        self._cors_headers()
        # Light caching for images; no cache for JSON/bin (data may be rebuilt)
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
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Range")


def _auto_data_dir() -> Path:
    repo = _HERE.parent
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
        "--data-dir", type=Path, default=None,
        help="Viewer data directory (default: latest run validation/3d-viewer)",
    )
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-open", action="store_true", help="Skip auto browser open")
    args = p.parse_args()

    if args.data_dir is None:
        args.data_dir = _auto_data_dir()
        print(f"[serve] Auto-selected data dir: {args.data_dir}")

    if not args.data_dir.is_dir():
        print(f"[serve] ERROR: data dir does not exist: {args.data_dir}")
        print(f"[serve] Run:  python 3d-viewer/build_viewer_data.py  first")
        return 2

    if not _STATIC_DIR.is_dir():
        print(f"[serve] ERROR: static dir not found: {_STATIC_DIR}")
        return 2

    meta = args.data_dir / "metadata.json"
    if meta.is_file():
        import json
        m = json.loads(meta.read_text())
        print(f"[serve] Scene: {m.get('n_objects_active')} active objects, "
              f"{m.get('n_bg_pts', 0):,} background pts")

    # Bind handler to dirs
    _Handler.data_dir = args.data_dir.resolve()
    _Handler.static_dir = _STATIC_DIR.resolve()

    server = HTTPServer((args.host, args.port), _Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"[serve] Serving at  {url}")
    print(f"[serve] Data:       {args.data_dir}")
    print(f"[serve] Static:     {_STATIC_DIR}")
    print(f"[serve] Ctrl+C to stop.")

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
