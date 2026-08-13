#!/usr/bin/env python3
"""Serve A/B compare viewer + experiment re-run API."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
_STATIC = _HERE / "static"

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ablation_registry import get_experiment, load_index, load_manifest_for_experiment, register_experiment  # noqa: E402
from experiment_runner import (  # noqa: E402
    ExperimentParams,
    get_config,
    get_status,
    refresh_running_status,
    start_experiment,
)


class _Handler(BaseHTTPRequestHandler):
    manifest_path: Path = Path("manifest.json")
    baseline_dir: Path = Path(".")
    experiment_dir: Path = Path(".")
    baseline_run_dir: Path = Path(".")
    experiment_run_dir: Path = Path(".")

    def log_message(self, fmt: str, *args: object) -> None:
        if args and str(args[1]) not in ("200", "304", "204"):
            super().log_message(fmt, *args)

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_response(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _read_file(self, base: Path, rel: str) -> tuple[bytes | None, str]:
        candidate = base / rel
        if not candidate.exists():
            return None, ""
        try:
            candidate.absolute().relative_to(base.resolve())
        except ValueError:
            return None, ""
        data = candidate.read_bytes()
        ctype = "application/octet-stream"
        low = rel.lower()
        if low.endswith(".json"):
            ctype = "application/json"
        elif low.endswith(".html"):
            ctype = "text/html"
        elif low.endswith(".js"):
            ctype = "application/javascript"
        elif low.endswith(".css"):
            ctype = "text/css"
        elif low.endswith((".jpg", ".jpeg")):
            ctype = "image/jpeg"
        elif low.endswith(".png"):
            ctype = "image/png"
        return data, ctype

    def _read_side(self, side: str, rel: str) -> tuple[bytes | None, str]:
        base = self.baseline_dir if side == "a" else self.experiment_dir
        run_dir = self.baseline_run_dir if side == "a" else self.experiment_run_dir
        data, ctype = self._read_file(base, rel)
        if data is None and rel.startswith("crops/"):
            fallback = run_dir / "phase4" / rel
            if fallback.is_file():
                data = fallback.read_bytes()
                ctype = "image/jpeg"
        return data, ctype

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        raw = unquote(self.path.split("?", 1)[0]).lstrip("/") or "index.html"

        if raw == "api/config":
            refresh_running_status()
            return self._json_response(200, get_config())
        if raw == "api/status":
            return self._json_response(200, refresh_running_status())
        if raw == "api/ablations":
            return self._json_response(200, load_index())
        if raw == "api/manifest":
            qs = parse_qs(urlparse(self.path).query)
            exp_id = (qs.get("experiment_id") or [None])[0]
            manifest = None
            if exp_id:
                manifest = load_manifest_for_experiment(exp_id)
            if manifest is None:
                latest = _REPO / "outputs" / "compare" / "latest" / "manifest.json"
                if latest.is_file():
                    manifest = json.loads(latest.read_text(encoding="utf-8"))
            if manifest is None:
                return self._json_response(404, {"error": "manifest missing"})
            local_manifest = _HERE / "_manifest.json"
            local_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            handler_cls = type(self)
            handler_cls.manifest_path = local_manifest
            apply_manifest(manifest, handler_cls)
            return self._json_response(200, manifest)

        for segment in raw.split("/"):
            if segment in ("..", "~"):
                self.send_error(403)
                return

        data: bytes | None = None
        ctype = "application/octet-stream"

        if raw == "manifest.json":
            if self.manifest_path.is_file():
                data = self.manifest_path.read_bytes()
                ctype = "application/json"
        elif raw.startswith("data/"):
            data, ctype = self._read_side("b", raw[5:])
        elif raw.startswith("a/"):
            data, ctype = self._read_side("a", raw[2:])
        elif raw.startswith("b/"):
            data, ctype = self._read_side("b", raw[2:])
        else:
            static = (_STATIC / raw).resolve()
            try:
                static.relative_to(_STATIC.resolve())
            except ValueError:
                self.send_error(403)
                return
            if static.is_file():
                data = static.read_bytes()
                if raw.endswith(".html"):
                    ctype = "text/html"
                elif raw.endswith(".js"):
                    ctype = "application/javascript"
                elif raw.endswith(".css"):
                    ctype = "text/css"

        if data is None:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        raw = unquote(self.path.split("?", 1)[0]).lstrip("/")
        if raw != "api/rerun":
            self.send_error(404)
            return
        try:
            body = self._read_json_body()
            params = ExperimentParams.from_dict(body)
            result = start_experiment(params)
            code = 200 if result.get("ok") else 409
            self._json_response(code, result)
        except Exception as exc:
            self._json_response(500, {"ok": False, "error": str(exc)})


def apply_manifest(manifest: dict, handler_cls: type) -> None:
    baseline_dir = Path(manifest["baseline"]["viewer_dir"]).resolve()
    experiment_dir = Path(manifest["experiment"]["viewer_dir"]).resolve()
    handler_cls.baseline_dir = baseline_dir
    handler_cls.experiment_dir = experiment_dir
    handler_cls.baseline_run_dir = baseline_dir.parent.parent
    handler_cls.experiment_run_dir = experiment_dir.parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description="A/B compare viewer + experiment API")
    p.add_argument("--manifest", type=Path, help="manifest.json (default: outputs/compare/latest)")
    p.add_argument("--port", type=int, default=8095)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-open", action="store_true")
    args = p.parse_args()

    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = _REPO / "outputs" / "compare" / "latest" / "manifest.json"
        if not manifest_path.is_file():
            index = load_index()
            exps = index.get("experiments", [])
            if exps:
                manifest_path = _REPO / exps[0]["manifest"]
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    local_manifest = _HERE / "_manifest.json"
    local_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    handler = type("CompareHandler", (_Handler,), {})
    handler.manifest_path = local_manifest
    apply_manifest(manifest, handler)

    server = HTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"3D viewer → {url}")
    print(f"  run = {manifest['experiment']['run_id']}")
    print("  Ablation dropdown + re-run API enabled")

    if not args.no_open:
        threading.Thread(target=lambda: (time.sleep(0.8), webbrowser.open(url)), daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
