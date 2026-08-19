#!/usr/bin/env python3
"""Embed one caption string. Prints dimension and L2-norm."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def _load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    path = ROOT / ".env"
    if not path.is_file():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def main() -> int:
    env = _load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=env.get("VLLM_HOST", "100.109.254.4"))
    parser.add_argument("--port", type=int, default=int(env.get("VLLM_EMBED_PORT", "8102")))
    parser.add_argument("--model", default=env.get("VLLM_EMBED_SERVED_NAME", "qwen3-emb-0.6b"))
    parser.add_argument("--api-key", default=env.get("VLLM_API_KEY") or os.environ.get("VLLM_API_KEY", ""))
    parser.add_argument("--text", default="blue corrugated shipping container")
    args = parser.parse_args()
    if not args.api_key:
        print("VLLM_API_KEY missing", file=sys.stderr)
        return 2

    payload = {"model": args.model, "input": [args.text]}
    url = f"http://{args.host}:{args.port}/v1/embeddings"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code} from {url}\n{err}", file=sys.stderr)
        return 1

    vec = body["data"][0]["embedding"]
    dim = len(vec)
    norm = math.sqrt(sum(x * x for x in vec))
    print(f"dim={dim} l2={norm:.6f} model={body.get('model')}")
    if dim < 128:
        print("embedding dim < 128 (Pakistan client expects >= 128)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
