#!/usr/bin/env python3
"""POST one 504x504 JPEG + center bbox to the caption VLM. Prints parsed JSON."""
from __future__ import annotations

import argparse
import base64
import io
import json
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


def _vocab_hint(max_items: int = 40) -> str:
    lines = [
        ln.strip()
        for ln in (ROOT / "data" / "construction_vocab.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    sample = lines[:max_items]
    if len(lines) > max_items:
        sample.append("…")
    return ", ".join(sample)


def _dummy_jpeg(width: int = 504, height: int = 504) -> bytes:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise SystemExit("Pillow required: pip install pillow") from exc
    img = Image.new("RGB", (width, height), (32, 96, 160))
    draw = ImageDraw.Draw(img)
    draw.rectangle([80, 80, 424, 424], fill=(180, 40, 40), outline=(240, 240, 240), width=6)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def main() -> int:
    env = _load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=env.get("VLLM_HOST", "100.109.254.4"))
    parser.add_argument("--port", type=int, default=int(env.get("VLLM_VLM_PORT", "8100")))
    parser.add_argument("--model", default=env.get("VLLM_VLM_SERVED_NAME", "qwen3-vl-8b"))
    parser.add_argument("--api-key", default=env.get("VLLM_API_KEY") or os.environ.get("VLLM_API_KEY", ""))
    args = parser.parse_args()
    if not args.api_key:
        print("VLLM_API_KEY missing", file=sys.stderr)
        return 2

    system = (ROOT / "prompts" / "caption_system.txt").read_text(encoding="utf-8").strip()
    schema = json.loads((ROOT / "prompts" / "caption_schema.json").read_text(encoding="utf-8"))
    template = (ROOT / "prompts" / "caption_user_template.txt").read_text(encoding="utf-8")
    user_text = (
        template.replace("{width}", "504")
        .replace("{height}", "504")
        .replace("{bbox_tag}", "<box>(160,160),(840,840)</box>")
        .replace("{vocab_hint}", _vocab_hint())
    )
    jpeg_b64 = base64.b64encode(_dummy_jpeg()).decode("ascii")
    payload = {
        "model": args.model,
        "temperature": 0.0,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"},
                    },
                ],
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "construction_caption", "schema": schema, "strict": True},
        },
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
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
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code} from {url}\n{err}", file=sys.stderr)
        return 1

    text = body["choices"][0]["message"]["content"]
    print(text)
    parsed = json.loads(text)
    decision = parsed.get("decision")
    if decision not in {"keep", "drop"}:
        print(f"unexpected decision={decision!r}", file=sys.stderr)
        return 1
    print(f"ok decision={decision} category={parsed.get('category')!r}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
