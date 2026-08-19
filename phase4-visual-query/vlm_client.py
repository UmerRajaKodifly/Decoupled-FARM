"""Remote VLM client — OpenAI-compatible vLLM (HK caption + embed + query parse)."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypeVar

import httpx

from prompts import CAPTION_SYSTEM_PROMPT

_REPO = Path(__file__).resolve().parent.parent
_FARM_SRC = _REPO / "farm_src" / "src"
_STRUCTURED = _FARM_SRC / "scene_graph" / "captioning" / "structured.py"
_spec = importlib.util.spec_from_file_location("phase4_structured_caption", _STRUCTURED)
assert _spec and _spec.loader
_structured = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _structured
_spec.loader.exec_module(_structured)
CAPTION_SCHEMA = _structured.CAPTION_SCHEMA

log = logging.getLogger("phase4.vlm")

DEFAULT_VL_MODEL = "qwen3-vl-8b"
DEFAULT_EMBED_MODEL = "qwen3-emb-0.6b"
DEFAULT_BASE_URL = "http://100.109.254.4:8100/v1"
DEFAULT_EMBED_BASE_URL = "http://100.109.254.4:8102/v1"

QWEN3_EMBED_TASK = os.getenv("QWEN3_EMBED_TASK") or (
    "Embed for object identity and affordance retrieval. "
    "Use the object's category, supercategory, and visible attributes; ignore viewpoint and wording."
)

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

_RETRYABLE = (
    "429",
    "503",
    "unavailable",
    "resource exhausted",
    "resource_exhausted",
    "timeout",
    "temporarily",
    "deadline exceeded",
    "connection reset",
    "internal error",
)

T = TypeVar("T")
EmbedMode = Literal["document", "query", "raw"]


def _load_dotenv() -> None:
    for env_file in (_REPO / ".env", _REPO.parent / "repo" / ".env"):
        if not env_file.is_file():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val
        break


_load_dotenv()


def _is_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in {429, 500, 502, 503, 504}:
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in {429, 500, 502, 503, 504}:
        return True
    return any(tok in msg for tok in _RETRYABLE)


def _call_with_retry(fn: Callable[[], T], *, attempts: int = 5, delay_s: float = 1.0) -> T:
    last: Optional[BaseException] = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if i == attempts - 1 or not _is_retryable(exc):
                raise
            sleep = delay_s * (2 ** i)
            log.warning("VLM retry %d/%d in %.1fs: %s", i + 1, attempts, sleep, exc)
            time.sleep(sleep)
    assert last is not None
    raise last


def _chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _embed_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/embeddings"
    return f"{base}/v1/embeddings"


def _wrap_embed_text(text: str, *, mode: EmbedMode) -> str:
    if mode == "raw":
        return (text or "").strip()
    canon = (text or "").strip().lower()
    if not canon:
        return ""
    if mode == "query":
        return f"Instruct: {QWEN3_EMBED_TASK}\nQuery: {canon}"
    return f"Instruct: {QWEN3_EMBED_TASK}\nDocument: {canon}"


class VlmClient:
    """OpenAI-compatible client for HK vLLM caption VLM + text embedder."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        embed_base_url: Optional[str] = None,
        vl_model: Optional[str] = None,
        text_model: Optional[str] = None,
        embed_model: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        rpm_delay_s: float = 0.05,
        timeout_s: Optional[float] = None,
        embed_timeout_s: Optional[float] = None,
        disable_thinking: Optional[bool] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("VLLM_API_KEY") or ""
        self.base_url = (base_url or os.environ.get("VLLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.embed_base_url = (
            embed_base_url or os.environ.get("VLLM_EMBED_BASE_URL") or DEFAULT_EMBED_BASE_URL
        ).rstrip("/")
        self.vl_model = vl_model or os.environ.get("VLLM_VL_MODEL") or DEFAULT_VL_MODEL
        self.text_model = text_model or self.vl_model
        self.embed_model = embed_model or os.environ.get("VLLM_EMBED_MODEL") or DEFAULT_EMBED_MODEL
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.rpm_delay_s = rpm_delay_s
        self.timeout_s = float(timeout_s or os.environ.get("VLLM_TIMEOUT_S", "120"))
        self.embed_timeout_s = float(embed_timeout_s or os.environ.get("VLLM_EMBED_TIMEOUT_S", "60"))
        if disable_thinking is None:
            disable_thinking = os.environ.get("VLLM_DISABLE_THINKING", "1") == "1"
        self.disable_thinking = disable_thinking
        self.n_api_calls = 0
        self.n_cache_hits = 0
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.api_key:
            raise RuntimeError(
                "VLLM_API_KEY is required. Set it in Decoupled-FARM/.env or export before running Track B."
            )
        log.info(
            "VlmClient vl=%s embed=%s base=%s embed_base=%s cache=%s",
            self.vl_model,
            self.embed_model,
            self.base_url,
            self.embed_base_url,
            self.cache_dir,
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _cache_path(self, kind: str, key: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{kind}_{h}.json"

    def _read_cache(self, path: Path) -> Optional[Any]:
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Ignoring corrupt cache file %s: %s", path, exc)
            return None

    def _write_cache(self, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _post_json(self, url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        def _do() -> dict[str, Any]:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, headers=self._headers(), json=payload)
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, dict):
                    raise RuntimeError(f"Unexpected JSON response from {url}")
                return data

        data = _call_with_retry(_do)
        self.n_api_calls += 1
        usage = data.get("usage")
        if usage:
            log.debug("VLM usage %s", usage)
        return data

    def _extract_chat_text(self, data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
            return "".join(parts).strip()
        return str(content or "").strip()

    def _caption_response_format(self) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "construction_caption",
                "strict": True,
                "schema": CAPTION_SCHEMA,
            },
        }

    def caption_image(self, *, image_path: Path, user_prompt: str) -> str:
        cache_key = (
            f"cap|{self.vl_model}|{self.disable_thinking}|{image_path}|{user_prompt}|"
            f"{image_path.stat().st_mtime_ns if image_path.is_file() else 0}"
        )
        cp = self._cache_path("caption", cache_key)
        if cp:
            hit = self._read_cache(cp)
            if hit is not None:
                text = str(hit.get("text", "")).strip()
                if text and "mock caption" not in text.lower():
                    self.n_cache_hits += 1
                    return text

        mime = _MIME_BY_SUFFIX.get(image_path.suffix.lower(), "image/jpeg")
        b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload: dict[str, Any] = {
            "model": self.vl_model,
            "temperature": 0.0,
            "max_tokens": 512,
            "messages": [
                {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                },
            ],
            "response_format": self._caption_response_format(),
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        data = self._post_json(_chat_url(self.base_url), payload, timeout=self.timeout_s)
        text = self._extract_chat_text(data)
        if not text:
            raise RuntimeError(f"Empty VLM caption for {image_path}")
        time.sleep(self.rpm_delay_s)
        if cp:
            self._write_cache(cp, {"text": text})
        return text

    def parse_json_text(self, *, system: str, user: str, model: Optional[str] = None) -> str:
        use_model = model or self.text_model
        cache_key = f"txt|{use_model}|{self.disable_thinking}|{system[:200]}|{user}"
        cp = self._cache_path("text", cache_key)
        if cp:
            hit = self._read_cache(cp)
            if hit is not None:
                text = str(hit.get("text", "")).strip()
                if text:
                    self.n_cache_hits += 1
                    return text

        payload: dict[str, Any] = {
            "model": use_model,
            "temperature": 0.0,
            "max_tokens": 512,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        data = self._post_json(_chat_url(self.base_url), payload, timeout=self.timeout_s)
        text = self._extract_chat_text(data)
        if not text:
            raise RuntimeError("Empty VLM JSON response")
        time.sleep(self.rpm_delay_s)
        if cp:
            self._write_cache(cp, {"text": text})
        return text

    def embed_texts(
        self,
        texts: list[str],
        *,
        mode: EmbedMode = "document",
    ) -> list[list[float]]:
        if not texts:
            return []

        out: list[list[float]] = []
        pending: list[tuple[int, str, str]] = []

        for i, raw in enumerate(texts):
            wrapped = _wrap_embed_text(raw, mode=mode)
            cache_key = f"emb|{self.embed_model}|{mode}|{wrapped}"
            cp = self._cache_path("embed", cache_key)
            if cp:
                hit = self._read_cache(cp)
                if hit is not None and "vector" in hit:
                    vec = hit["vector"]
                    if isinstance(vec, list) and len(vec) == 64:
                        log.warning("Ignoring leftover 64-d mock embedding cache at %s", cp)
                    elif isinstance(vec, list) and len(vec) >= 128:
                        self.n_cache_hits += 1
                        out.append(vec)
                        continue
            pending.append((i, raw, wrapped))
            out.append([])

        if not pending:
            return out

        wrapped_batch = [w for _, _, w in pending if w]
        if not wrapped_batch:
            return out

        payload = {
            "model": self.embed_model,
            "input": wrapped_batch,
        }
        data = self._post_json(_embed_url(self.embed_base_url), payload, timeout=self.embed_timeout_s)

        items = data.get("data") or []
        by_index: dict[int, list[float]] = {}
        for fallback_idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            idx_raw = item.get("index", fallback_idx)
            try:
                idx = int(idx_raw)
            except (TypeError, ValueError):
                idx = fallback_idx
            emb = item.get("embedding") or []
            by_index[idx] = [float(x) for x in emb]

        batch_i = 0
        for out_i, raw, wrapped in pending:
            if not wrapped:
                continue
            vec = by_index.get(batch_i, [])
            batch_i += 1
            if len(vec) < 128:
                raise RuntimeError(f"Embedding too short ({len(vec)}-d) for {raw[:80]!r}")
            out[out_i] = vec
            cp = self._cache_path("embed", f"emb|{self.embed_model}|{mode}|{wrapped}")
            if cp:
                self._write_cache(cp, {"vector": vec})
            time.sleep(self.rpm_delay_s * 0.25)

        return out
