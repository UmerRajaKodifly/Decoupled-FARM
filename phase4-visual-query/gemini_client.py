"""Gemini API client with disk cache. Live calls only — requires an API key."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from prompts import CAPTION_SYSTEM_PROMPT

log = logging.getLogger("phase4.gemini")

DEFAULT_CAPTION_MODEL = "gemini-3-flash-preview"
DEFAULT_EMBED_MODEL = "text-embedding-004"
DEFAULT_THINKING_LEVEL = "minimal"

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


def _is_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in {429, 500, 502, 503, 504}:
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
            log.warning("Gemini retry %d/%d in %.1fs: %s", i + 1, attempts, sleep, exc)
            time.sleep(sleep)
    assert last is not None
    raise last


class GeminiClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        caption_model: str = DEFAULT_CAPTION_MODEL,
        text_model: str = DEFAULT_CAPTION_MODEL,
        embed_model: str = DEFAULT_EMBED_MODEL,
        cache_dir: Optional[Path] = None,
        rpm_delay_s: float = 0.35,
        thinking_level: str = DEFAULT_THINKING_LEVEL,
    ) -> None:
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.caption_model = caption_model
        self.text_model = text_model
        self.embed_model = embed_model
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.rpm_delay_s = rpm_delay_s
        self.thinking_level = thinking_level
        self._client = None
        self.n_api_calls = 0
        self.n_cache_hits = 0
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY or GEMINI_API_KEY is required. "
                "Export one before running Track B."
            )
        log.info(
            "GeminiClient model=%s embed=%s thinking=%s cache=%s",
            self.caption_model,
            self.embed_model,
            self.thinking_level,
            self.cache_dir,
        )

    def _live_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

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

    def _log_usage(self, resp: Any) -> None:
        um = getattr(resp, "usage_metadata", None)
        if um is None:
            return
        prompt = getattr(um, "prompt_token_count", None)
        thoughts = getattr(um, "thoughts_token_count", None)
        candidates = getattr(um, "candidates_token_count", None)
        total = getattr(um, "total_token_count", None)
        log.info(
            "Gemini usage prompt=%s thoughts=%s candidates=%s total=%s",
            prompt,
            thoughts,
            candidates,
            total,
        )

    def _generate_config(self, *, system: str):
        from google.genai import types

        kwargs: dict[str, Any] = {
            "system_instruction": system,
            "temperature": 0.0,
            "response_mime_type": "application/json",
        }
        thinking_cls = getattr(types, "ThinkingConfig", None)
        if thinking_cls is not None and self.thinking_level:
            try:
                kwargs["thinking_config"] = thinking_cls(thinking_level=self.thinking_level)
            except TypeError:
                log.warning("ThinkingConfig(thinking_level=...) unsupported; omitting thinking cap")
        return types.GenerateContentConfig(**kwargs)

    def caption_image(self, *, image_path: Path, user_prompt: str) -> str:
        cache_key = (
            f"cap|{self.caption_model}|{self.thinking_level}|{image_path}|{user_prompt}|"
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
                if "mock caption" in text.lower():
                    log.warning("Ignoring leftover mock caption cache at %s", cp)

        from google.genai import types

        mime = _MIME_BY_SUFFIX.get(image_path.suffix.lower(), "image/jpeg")
        img_bytes = image_path.read_bytes()
        parts = [
            types.Part.from_text(text=user_prompt),
            types.Part.from_bytes(data=img_bytes, mime_type=mime),
        ]

        def _do():
            return self._live_client().models.generate_content(
                model=self.caption_model,
                contents=[types.Content(role="user", parts=parts)],
                config=self._generate_config(system=CAPTION_SYSTEM_PROMPT),
            )

        resp = _call_with_retry(_do)
        self.n_api_calls += 1
        self._log_usage(resp)
        text = (resp.text or "").strip()
        if not text:
            raise RuntimeError(f"Empty Gemini caption for {image_path}")
        time.sleep(self.rpm_delay_s)
        if cp:
            self._write_cache(cp, {"text": text})
        return text

    def parse_json_text(self, *, system: str, user: str, model: Optional[str] = None) -> str:
        use_model = model or self.text_model
        cache_key = f"txt|{use_model}|{self.thinking_level}|{system[:200]}|{user}"
        cp = self._cache_path("text", cache_key)
        if cp:
            hit = self._read_cache(cp)
            if hit is not None:
                text = str(hit.get("text", "")).strip()
                if text:
                    self.n_cache_hits += 1
                    return text

        from google.genai import types

        def _do():
            return self._live_client().models.generate_content(
                model=use_model,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=user)])],
                config=self._generate_config(system=system),
            )

        resp = _call_with_retry(_do)
        self.n_api_calls += 1
        self._log_usage(resp)
        text = (resp.text or "").strip()
        if not text:
            raise RuntimeError("Empty Gemini JSON response")
        time.sleep(self.rpm_delay_s)
        if cp:
            self._write_cache(cp, {"text": text})
        return text

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        pending: list[tuple[int, str]] = []

        for i, t in enumerate(texts):
            cache_key = f"emb|{self.embed_model}|{t}"
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
            pending.append((i, t))
            out.append([])

        from google.genai import types

        for i, t in pending:
            def _do(text: str = t):
                return self._live_client().models.embed_content(
                    model=self.embed_model,
                    contents=text,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                )

            resp = _call_with_retry(_do)
            self.n_api_calls += 1
            vals = resp.embeddings[0].values if resp.embeddings else []
            vec = [float(x) for x in vals]
            if len(vec) < 128:
                raise RuntimeError(f"Embedding too short ({len(vec)}-d) for {t[:80]!r}")
            out[i] = vec
            cp = self._cache_path("embed", f"emb|{self.embed_model}|{t}")
            if cp:
                self._write_cache(cp, {"vector": vec})
            time.sleep(self.rpm_delay_s * 0.5)
        return out
