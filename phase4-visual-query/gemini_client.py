"""Gemini API client with disk cache and mock mode for offline dev."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from prompts import CAPTION_SYSTEM_PROMPT, QUERY_PARSER_SYSTEM


class GeminiClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        caption_model: str = "gemini-3.0-flash",
        text_model: str = "gemini-3.0-flash",
        embed_model: str = "text-embedding-004",
        cache_dir: Optional[Path] = None,
        mock: bool = False,
        rpm_delay_s: float = 0.35,
    ) -> None:
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.caption_model = caption_model
        self.text_model = text_model
        self.embed_model = embed_model
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.mock = mock or not self.api_key
        self.rpm_delay_s = rpm_delay_s
        self._client = None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if not self.mock:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)

    def _cache_path(self, kind: str, key: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return self.cache_dir / f"{kind}_{h}.json"

    def _read_cache(self, path: Path) -> Optional[Any]:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _write_cache(self, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def caption_image(self, *, image_path: Path, user_prompt: str) -> str:
        cache_key = f"cap|{self.caption_model}|{image_path}|{user_prompt}|{image_path.stat().st_mtime_ns if image_path.is_file() else 0}"
        cp = self._cache_path("caption", cache_key)
        if cp:
            hit = self._read_cache(cp)
            if hit is not None:
                return str(hit.get("text", ""))

        if self.mock:
            stem = image_path.stem.replace("_", " ")
            text = json.dumps(
                {
                    "category": "construction object",
                    "supercategory": "other",
                    "attributes": ["mock"],
                    "description": f"mock caption for {stem}",
                    "decision": "keep",
                }
            )
        else:
            from google.genai import types

            img_bytes = image_path.read_bytes()
            parts = [
                types.Part.from_text(text=user_prompt),
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            ]
            resp = self._client.models.generate_content(
                model=self.caption_model,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    system_instruction=CAPTION_SYSTEM_PROMPT,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            text = (resp.text or "").strip()
            time.sleep(self.rpm_delay_s)

        if cp:
            self._write_cache(cp, {"text": text})
        return text

    def parse_json_text(self, *, system: str, user: str, model: Optional[str] = None) -> str:
        use_model = model or self.text_model
        cache_key = f"txt|{use_model}|{system[:200]}|{user}"
        cp = self._cache_path("text", cache_key)
        if cp:
            hit = self._read_cache(cp)
            if hit is not None:
                return str(hit.get("text", ""))

        if self.mock:
            if "Query:" in user:
                q = user.split('Query: "')[1].split('"')[0] if 'Query: "' in user else "object"
                text = json.dumps(
                    {
                        "target_description": q,
                        "target_class": q.split()[-1] if q.split() else None,
                        "predicates": [],
                        "reasoning": "mock",
                    }
                )
            else:
                text = "{}"
        else:
            from google.genai import types

            resp = self._client.models.generate_content(
                model=use_model,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=user)])],
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            text = (resp.text or "").strip()
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
                    out.append(hit["vector"])
                    continue
            pending.append((i, t))
            out.append([])  # placeholder

        if self.mock:
            for i, t in pending:
                vec = _mock_embed(t)
                out[i] = vec
                cp = self._cache_path("embed", f"emb|{self.embed_model}|{t}")
                if cp:
                    self._write_cache(cp, {"vector": vec})
            return out

        from google.genai import types

        for i, t in pending:
            resp = self._client.models.embed_content(
                model=self.embed_model,
                contents=t,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            vals = resp.embeddings[0].values if resp.embeddings else []
            vec = [float(x) for x in vals]
            out[i] = vec
            cp = self._cache_path("embed", f"emb|{self.embed_model}|{t}")
            if cp:
                self._write_cache(cp, {"vector": vec})
            time.sleep(self.rpm_delay_s * 0.5)
        return out


def _mock_embed(text: str, dim: int = 64) -> list[float]:
    """Deterministic pseudo-embedding for offline smoke tests."""
    import math

    h = hashlib.sha256(text.encode("utf-8")).digest()
    vec = []
    for j in range(dim):
        b = h[j % len(h)]
        vec.append(math.sin(b + j * 0.17))
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]
