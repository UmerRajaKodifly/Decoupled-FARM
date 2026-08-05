"""Adapter: spatialGPT construction-site JSON → FARM YOLOE vocab lines.

FARM's loader (`yoloe._load_vocab_list`) expects a plain text file: one class
name per line, no JSON, no descriptions. spatialGPT stores structured objects
with `label`, `description`, `counting_mode`, etc.

This adapter does **not** edit the upstream JSON. It writes a derived `.txt`
beside the pipeline outputs (or a cache path) containing prompt strings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_SPATIALGPT_VOCAB = Path(
    "/home/kodifly/Desktop/farm-rnd/ss-spatial-gpt/spatial-app/"
    "construction_site_object_vocabulary.json"
)


@dataclass(frozen=True)
class VocabAdapterReport:
    source: str
    n_objects: int
    n_aliases_ignored: int
    prompt_names: list[str]
    unmapped_top_level_keys: list[str]
    notes: list[str]


def load_construction_vocab(path: str | Path | None = None) -> tuple[list[str], VocabAdapterReport]:
    path = Path(path or DEFAULT_SPATIALGPT_VOCAB)
    payload = json.loads(path.read_text(encoding="utf-8"))
    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f"{path} has no 'objects' list; cannot adapt to FARM vocab.txt")

    prompts: list[str] = []
    seen: set[str] = set()
    for entry in objects:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label") or "").strip()
        if not label:
            continue
        # FARM / YOLOE prompts are natural phrases ("steel beam"), not snake_case ids.
        retrieval = entry.get("retrieval_query")
        if isinstance(retrieval, str) and retrieval.strip():
            prompt = retrieval.strip()
        else:
            prompt = label.replace("_", " ").strip()
        key = prompt.lower()
        if key in seen:
            continue
        seen.add(key)
        prompts.append(prompt)

    aliases = payload.get("label_aliases") if isinstance(payload.get("label_aliases"), dict) else {}
    known_top = {"site", "source", "purpose", "usage_notes", "objects", "label_aliases", "stats"}
    extra_keys = sorted(set(payload.keys()) - known_top)

    report = VocabAdapterReport(
        source=str(path),
        n_objects=len(prompts),
        n_aliases_ignored=len(aliases),
        prompt_names=list(prompts),
        unmapped_top_level_keys=extra_keys,
        notes=[
            "FARM vocab is one prompt string per line; descriptions / counting_mode / "
            "retrieval_hard are not consumed by YOLOE.",
            "label_aliases are VLM synonym maps for Spatial GPT, not YOLOE class ids — left unused.",
            "Prompt text prefers retrieval_query when present, else label with underscores→spaces.",
        ],
    )
    return prompts, report


def write_farm_vocab_txt(
    dest: str | Path,
    *,
    json_path: str | Path | None = None,
) -> tuple[Path, VocabAdapterReport]:
    prompts, report = load_construction_vocab(json_path)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    return dest, report
