"""Phase 4b — batch VLM captioning via Gemini.

Default visual input is the **full perspective face** plus a TARGET BOUNDING BOX
(the format validated in Google AI Studio). Object crops remain a fallback.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import torch

from gemini_client import GeminiClient
from prompts import build_caption_user_prompt, format_bbox_tag, load_vocab_hint
from scene_io import (
    caption_summary,
    crop_path_for_object,
    ensure_caption_fields,
    face_view_for_object,
    infer_crops_dir,
    infer_faces_dir,
    is_active,
    object_count,
    write_caption,
)

log = logging.getLogger("phase4b.caption")

from scene_graph.captioning.structured import parse_structured_caption  # noqa: E402


@dataclass
class CaptionJobResult:
    object_index: int
    ok: bool
    decision: str = ""
    category: str = ""
    description: str = ""
    error: str = ""
    image_path: str = ""


def run_captioning(
    scene_state: dict,
    *,
    vocab_file: Optional[Path] = None,
    client: Optional[GeminiClient] = None,
    max_objects: int = 0,
    only_active: bool = True,
    skip_existing: bool = True,
    cache_dir: Optional[Path] = None,
    mock: bool = False,
    crops_dir: Optional[Path] = None,
    faces_dir: Optional[Path] = None,
    scene_state_path: Optional[Path] = None,
    fail_fast: bool = False,
    use_full_face: bool = True,
) -> List[CaptionJobResult]:
    ensure_caption_fields(scene_state)
    if crops_dir is not None:
        scene_state["_crops_dir"] = str(crops_dir)
    elif infer_crops_dir(scene_state, scene_state_path) is not None:
        scene_state["_crops_dir"] = str(infer_crops_dir(scene_state, scene_state_path))
    if faces_dir is not None:
        scene_state["_faces_dir"] = str(faces_dir)
    else:
        inferred_faces = infer_faces_dir(scene_state, scene_state_path)
        if inferred_faces is not None:
            scene_state["_faces_dir"] = str(inferred_faces)
            faces_dir = inferred_faces

    n = object_count(scene_state)
    vocab_hint = load_vocab_hint(vocab_file)
    gem = client or GeminiClient(cache_dir=cache_dir, mock=mock)
    fail_fast = fail_fast or os.environ.get("FAIL_FAST", "") in {"1", "true", "yes"}

    results: List[CaptionJobResult] = []
    t0 = time.time()
    processed = 0

    for idx in range(n):
        if only_active and not is_active(scene_state, idx):
            continue
        if max_objects > 0 and processed >= max_objects:
            break

        if skip_existing:
            existing = str(scene_state.get("object_caption_decision", [""] * n)[idx] or "")
            if existing in {"keep", "drop"}:
                continue

        image_path: Optional[Path] = None
        user_prompt = ""
        if use_full_face:
            view = face_view_for_object(
                scene_state,
                idx,
                faces_dir=faces_dir,
                scene_state_path=scene_state_path,
            )
            if view is not None:
                bbox_tag = format_bbox_tag(
                    view.bbox_xyxy,
                    image_width=view.image_width,
                    image_height=view.image_height,
                )
                user_prompt = build_caption_user_prompt(
                    vocab_hint=vocab_hint,
                    bbox_tag=bbox_tag,
                    image_width=view.image_width,
                    image_height=view.image_height,
                )
                image_path = view.rgb_path

        if image_path is None:
            crop = crop_path_for_object(scene_state, idx, crops_dir=crops_dir)
            if crop is None:
                results.append(CaptionJobResult(idx, False, error="no_face_or_crop"))
                continue
            user_prompt = build_caption_user_prompt(vocab_hint=vocab_hint, bbox_tag=None)
            image_path = crop

        try:
            raw = gem.caption_image(image_path=image_path, user_prompt=user_prompt)
            parsed = parse_structured_caption(raw)
            write_caption(scene_state, idx, parsed)
            results.append(
                CaptionJobResult(
                    idx,
                    True,
                    decision=parsed.decision or "",
                    category=parsed.category or "",
                    description=parsed.description or "",
                    image_path=str(image_path),
                )
            )
            processed += 1
            if processed % 25 == 0:
                log.info("Captioned %d objects (%.1fs)", processed, time.time() - t0)
        except Exception as exc:
            log.warning("Object %d caption failed: %s", idx, exc)
            results.append(CaptionJobResult(idx, False, error=str(exc), image_path=str(image_path)))
            if fail_fast:
                raise

    log.info(
        "Caption batch done: ok=%d fail=%d elapsed=%.1fs summary=%s",
        sum(1 for r in results if r.ok),
        sum(1 for r in results if not r.ok),
        time.time() - t0,
        caption_summary(scene_state),
    )
    return results


def save_caption_manifest(path: Path, results: List[CaptionJobResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(r) for r in results], indent=2),
        encoding="utf-8",
    )


def load_scene(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def save_scene(path: Path, scene_state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(scene_state, path)
