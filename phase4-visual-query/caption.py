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
    caption_checkpoint_is_mock,
    caption_summary,
    crop_path_for_object,
    ensure_caption_fields,
    face_view_for_object,
    infer_crops_dir,
    infer_faces_dir,
    is_active,
    object_count,
    overlay_captions,
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
    crops_dir: Optional[Path] = None,
    faces_dir: Optional[Path] = None,
    scene_state_path: Optional[Path] = None,
    fail_fast: bool = False,
    use_full_face: bool = True,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 25,
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
    gem = client or GeminiClient(cache_dir=cache_dir)
    fail_fast = fail_fast or os.environ.get("FAIL_FAST", "") in {"1", "true", "yes"}
    if checkpoint_every <= 0:
        checkpoint_every = 25

    if checkpoint_path is not None and Path(checkpoint_path).is_file() and skip_existing:
        prev = load_scene(Path(checkpoint_path))
        if caption_checkpoint_is_mock(prev):
            raise RuntimeError(
                f"Refusing to resume leftover mock checkpoint {checkpoint_path}. "
                "Delete scene_state_captioned.pt before a Gemini run."
            )
        n_overlaid = overlay_captions(scene_state, prev)
        log.info("Resumed %d captions from %s", n_overlaid, checkpoint_path)

    results: List[CaptionJobResult] = []
    t0 = time.time()
    processed = 0

    def _checkpoint(reason: str) -> None:
        if checkpoint_path is None:
            return
        save_scene(Path(checkpoint_path), scene_state)
        log.info("Checkpoint (%s) %s summary=%s", reason, checkpoint_path, caption_summary(scene_state))

    try:
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
                    if bbox_tag is None:
                        results.append(
                            CaptionJobResult(idx, False, error="face_missing_bbox", image_path=str(view.rgb_path))
                        )
                        continue
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
                if parsed.decision not in {"keep", "drop"}:
                    raise ValueError("caption JSON missing keep/drop (left uncaptioned for retry)")
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
                if checkpoint_every > 0 and processed % checkpoint_every == 0:
                    _checkpoint(f"every {checkpoint_every}")
            except Exception as exc:
                log.warning("Object %d caption failed: %s", idx, exc)
                results.append(CaptionJobResult(idx, False, error=str(exc), image_path=str(image_path)))
                if fail_fast:
                    raise
    finally:
        if processed > 0:
            _checkpoint("exit")

    log.info(
        "Caption batch done: ok=%d fail=%d elapsed=%.1fs api_calls=%s cache_hits=%s summary=%s",
        sum(1 for r in results if r.ok),
        sum(1 for r in results if not r.ok),
        time.time() - t0,
        getattr(gem, "n_api_calls", "?"),
        getattr(gem, "n_cache_hits", "?"),
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
    tmp = path.with_name(path.name + ".tmp")
    torch.save(scene_state, tmp)
    tmp.replace(path)
