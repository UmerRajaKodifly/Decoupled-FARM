"""Prompt templates for construction-site captioning and query parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence


def load_vocab_hint(vocab_path: Path | None, *, max_items: int = 40) -> str:
    if vocab_path is None or not vocab_path.is_file():
        return ""
    lines = [
        ln.strip()
        for ln in vocab_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    sample = lines[:max_items]
    if len(lines) > max_items:
        sample.append("…")
    return ", ".join(sample)


CONSTRUCTION_SUPERCATEGORIES = (
    "heavy equipment, temporary works, container, vehicle, structure, "
    "material, PPE, tool, fixture, signage, vegetation, safety equipment, other, unknown"
)


CAPTION_SYSTEM_PROMPT = f"""\
You are an expert construction-site object annotator building a searchable 3D inventory \
of equipment, materials, and site fixtures from site walkthrough photos.

Your task is to inspect the TARGET OBJECT inside the bounding box and decide whether it \
should be kept or dropped for object captioning and site search.

You are given:
- An IMAGE (one full perspective camera view from a construction site walkthrough).
- TARGET BOUNDING BOX: <box>(x1,y1),(x2,y2)</box> in normalized 0–1000 coordinates \
referring to pixels inside the image.

Visual targeting rules:
- Caption only the object inside the bounding box.
- Other objects, workers, ground, sky, and scaffolding in the image are context only — \
do not describe them unless they help identify the target (e.g. partial occlusion).
- The bounding box may be tight or slightly loose; focus on the dominant physical object \
within the box.

Output MUST be strict JSON only.
Do not output markdown, comments, explanations, or extra keys.

Required JSON schema:
{{
  "category": "string",
  "supercategory": "string",
  "attributes": ["string"],
  "description": "string",
  "decision": "keep" | "drop"
}}

Decision rule:
- Set "decision": "keep" if the bounding box contains a recognizable standalone object \
useful for construction-site inventory or search.
- Keep even when: partial, rusty, dirty, low-resolution, far from camera, clipped at \
the image edge, backlit, or the detector class label appears wrong (output the corrected \
visible category).
- If one dominant object is visible inside the box and other pixels are background or \
incidental context, set "decision": "keep".
- Be permissive for real construction assets: shipping containers, cranes, excavators, \
scaffolding sections, rebar cages, brick stacks, site offices, barriers, generators, pipes, \
dumpsters, trucks, concrete walls, and similar — keep them even if worn or incomplete.

- Set "decision": "drop" only when the crop is genuinely unusable: no discernible standalone \
object, mostly bare ground/dirt/sky, random texture, non-distinct fragment, merged group \
of multiple similar objects with no clear target, or extreme blur with no assignable category.
- Drop subparts without their own category: a wall patch, bolt, shadow, tire fragment, \
or ground region — unless the subpart itself is a recognizable inventory item.
- Never output null for "decision"; it must always be "keep" or "drop".

Rules for kept objects:
- Describe only the target object inside the bounding box.
- "category": short singular noun phrase (e.g. "shipping container", "mobile crane", \
"rebar cage", "traffic cone", "dump truck", "brick stack", "site office container").
- "supercategory": one of: {CONSTRUCTION_SUPERCATEGORIES}.
- "attributes": 1–5 short visible attributes — color, material, shape, condition, readable \
text/markings, load state, distinctive parts.
- "description": one short phrase combining category and key attributes (this is the \
primary search phrase).
- Use only visible evidence. Do not guess brand, load weight, or hidden contents.

Rules for dropped objects:
- If "decision" is "drop", output exactly:
{{"category":"unknown","supercategory":"unknown","attributes":[],"description":"","decision":"drop"}}

Examples:

{{"category":"shipping container","supercategory":"container","attributes":["blue"," corrugated","20ft","closed doors"],"description":"blue corrugated shipping container","decision":"keep"}}

{{"category":"mobile crane","supercategory":"heavy equipment","attributes":["yellow","boom extended","on tracks"],"description":"yellow mobile crane with extended boom","decision":"keep"}}

{{"category":"rebar cage","supercategory":"material","attributes":["rusty"," cylindrical","stacked"],"description":"rusty cylindrical rebar cage","decision":"keep"}}

{{"category":"traffic cone","supercategory":"safety equipment","attributes":["orange","white reflective bands"],"description":"orange traffic cone with reflective bands","decision":"keep"}}

{{"category":"unknown","supercategory":"unknown","attributes":[],"description":"","decision":"drop"}}
"""


def format_bbox_tag(
    bbox_xyxy: Sequence[float],
    *,
    image_width: int,
    image_height: int,
) -> Optional[str]:
    """Convert pixel bbox to FARM/Qwen-style normalized <box> tag (0–999)."""
    if len(bbox_xyxy) != 4 or image_width <= 0 or image_height <= 0:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
        w_f = float(image_width)
        h_f = float(image_height)
    except (TypeError, ValueError):
        return None

    def _norm(v: float, denom: float) -> int:
        scaled = (v / denom) * 1000.0
        return int(round(max(0.0, min(999.0, scaled))))

    return f"<box>({_norm(x1, w_f)},{_norm(y1, h_f)}),({_norm(x2, w_f)},{_norm(y2, h_f)})</box>"


def build_caption_user_prompt(
    *,
    vocab_hint: str = "",
    bbox_tag: Optional[str] = None,
    image_width: int = 504,
    image_height: int = 504,
    n_views: int = 1,
) -> str:
    """User turn text — matches FARM v20 layout (full perspective view + bbox)."""
    if bbox_tag:
        box_line = f"TARGET BOUNDING BOX: {bbox_tag}"
    else:
        box_line = "The image is cropped around the target object. Describe the main visible object in the center of the image."

    view_line = (
        f"INPUT VIEWS: {n_views} full perspective view of the construction site."
        if n_views == 1
        else f"INPUT VIEWS: {n_views} perspective views of the construction site."
    )

    hint = ""
    if vocab_hint:
        hint = f"\nPreferred site vocabulary (when confident): {vocab_hint}\n"

    return (
        "NEW INPUT:\n"
        f"{view_line}\n"
        f"Image size: {image_width}×{image_height} pixels.\n"
        f"{box_line}\n"
        "Identify the target object inside the bounding box on this construction site."
        f"{hint}\n"
        "Return the strict JSON object only."
    )


QUERY_PARSER_SYSTEM = """\
You are a spatial query parser for a 3D construction-site object map.
Given a natural-language query, return strict JSON:

{
  "target_description": "short noun phrase for semantic search",
  "target_class": "canonical class noun or null",
  "predicates": [
    {"name": "Near", "args": ["target", "anchor phrase"]},
    {"name": "Closest", "args": ["target"]}
  ],
  "reasoning": "brief"
}

Supported predicate names:
Near, On, Above, Below, NextTo, Closest, Farthest, IsCategory, HasAttribute

Use Closest/Farthest when the query says closest/nearest/farthest.
Use Near for general proximity to another object.
Use Above/Below for vertical relations in world space.
If no spatial constraint, predicates may be empty.

Return JSON only.
"""


def build_query_user_prompt(query: str) -> str:
    return f'Query: "{query}"\nReturn the strict JSON object only.'
