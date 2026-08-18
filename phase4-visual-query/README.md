# Phase 4 — Visual Query (Track B)

Construction-site captioning, embeddings, post-caption identity merge, and natural-language retrieval.

Captioning sends **one full perspective face + TARGET BOUNDING BOX** per object (the format validated in Google AI Studio). Local VLM serving is a separate refactor; this pipeline is Gemini-compatible (`gemini-3.0-flash` by default) and works offline with `MOCK=1`.

## Pipeline

```
Phase 4a (best-view) →  scene_state_with_crops.pt  (bbox + face path on each object)
Phase 4b (caption)   →  scene_state_captioned.pt
Phase 4c (embed)     →  scene_state_enriched.pt
Phase 4d (merge)     →  scene_state_merged.pt
Query index          →  query_index.json
Viewer               →  validation/3d-viewer-trackb/ + POST /api/query
```

## Quick start

```bash
conda activate farm-phase2
cd farm-object-map

# After Phase 4a exists on a run:
export GOOGLE_API_KEY=your_key
bash scripts/run_track_b.sh

# Offline smoke test (mock captions + pseudo-embeddings):
MOCK=1 MAX_OBJECTS=20 bash scripts/run_track_b.sh
```

## Manual steps

```bash
pip install -r phase4-visual-query/requirements.txt

python phase4-visual-query/run_phase4b_caption.py \
  --scene-state outputs/latest/phase4/scene_state_with_crops.pt \
  --faces-dir outputs/latest/phase1.5/faces

python phase4-visual-query/run_phase4c_embed.py \
  --scene-state outputs/latest/phase4/scene_state_captioned.pt

python phase4-visual-query/run_phase4d_merge.py \
  --scene-state outputs/latest/phase4/scene_state_enriched.pt

python phase4-visual-query/run_build_query_index.py \
  --scene-state outputs/latest/phase4/scene_state_merged.pt \
  --output outputs/latest/phase4/query_index.json

python phase4-visual-query/run_query.py "mobile crane near containers"

python 3d-viewer/build_viewer_data.py \
  --stella-state outputs/latest/phase4/scene_state_merged.pt \
  --output-dir outputs/latest/validation/3d-viewer-trackb
python 3d-viewer/serve.py --data-dir outputs/latest/validation/3d-viewer-trackb
```

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | — | Gemini API access |
| `MOCK=1` | off | Offline mock captions/embeddings |
| `MAX_OBJECTS` | 0 (all) | Limit caption count for dev |
| `CAPTION_MODEL` | `gemini-3.0-flash` | VLM for structured JSON |
| `EMBED_MODEL` | `text-embedding-004` | Caption text vectors |
| `FAIL_FAST=1` | off | Stop the caption batch on the first API error |
| `CAPTION_MERGE_HELLINGER_THRESH` | `0.65` | Spatial neighbour gate (Phase 4d) |
| `CAPTION_MERGE_CAPTION_THRESH` | `0.92` | Caption cosine gate |
| `CAPTION_MERGE_VISUAL_THRESH` | `0.90` | DINO feature cosine gate |
| `CAPTION_MERGE_REQUIRE_VISUAL` | `true` | Require caption + visual pass |

Phase 4d reuses FARM cannot-link checks and `update_scene_graph_state`. Visual similarity uses per-object DINO `features` (384-d) unless SigLIP2 embeddings are present.

Responses are cached under `phase4/gemini_cache/` keyed by image + prompt, so a restarted batch does not re-call completed objects.

## Scene state fields written

- `object_caption`, `object_category`, `object_supercategory`
- `object_key_attributes`, `object_caption_decision`
- `object_caption_embedding`
- After 4d: inactive merged losers, updated `id_redirect`, merged caption histories

## Viewer API

`POST /api/query` with body `{"query": "red generator", "top_k": 15}`

Returns ranked object indices with semantic + geometric scores.

## Track A integration

Track A covers **pre-Phase 4** fragmentation only. Track B is caption → embed → post-caption merge → query.

When Track A coalescence lands, re-run Phase 4a on the merged map, then:

```bash
SCENE_IN=outputs/runs/RUN_ID/phase4/scene_state_with_crops.pt bash scripts/run_track_b.sh
```
