# Phase 4 — Recommendations

**Based on:** `comparison.md` (and underlying audits)  
**Date:** 2026-08-04  
**Goal:** Concrete adoption / adaptation / rejection guidance for Spatial GPT.

---

## 1. Concrete changes to close identified gaps

Each item cites the FARM technique/code to borrow from.

### R1 — Introduce a persistent object memory after Phase 2/3  
**Gap closed:** #1 (persistent memory)  
**FARM reference:** `scene_graph/map_update/models.py` (`SceneState`), `object_update.py`, snapshot `scene_state.pt`  
**Change in Spatial GPT:**
- After Gemini detections (and optionally EDM tracks), write a durable `objects.json` / `object_memory.pt` keyed by `instance_id` with: best crop, bbox history, frame names, optional Stella camera pose, SigLIP/crop embedding.
- Chat tools `locate_object` / `count_*` / `get_visual_evidence` should **read the memory first** and only re-detect on cache miss or `force=true`.
- Keep Gemini for first observation; stop re-spending `DETECT_MAX_FRAMES` every turn for the same label.

### R2 — Add a thin relational query layer on top of posed stations/objects  
**Gap closed:** #2 (relational predicates)  
**FARM reference:** `retrieval/spatial_reasoning/query_parser.py`, `predicates.py`, `methods.py` (`unified_soft_w50`)  
**Change:**
- When Stella trajectory exists, attach camera xyz (and later object xyz if available) to memory entries.
- Implement a **subset** of predicates first: `Near`, `LeftOf`/`RightOf` (view-dependent from shared frame), `Closest` — matching consultant questions.
- Parse with Gemini function-calling into a small `QueryGraph` schema (mirror FARM’s predicate names for interoperability), then score geometrically instead of asking Gemini to invent spatial relations in prose.

### R3 — Turn on and harden 3D/temporal identity (don’t ship EDM-off as default)  
**Gap closed:** #3 (association quality)  
**FARM reference:** `union_find.py`, `cannot_link.py`, `get_neighbors.py` (Hellinger + feature sim)  
**Change:**
- Short term: default `USE_EDM=true` once `model.pt` is vendored or fetched in bootstrap; document checkpoint in README.
- Medium term: if Stella + depth (or LiDAR) becomes available, replace/cross-check EDM with FARM-style **cannot-link** (two boxes in same frame ≠ same ID) and pose-consistent merge — even without full Gaussians.

### R4 — Build a minimal quantitative eval harness  
**Gap closed:** #4 (evaluation)  
**FARM reference:** `EVALUATION.md`, `scripts/eval_farm_scenes.py`, `eval/unified_scoring.py`, metrics Acc@1@IoU / Recall@K  
**Change:**
- Curate 20–50 labeled Maaksons queries: `{query, run_id, gt_count}` and/or `{query, gt_frame, gt_bbox}`.
- Metrics: count MAE / exact-match; locate top-1 frame hit; optional IoU if boxes labeled.
- Gate PRs that touch retrieval/detection on this suite (even if tiny).

### R5 — Optional: local open-vocab detector path beside Gemini  
**Gap closed:** #5  
**FARM reference:** `segmentation/yoloe.py`, `configs/yoloe_vocabulary.txt`  
**Change:**
- Prototype YOLOE (or a lighter open-vocab seg model) on **perspective yaw crops** of equirect frames, restricted to our construction vocab prompts.
- Use as proposal generator; keep Gemini as verifier for high-stakes counts.
- **Legal check first:** FARM/YOLOE is AGPL-3.0 — may require isolation or a non-AGPL alternative (e.g. other open-vocab detectors) for commercial Spatial GPT.

### R6 — Per-object captions + crop-level embeddings for retrieval  
**Gap closed:** #6  
**FARM reference:** `captioning/`, `semantic_retrieval.py` (RRF multi-channel; captions carry most signal)  
**Change:**
- After detection, caption each instance crop once (Gemini or local VLM); embed crop with SigLIP; store on memory object.
- Retrieve objects by embedding similarity instead of whole-frame SigLIP only — helps small `retrieval_hard` classes (wheelbarrow, etc.).

### R7 — Covisibility / co-occurrence edges (lightweight)  
**Gap closed:** #7  
**FARM reference:** `map_update/covisibility.py`  
**Change:**
- When two instance IDs appear in the same frame (or adjacent stations), record an edge.
- Use for “what’s near the crane?” without full metric predicates.

### Explicitly defer / reject for now

| FARM piece | Decision | Why |
|------------|----------|-----|
| Full ROS streaming mapper | Defer | Product is post-walk 360°, not robot RGB-D |
| Replace Stella with “assume poses” | Reject | We must keep video→pose path |
| Replace Gemini agent with Qwen-only stack | Defer | Cloud Gemini already productized; local 50 GB stack is ops-heavy |
| Adopt AGPL YOLOE into main binary | Reject until legal review | License risk |
| Viser as primary UI | Reject | Keep React stakeholder UX; optional internal debug viewer OK |
| FARM-Scenes as sole eval | Reject as primary | Domain mismatch (robot RGB-D); use only for R&D transfer experiments |

---

## 2. Experiments worth running before committing

### E1 — Memory hit-rate on multi-turn chat  
**Hypothesis:** Persistent object memory cuts Gemini calls ≥50% with same locate/count quality.  
**Plan:**
1. Implement R1 read-through cache on one run.  
2. Replay 10 chat transcripts (or scripted tool sequences) with/without memory.  
3. Measure: API calls, latency, answer agreement vs baseline.

### E2 — Relational predicate accuracy with Stella poses only  
**Hypothesis:** Even without metric object depth, camera-station geometry + detection yaw enables useful Near/LeftOf answers.  
**Plan:**
1. On a Stella-completed run, place instance anchors at camera position + bearing from bbox center yaw.  
2. Implement Near / LeftOf scoring (port logic patterns from `predicates.py`).  
3. Manually label 30 relational questions; report Acc@1.  
4. **Go/no-go** for R2 productization.

### E3 — EDM on vs off count error  
**Hypothesis:** `USE_EDM=true` reduces double-counting on `instance_track` classes.  
**Plan:**
1. Ensure `pipeline/edm/model.pt` available.  
2. Pick 5 walks with known GT equipment counts (crane, truck, column).  
3. Compare absolute count error EDM on/off.  
4. If error drops materially, flip default to `true`.

### E4 — Crop-SigLIP vs frame-SigLIP for `retrieval_hard` labels  
**Hypothesis:** Object-crop embeddings (R6) recover wheelbarrow / small tools better than full equirect SigLIP.  
**Plan:** Label 20 hard queries; measure top-5 frame recall before Gemini. Compare frame-index vs crop-index.

### E5 — YOLOE proposals vs Gemini-only (isolated sandbox)  
**Hypothesis:** Local proposals reduce cost with acceptable precision.  
**Plan:** Run AGPL YOLOE **only in a disposable research container** (not merged). Compare proposal recall vs Gemini boxes on 3 walks. If promising, evaluate non-AGPL alternatives for production.

### E6 — Cross-system smoke on FARM-Scenes warehouse (optional R&D)  
**Hypothesis:** Understanding FARM’s memory quality on construction-like scenes informs how dense our memory must be.  
**Plan:** Follow FARM README quickstart on GrandTour warehouse prebuilt `.pt`; query construction-like phrases; note failure modes (over-segmentation, caption noise). **Do not** treat Acc@1 as our KPI.

---

## 3. Promising on paper but may not transfer

| Technique | Why it looks good | Transfer risk to Spatial GPT |
|-----------|-------------------|------------------------------|
| **3D Gaussian + Hellinger association** | Strong identity under pose+depth | We lack metric depth on 360° walks; Stella is sparse monocular. Without depth, Gaussians are ill-posed. |
| **YOLOE open-vocab on robot RGB** | Real-time, no API | Equirect distortion; construction open-vocab ≠ our closed QA needs; **AGPL**. |
| **Qwen3.5 local caption + parse stack** | Reproducible, no cloud | ~50 GB VRAM ops vs Gemini API; worse fit for laptop field demos. |
| **`unified_soft_w50` locked protocol** | Paper-reproducible | Tuned for ScanNet/HM3D/FARM-Scenes object graphs — weights won’t be optimal for equirect station graphs without retuning. |
| **Visible-mask IoU scoring** | Fair occlusion-aware metric | Needs depth + voxel support we don’t store. |
| **Streaming 5–10 Hz ROS mapper** | Live robots | Our capture model is passive walk video uploaded later. |
| **Multi-embedding RRF (Qwen3-VL + SigLIP2 + caption)** | Strong retrieval in FARM eval | Heavy local embedding servers; captions must exist first — cold-start costly on long walks. |
| **FARM Acc@1 numbers as targets** | Clear SOTA bar | Different task; chasing Acc@1@0.25 on referring may not improve “how many rebar bundles?” |

---

## Suggested adoption sequence

```
Week 1–2   R4 eval harness (tiny) + E3 EDM on/off
Week 2–4   R1 object memory + E1 chat cost/quality
Week 4–6   R6 crop embeddings + E4 retrieval_hard
Week 6–8   R2 predicates on Stella poses + E2 relational Acc
Then       Revisit YOLOE/legal (E5) and depth-backed association (R3 medium-term)
```

**North star:** Move Spatial GPT from “stateless Gemini-on-retrieved-frames” toward **FARM-like requeryable object memory**, while keeping equirect ingestion, construction counting modes, VPA change analysis, and the stakeholder UI.

---

## Summary verdict

| Action | Items |
|--------|-------|
| **Adopt / adapt** | Persistent object memory; relational predicates over Stella; crop embeddings; eval harness; EDM-on by default |
| **Test first** | E1–E5 above |
| **Reject / defer** | Full FARM stack swap; ROS-first architecture; AGPL YOLOE in product without counsel; Viser as UX; assuming RGB-D inputs |
