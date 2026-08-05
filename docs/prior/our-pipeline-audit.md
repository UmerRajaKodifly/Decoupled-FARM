# Phase 1 — Audit: ss-spatial-gpt (Spatial GPT)

**Repo:** https://github.com/s4rmx/ss-spatial-gpt (cloned as `ss-spatial-gpt/`)  
**Audit date:** 2026-08-04  
**Scope:** Full workspace (`spatial-app/` + `vpa-pipeline/`). “Spatial GPT” in product terms is primarily `spatial-app/`; `vpa-pipeline/` is the complementary geometry-first scan-comparison stack that Spatial GPT also consumes for Stella SLAM.

---

## 1. Architecture

### What the repo is

`ss-spatial-gpt` consolidates **two complementary construction-site pipelines** that share Stella VSLAM:

| Subsystem | Role | Interface |
|-----------|------|-----------|
| `spatial-app/` | **Spatial GPT** — single-walk chat-first object tracking, site reports, workplan assessment | FastAPI + React, or `run_pipeline.py` CLI |
| `vpa-pipeline/` | **VPA** — two-walk geometric alignment + before/after change analysis | Docker + shell (`run_e2e.sh`) |

They are independent products that share the Stella Docker image and export scripts under `vpa-pipeline/vslam/`.

### Spatial GPT (`spatial-app`) end-to-end data flow

```
360° video / frame folder
  → Frame extraction (OpenCV keyframes)
  → [Chat / CLI query]
  → Phase 0: Query parse (Gemini + closed vocab) → label, intent, counting_mode
  → Phase 1: SigLIP embed + cosine top-K retrieval (yaw-cropped equirect)
  → Phase 2: Gemini bbox detection on DETECT_MAX_FRAMES
       ├─ instance_track → optional EDM temporal ID + cross-window merge
       ├─ stock_unit → densest single-frame count
       ├─ presence → frame-fraction (+ station text fallback)
       └─ vlm_estimate → narrative quantity estimate
  → Phase 3 (optional): Stella SLAM → sample stations → Gemini station VQA → site synthesis
  → Reports (HTML/PDF), workplan assess, text-only compare of two runs
```

**Key modules / classes**

| Path | Responsibility |
|------|----------------|
| `agent/orchestrator.py` (`SiteAgentOrchestrator`) | Gemini function-calling chat loop |
| `agent/tools.py` | 9 tool schemas + dispatch into stages |
| `pipeline/query_parser.py` | NL → vocab label / counting_mode / intent |
| `pipeline/frame_extractor.py` | Video/dir → JPEG keyframes |
| `pipeline/siglip_retriever.py` (`SigLIPRetriever`) | Embedding cache + top-K retrieval |
| `pipeline/gemini_detector.py` | Per-frame Gemini bbox detection |
| `pipeline/stages.py` | All count/locate/presence/report stage functions |
| `pipeline/edm/*` (`EDMMatcher`, `MultiWindowTracker`) | Deep metric instance tracking |
| `scene_understanding/*` | Stella, station sampling, VQA, synthesis |
| `backend/routes/*` | Runs, chat, reports, workplan, compare APIs |
| `workplan/*`, `compare/*`, `report/*` | Schedule gap analysis, scan-vs-scan text diff, PDF/HTML |

**Chat routing** (primary UX): user message → Gemini chooses a tool → tool runs a stage → evidence JSON + overlays written under `outputs/<run_id>/`.

### VPA (`vpa-pipeline`) end-to-end data flow

```
scan1.mp4 + scan2.mp4
  → Zone A: Stella SLAM both walks → localize scan1 in scan2 map → Sim(3) (Umeyama+RANSAC)
  → Zone B: DINOv2 VPR × soft pose gate (banded independent matching) → confident pairs + SSIM
  → Zone C: Gemini per-pair VQA → site change report (MD/HTML/PDF) + optional agent chat
```

Formal method writeup: `vpa-pipeline/docs/TECHNICAL_METHOD.md`.

---

## 2. Core method

### What is “GPT” doing vs. other components?

“Spatial GPT” is **not** a trained spatial foundation model. It is an **inference-time orchestration** of frozen vision models + Gemini (LLM/VLM) behind a closed-world tool agent.

| Component | Model / method | Role |
|-----------|----------------|------|
| **Query understanding** | Gemini text | Map NL → 43-label construction vocabulary + counting mode |
| **Frame retrieval** | SigLIP (`ViT-SO400M-14-SigLIP-384` via `open_clip`) | Semantic top-K over equirect keyframes (multi-yaw crops) |
| **Detection / counting / locate** | Gemini Vision | Bounding boxes (0–1000 norm), presence, quantity estimates |
| **Instance dedup** | EDM (TorchScript, CVPR 2025 equirect matcher) | Match detection crops across frames within temporal windows |
| **Camera poses / map** | Stella VSLAM (ORB, equirect) | Trajectory + sparse map for Phase 3 only |
| **Station / site narrative** | Gemini Vision + text | Per-station structured VQA + Markdown synthesis |
| **Chat agent** | Gemini function calling | Tool routing; forbidden from inventing counts/locations |
| **Workplan / compare** | Gemini text-only | Schedule gap assessment; narrative change between two runs |

**GPT (Gemini) does:** NL understanding, open-vocabulary bbox detection on retrieved frames, station analysis, report synthesis, agent tool selection, workplan expansion/assessment, text-only scan compare.

**Non-GPT spatial/vision components do:** keyframe extraction, SigLIP indexing, EDM tracking, Stella SLAM geometry, SSIM / DINOv2 VPR (VPA only), optional SAM2 masks.

### Domain specialization

Detection is **closed-world** to `construction_site_object_vocabulary.json` (~43 labels derived from prior Maaksons Gemini VQA runs), with counting modes:

- `instance_track` — distinct equipment (cranes, trucks, columns)
- `stock_unit` — staged materials (rebar bundles)
- `presence` — slabs/walls/rubble (coverage fraction)
- `vlm_estimate` — irregular quantities

### VPA geometric core (shared Stella)

Cross-visit registration is geometry-first: localize-then-Sim(3), then DINOv2 place embeddings fused with soft pose consistency and confidence gating (`c ≥ 0.35`). Change language is conditioned only on confident matched pairs.

---

## 3. Data handling

### Inputs

| Pipeline | Input | Assumptions |
|----------|-------|-------------|
| Spatial GPT | One equirectangular 360° walk video (`mp4`/`mov`) **or** a folder of frames | Outdoor construction site; objects in vocab; no metric depth required for Phases 0–2 |
| VPA | Two videos `scan1.mp4` + `scan2.mp4` + `meta.json` (`run_id`, `frame_skip`) | Same site, different times; roughly overlapping walk coverage; monocular scale |

### Preprocessing

- **Keyframe extraction:** `FRAME_STRIDE_SECONDS` (default 3s); resize to `FRAME_MAX_SIDE` (1920); JPEG quality 85.
- **SigLIP:** Each 360° frame encoded at yaw crops 0°/90°/180°/270°; embeddings cached as `.npy` under `siglip_cache/`.
- **Gemini payloads:** Max side `GEMINI_MAX_IMAGE_SIDE` (1280), JPEG quality 85.
- **Stella:** Equirect camera YAML auto-selected by resolution (`configs/equirectangular_1920.yaml` / `_3840.yaml`); video frame skip configurable.
- **VPA Zone A:** `frame_skip` in `meta.json` (typical 15) for SLAM rate.

### Datasets

- **No public academic benchmark** is wired into Spatial GPT.
- Operational data is **Maaksons construction-site** walks (referenced in vocab source notes and report branding).
- VPA `experiments/` contains an **alignment method comparison** on private PCD/trajectory artifacts (baseline vs stratified vs gravity vs ICP) with RMSE/coverage metrics in `experiments/outputs/summary.csv` — geometry experiments only, not object-detection benchmarks.
- Demo path: `vpa-pipeline/scripts/bootstrap_demo_run.sh` seeds precomputed matching outputs for Zone C without GPU.

### Baked-in assumptions

1. Equirectangular 360° (not pinhole RGB-D).
2. Closed construction vocabulary; free-form objects outside vocab are poorly supported.
3. Phases 0–2 need **no poses/depth**; Phase 3 / VPA need SLAM (monocular → map units, not absolute metres).
4. Spatial-app **compare** is text-only (site synthesis narratives), not geometrically registered like VPA.
5. `USE_EDM` defaults to **false** — production counting often uses detection-only fallback unless explicitly enabled.
6. `outputs/` and `vpa-pipeline/data/runs/` may be **symlinks** to external storage (fragile if originals move).

---

## 4. Training / inference setup

### Training

**No training code.** All models are pretrained / API:

- SigLIP / open_clip weights from HuggingFace
- EDM TorchScript weights (`pipeline/edm/model.pt` — gitignored per commit `SX-3508 ignore local EDM weights`; must be supplied locally)
- Stella ORB vocabulary (`orb_vocab.fbow`)
- Gemini via Google API
- Optional SAM2 via HF token

### Entry points

| Entry | Purpose |
|-------|---------|
| `./scripts/start_spatial_app.sh` | FastAPI `:8000` + Vite `:5173` |
| `uvicorn backend.main:app` | API only |
| `spatial-app/run_pipeline.py` | Batch CLI: parse → count → optional scene → report |
| `vpa-pipeline/scripts/build_images.sh` | Build 4 Docker images |
| `vpa-pipeline/scripts/run_e2e.sh <run_id>` | Full VPA Zone A→B→C |
| `vpa-pipeline/scripts/chat.sh <run_id>` | VPA agent REPL |

### Key hyperparameters (Spatial GPT)

| Knob | Default | Effect |
|------|---------|--------|
| `GEMINI_MODEL` / `AGENT_MODEL` | `gemini-2.0-flash` | Detection / agent |
| `RETRIEVAL_TOP_K` | 20 | SigLIP candidates |
| `DETECT_MAX_FRAMES` | 5 | Hard Gemini image budget |
| `USE_EDM` | `false` | Temporal instance tracking |
| `FRAME_STRIDE_SECONDS` | 3 | Keyframe density |
| `SCENE_SAMPLE_FRAMES` | 8 | Site-report stations |
| `RETRIEVAL_SCORE_MIN` | 0.05 | Degraded retrieval gate |
| EDM IoU/center/cert thresholds | 0.05 / 0.2 / 0.02 | Match acceptance |

### Dependencies

- `spatial-app/requirements.txt`: torch, open_clip, google-generativeai/genai, fastapi, opencv, reportlab, …
- VPA: Docker images (Stella, stella-tools, matching/DINOv2, site-report); host Python not required for full e2e
- Frontend: Node 18+, React/Vite/Tailwind
- GPU recommended for SigLIP/EDM/Stella; Gemini is cloud API

---

## 5. Evaluation

### Spatial GPT (`spatial-app`)

**No quantitative benchmark harness** was found (no Acc@k, mAP, count-error scripts, or held-out labeled sets in-repo).

Quality signals that *do* exist:

- Per-run **process logs** (`logs/process.jsonl`) with stage timings and `ok|error|degraded` statuses
- Agent session JSONL under `ai/agent_sessions/`
- Qualitative client PDFs / HTML site reports
- UI evidence overlays for human inspection

### VPA

| Artifact | What it measures |
|----------|------------------|
| `step3_aligned_pairs.json` | VPR score, pose distance, confidence gate |
| `step5_baseline_comparison.json` | Per-pair SSIM |
| `experiments/outputs/summary.csv` | Sim(3) RMSE / coverage for alignment method variants |
| `vslam/kodifly` benchmarks | Timing / resolution JSON for SLAM export |

These evaluate **registration / matching**, not object counting or referring accuracy.

### Flag

**Empirical performance vs. academic SOTA cannot be assessed from this repo alone.** There are no shared benchmark numbers comparable to FARM’s ReferIt3D / FARM-Scenes tables.

---

## 6. Known weaknesses

Documented and code-evident limitations:

1. **No metric evaluation** for detection/count/locate — hard to know when changes help.
2. **`USE_EDM=false` by default** — instance counts may double-count the same object across frames unless EDM is enabled and `model.pt` is present.
3. **EDM weights not in git** (ignored) — reproducibility requires an external checkpoint.
4. **SigLIP reliability** — `retrieval_hard` vocab flags and `RETRIEVAL_SCORE_MIN` acknowledge weak retrieval for small/ambiguous classes; degraded path still returns closest frames (risk of false evidence).
5. **Hard Gemini frame budget** (`DETECT_MAX_FRAMES=5`) — can miss objects outside top SigLIP hits.
6. **Closed vocab (43 labels)** — open-world site objects outside vocabulary fall through fallback alias matching.
7. **Stella fallback** — if Docker/GPU unavailable, Phase 3 uses a **synthetic linear trajectory** (site report continues with weaker spatial grounding). Documented in `CODEBASE_OVERVIEW.md` / troubleshooting.
8. **Monocular SLAM scale** — distances are map units; absolute metres unavailable without external scale (`TECHNICAL_METHOD.md` §9.3).
9. **Spatial-app compare ≠ VPA** — compare is text-only narrative diff; no Sim(3) / pose-gated pairs. Easy to confuse product features.
10. **Almost no automated tests** — only `vpa-pipeline/packages/vpa_ai/tests/test_enrich_station.py` outside vendored Stella; spatial-app has no unit tests.
11. **API cost / rate limits** — heavy Gemini usage with retries; concurrent workers can hit 429s.
12. **Product gaps vs. roadmap** (`spatial-app/notes.md`) — structured compliance specs, issue tickets, role-based views, and closed-loop resolution are planned, not shipped as first-class modules.
13. **Equirect-specific** — pipeline assumes 360° walks; pinhole RGB-D / robot streams are out of scope for Spatial GPT proper.

---

## Audit notes / confidence

- Architecture and method claims are grounded in repo docs (`README.md`, `CODEBASE_OVERVIEW.md`, `PIPELINES.md`, `spatial-app/CODEBASE.md`) and cross-checked against entry points (`run_pipeline.py`, `pipeline/orchestrator.py`, `pipeline/stages.py`, `edm_matcher.py`).
- Full end-to-end execution was **not** run here (no `GEMINI_API_KEY`, no site video, no EDM weights, no Stella image build in this workspace). Behavior of degraded/fallback paths is inferred from code + docs; flag any claim that depends on live API responses as unverified at runtime.
- Primary comparison target for later phases should treat **Spatial GPT object memory / query** separately from **VPA geometric change analysis** — FARM is closer to the former’s “find things with spatial language” goal, while VPA’s geometry-first registration is a different product axis.
