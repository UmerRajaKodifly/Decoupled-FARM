# Phase 3 — Structured Comparison: Spatial GPT vs FARM

**Inputs:** `our-pipeline-audit.md`, `farm-pipeline-audit.md`  
**Date:** 2026-08-04

---

## Framing (important)

These systems are **not drop-in substitutes**. They share a high-level goal—“find / reason about things in large real-world scenes from language”—but differ in input modality, memory representation, and product surface.

| | **Spatial GPT** (`spatial-app` + VPA) | **FARM** |
|--|--------------------------------------|----------|
| Mission | Construction-site intelligence from **360° walks** (count, locate, report, schedule, change) | **Relational 3D object memory** from **posed RGB-D** (find X relative to Y) |
| Memory | Per-query 2D detections; optional SLAM trajectory; no persistent object graph | Persistent object-level Gaussians + captions + embeddings (`scene_state.pt`) |
| Spatial language | Weak (frame evidence / station narrative) | Strong (metric + view-dependent predicates) |
| Evaluation | Operational / qualitative | Locked Acc@k / Recall@k / MRR on public + gated benchmarks |

Comparisons below are therefore about **architectural and methodological gaps**, not a single leaderboard score.

---

## Architecture differences

| Aspect | Spatial GPT | FARM | Why it matters |
|--------|-------------|------|----------------|
| **Pipeline shape** | Query-triggered stages (retrieve → detect → optional track/report) | Always-on mapping → query against stored graph | FARM amortizes perception into a **requeryable memory**; Spatial GPT re-runs detection per question |
| **Geometry** | Stella SLAM optional (Phase 3) or VPA Zone A for two-walk Sim(3) | **Assumes** poses + metric depth | Spatial GPT can start from raw video; FARM cannot without an upstream SLAM/depth stack |
| **Object identity** | EDM 2D crop matching across equirect frames (off by default) | Union-find + Hellinger + DINO in 3D + cannot-link | FARM identity is **3D-metric**; ours is **2D appearance** unless EDM is on |
| **Representation** | Frames, overlays, `instances.json`, station JSON | 3D Gaussians + voxels + covisibility bitsets | FARM supports geometric predicates; we mostly support “seen in frame F” |
| **Interfaces** | FastAPI + React chat, workplan, PDFs | Docker CLI, ROS 2, Viser | Ours is productized for site stakeholders; FARM is research/robot ops |
| **Online path** | Batch / interactive chat on a finished walk | Streaming ROS mapper at 5–10 Hz | FARM targets live robots; we target post-walk analysis |

**Structural implication:** Adopting FARM wholesale would replace our product stack, not just improve a stage. The reusable idea is **persistent object-level spatial memory + relational grounding**, not the ROS/Viser shell.

---

## Methodological differences

### Modeling choices

| Choice | Spatial GPT | FARM |
|--------|-------------|------|
| Detector | Gemini Vision (closed ~43-label vocab) | YOLOE open-vocab (~1.5k prompts) |
| Embeddings | SigLIP for **frame** retrieval | SigLIP2 + Qwen3(+VL) embeddings for **object** retrieval (RRF) |
| LLM | Gemini cloud (agent, detect, synthesize) | Local Qwen3.5-9B (caption + query parse) |
| Tracking | EDM (learned equirect matcher) / detection fallback | Classic 3D association (no learned tracker) |
| Spatial reasoning | Implicit in Gemini text / “where is X?” evidence cards | Explicit predicate algebra over 3D memory (`unified_soft_w50` / `joint_v1`) |
| Captions | Station-level site narrative, not per-object memory fields | Per-object VLM captions as primary retrieval signal |

### Techniques FARM uses that we don’t

1. **Persistent 3D object scene graph** with mergeable Gaussians  
2. **Cannot-link** same-frame identity constraints  
3. **Covisibility graph** for relational / proximity structure  
4. **LLM → QueryGraph → geometric predicates** (LeftOf, Between, Closest, …)  
5. **Multi-channel RRF retrieval** over object embeddings (not frame embeddings)  
6. **Locked eval protocol** with Acc@IoU / Recall@K / MRR  
7. **Open-vocabulary YOLOE** segmentation with depth-backed 3D stats  
8. **Streaming / ROS** online mapping path  

### Techniques we use that FARM doesn’t

1. **Native equirectangular 360° walk** pipeline (no depth/pose required for count/locate)  
2. **Closed-world construction vocab + counting modes** (instance / stock / presence / estimate)  
3. **Gemini function-calling product agent** with workplan assess and branded PDFs  
4. **VPA geometry-first two-walk change analysis** (Sim(3) + pose-gated VPR + SSIM)  
5. **EDM equirect instance matching** (domain-specific for 360°)  
6. **Stakeholder UX** (schedule Gantt, compare drawer, client reports)

---

## Performance

### Like-for-like?

**No shared benchmark exists in either repo.**

| Claim | Status |
|-------|--------|
| FARM Acc@1@0.25 ≈ 0.10 on FARM-Scenes, ≈ 0.26 on ReferIt3D-30 | Documented in FARM `EVALUATION.md` |
| Spatial GPT count / locate accuracy | **Not measured** in-repo |
| VPA alignment RMSE / SSIM | Measured only for geometry experiments |
| “FARM outperforms us empirically on construction counting” | **Unsupported** — different tasks, no head-to-head |

### Where comparison would be fair (if we built it)

| Task | Possible protocol | Caveat |
|------|-------------------|--------|
| Relational find (“crane left of rebar”) | Annotate Spatial GPT runs with GT boxes/poses; run FARM-style Acc@IoU | We lack metric depth/poses unless Stella (+depth proxy) is added |
| Instance count on 360° walks | Manual GT counts vs Spatial GPT / vs FARM-after-projection | FARM needs RGB-D conversion from equirect — non-trivial |
| Change detection | VPA pairs vs FARM multi-visit memory diff | FARM does not ship a scan-to-scan change product |

**Flag:** Do not treat FARM’s Acc@1 numbers as evidence that Spatial GPT is worse at “how many cranes?” — those metrics score **3D referring expression grounding**, not inventory counting.

---

## Engineering quality

| Dimension | Spatial GPT | FARM | Adoption note |
|-----------|-------------|------|---------------|
| **Docs** | Excellent product docs (`CODEBASE_OVERVIEW`, `PIPELINES`) | Excellent research/run docs (`README`, `EVALUATION`, `CLAUDE`) | Both mature |
| **Reproducibility** | Needs Gemini key, EDM weights (gitignored), optional Stella images | Docker-locked; public FARM-Scenes + eval scripts | FARM stronger for **paper-repro**; ours stronger for **client deploy without 50 GB VRAM** |
| **Tests** | Nearly none | Predicate/covisibility tests + full eval harness | FARM wins |
| **Deps footprint** | Host pip + optional Docker for SLAM; cloud Gemini | Full CUDA/ROS/vLLM stack ~50 GB VRAM | Ours far lighter for field laptops |
| **License** | Not emphasized as AGPL in audit scope (check separately) | **AGPL-3.0** (YOLOE) | May block closed-source product fusion |
| **Extensibility** | Clear stage/tool boundaries; vocab JSON | Clear `pipeline/steps.py` + retrieval methods registry | Both extendable; different extension points |
| **Maintainability** | Two pipelines (spatial-app + vpa) with some duplication (PDF builders) | Shared online/offline algorithm (good); dual Python install (venv + colcon) is a footgun | FARM algorithm cohesion is better; our product surface is broader |

**Adoption feasibility:** Porting *ideas* is feasible; running FARM as-is inside Spatial GPT’s deploy model is **hard** (VRAM, Docker-only, AGPL, RGB-D requirement).

---

## Gaps — what FARM does that we don’t

Ranked by **likely impact on our results / roadmap** (construction chat + site intelligence), not by research novelty alone:

| Rank | Gap | Likely impact | Notes |
|------|-----|---------------|-------|
| **1** | **Persistent object-level spatial memory** (requery without re-detect) | **High** | Directly improves multi-turn chat, locate consistency, and “show me all X” without reburning Gemini |
| **2** | **Explicit relational spatial predicates** (LeftOf, Near, Between, …) | **High** | Unlocks consultant-style questions we currently answer poorly or narratively |
| **3** | **3D association (union-find + cannot-link + Hellinger)** vs 2D EDM-off counting | **High** for counts | Would reduce double-counting if we had metric geometry; depends on depth/pose |
| **4** | **Quantitative eval harness** for grounding / retrieval | **High** (process) | Without this we cannot tell if FARM-inspired changes help |
| **5** | **Open-vocab local detector** (YOLOE) instead of Gemini-only boxes | **Medium–High** | Cuts API cost/latency; may hurt construction-specific accuracy without fine prompts/vocab |
| **6** | **Per-object captions + multi-embedding RRF** | **Medium** | Better semantic find than SigLIP-on-full-frames for small objects |
| **7** | **Covisibility graph** | **Medium** | Helps “near / in same view” without full metric predicates |
| **8** | **Streaming ROS mapper** | **Low–Medium** for current Maaksons walk product | Higher if we move to robot/LiDAR capture |
| **9** | **Published large-scale scene dataset (FARM-Scenes)** | **Low** direct | Useful for R&D experiments, not client PDF workflow |
| **10** | **Viser interactive 3D query UI** | **Low** as product | Nice demo; our React UI already serves stakeholders |

### Reverse gaps (we do, FARM doesn’t) — don’t lose these

- Equirect-native ingestion without depth  
- Counting-mode specialization + closed construction vocab  
- Workplan / progress assessment  
- VPA geometric before/after change reports  
- Client-facing PDF / chat product polish  

---

## Bottom line for Phase 4

FARM’s advantage is **memory + relational grounding + measurable retrieval**, under RGB-D+pose assumptions. Spatial GPT’s advantage is **360° video productization + construction workflows**. The highest-value path is **selective transfer** of FARM’s memory/retrieval ideas onto our Stella-posed (or depth-augmented) walks—not a full stack swap.
