# Phase 2 — Audit: FARM-Project

**Repo:** [https://github.com/GoldenGait/FARM-Project](https://github.com/GoldenGait/FARM-Project) (cloned as `FARM-Project/`)  
**Paper:** arXiv:2606.15476 — *Find Anything using Relational Spatial Memory*  
**Audit date:** 2026-08-04  
**Package import name:** `scene_graph` (PyPI project name: `farm-scene-graph`)

---

## 1. Architecture

### What FARM is

FARM (**Find Anything using Relational Spatial Memory**) is a **robot / embodied spatial memory** system. From posed RGB-D streams it builds, in real time (~5–10 Hz), an **open-vocabulary object-level 3D scene graph**, then answers **relational natural-language queries** (“the ladder *left of* the crates”) by parsing predicates and grounding them against that memory.

It is **not** a SLAM stack and **not** a construction-report product. It assumes metric depth + camera pose per frame and focuses on persistent object identity + relational retrieval.

### Directory layout


| Path                             | Role                                                                    |
| -------------------------------- | ----------------------------------------------------------------------- |
| `src/scene_graph/`               | Core library (zero ROS imports in pure logic)                           |
| `src/scene_graph/offline/`       | Offline driver + frame sources (`sens`, `npz`, `rosbag`, `frames-json`) |
| `src/scene_graph/pipeline/`      | Shared pure-function steps + `PipelineOrchestrator`                     |
| `src/scene_graph/segmentation/`  | YOLOE + DINOv3                                                          |
| `src/scene_graph/map_update/`    | Gaussian fusion, union-find, cannot-link, covisibility, pruning         |
| `src/scene_graph/captioning/`    | Async VLM caption workers                                               |
| `src/scene_graph/retrieval/`     | Multi-modal embeddings + spatial reasoning                              |
| `src/scene_graph/eval/`          | ReferIt3D, IRef-VLA, FARM-Scenes / largescale scorers                   |
| `ros/mapping/`                   | ROS 2 nodes (`streaming_mapper`, `frame_pub`, launches)                 |
| `scripts/`                       | CLI: map, view, query, eval                                             |
| `configs/`                       | YAML + ~1,500-class YOLOE vocabulary                                    |
| `docker/`, `run.sh`              | CUDA + ROS 2 Humble containerized runtime                               |
| `benchmarks/curated_utterances/` | Locked UID lists for paper subsets                                      |




### End-to-end data flow

```
Posed RGB-D frame(s)
  → Segment (YOLOE masks → 3D Gaussian + DINOv3 features + sparse voxels)
  → Filter (border, size, distance, uninformative labels, duplicate IoU)
  → get_neighbors (DINO cosine + Hellinger on Gaussians)
  → find_object_correspondence (union-find; cannot-link blocks same-frame merges)
  → update_scene_graph_state (Gaussian/voxel fusion; optional covisibility edges)
  → Async caption + multi-modal embeddings (optional but critical for retrieval)
  → Prune low-information objects
  → scene_state.pt  (SceneState pickle)

Query (NL)
  → LLM parse_query → QueryGraph (target + spatial predicates)
  → Multi-channel RRF semantic retrieval (caption text, SigLIP2, Qwen3-VL, …)
  → Geometric / soft predicate scoring (Near, LeftOf, Between, …)
  → Ranked object IDs (+ optional Viser visualization)
```



### ROS vs offline

Online (ROS 2) and offline share the **same algorithm**: `StreamingMapper._run_mapping_batch` (ROS node) is reused by `python -m scene_graph.offline.run`; only ingress differs (`RGBDFrame` topics vs `FrameSource` iterators). A secondary path `scripts/run_pipeline.py` drives `PipelineOrchestrator` from YAML (e.g. Replica) using the same `pipeline/steps.py` functions.

**Output artifact:** `scene_state.pt` — per-object 3D Gaussians, voxel AABBs, captions, embeddings, covisibility bitsets, image evidence refs (`map_update/models.py`).

---



## 2. Core method



### Mapping stack


| Component            | Method                                                                           | Files                                               |
| -------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------- |
| Detection / masks    | YOLOE v8l, open-vocab prompts (~1,517 classes in `configs/yoloe_vocabulary.txt`) | `segmentation/yoloe.py`                             |
| Merge features       | DINOv3 ViT-S/16 (bundled) or gated ViT-S+/16 (paper)                             | `segmentation/dino.py`                              |
| 3D object model      | 3D Gaussian (mean + cov6) + sparse voxel cloud (cap ~1000 voxels/object)         | YOLOE → `object_update.py`                          |
| Association          | Feature sim + Hellinger → union-find                                             | `get_neighbors.py`, `union_find.py`                 |
| Identity constraints | Same-frame **cannot-link** pairs                                                 | `cannot_link.py`                                    |
| Co-visibility        | Packed u64 adjacency bitsets                                                     | `covisibility.py`                                   |
| Captions             | Qwen3.5-9B via vLLM (async)                                                      | `captioning/`                                       |
| Retrieval embeddings | Qwen3-Embedding-0.6B, Qwen3-VL-Embedding-2B, SigLIP2, caption text               | `retrieval/spatial_reasoning/semantic_retrieval.py` |


Default association thresholds (`pipeline/steps.py`): `feature_sim_thresh=0.5`, `hellinger_thresh=0.8`.

### Retrieval / spatial reasoning

1. `parse_query()` — LLM (Qwen3.5-9B) decomposes NL into a `QueryGraph` with predicates such as `Near`, `On`, `Above`, `Below`, `NextTo`, `Between`, `Inside`, `LeftOf`, `RightOf`, `InFrontOf`, `Behind`, `Closest`, `Farthest`, …
2. `execute_spatial_query()` — RRF over embedding channels (`retrieval_mode=multi`), then predicate evaluation.
3. **Locked paper method:** `unified_soft_w50` — soft predicates with weight 0.50, soft class mismatch floor 0.3, `force_no_vlm=True` (eval disables VLM predicate fallback). Defined in `retrieval/spatial_reasoning/methods.py`.
4. **Interactive viewer default:** `joint_v1` (truncation-free anchor grounding, vectorized predicates) unless `VISER_SPATIAL_METHOD=unified_soft_w50`.

View-dependent predicates (`LeftOf` / `RightOf` / …) use stored shared image views; vertical axis is inferred for ScanNet (Z-up) vs HM3D (Y-up) in `predicates.py`.

### What LLMs/VLMs do vs geometry


| Stage                                         | LLM / VLM / vision net                    | Geometric / classical                  |
| --------------------------------------------- | ----------------------------------------- | -------------------------------------- |
| Detect                                        | YOLOE                                     | Depth unprojection, Gaussian fit       |
| Track / mergecannot_[link.py](http://link.py) | DINOv3 features                           | Hellinger, union-find, cannot-link     |
| Caption                                       | Qwen3.5-9B                                | Best-view crop selection               |
| Query parse                                   | Qwen3.5-9B                                | —                                      |
| Candidate retrieve                            | Multi-embedding RRF (+ SigLIP2)           | Active-object pool, alias geometry     |
| Spatial ground                                | Optional VLM prompts (off in locked eval) | Metric / soft predicates, covisibility |
| Score (eval)                                  | —                                         | 3D-AABB IoU or visible-mask IoU        |


**Divergence from Spatial GPT:** FARM has no Stella/EDM/Gemini; SigLIP2 is a **retrieval channel**, not the primary detector; object identity is **3D Gaussian association**, not 2D crop matching across equirect frames.

---



## 3. Data handling



### Hard input contract

Each frame must provide:

- RGB (`uint8` H×W×3)
- **Metric depth** (float32 metres; NaN/0 invalid)
- Intrinsics (pinhole)
- **Camera-to-world pose** (4×4)

FARM does **not** estimate pose or run SLAM. Wrong/missing poses break association.

### Supported sources (`DATA.md` + `offline/frame_sources/`)


| Source           | CLI                                        | Notes                                                                                        |
| ---------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------- |
| FARM-Scenes      | `--source frames-json`                     | JPEG + depth + `frames.json`                                                                 |
| ScanNet          | `--source sens`                            | `.sens` archives                                                                             |
| Habitat / custom | `--source npz`                             | `images`, `depths`, `camtoworlds`, `K`                                                       |
| Recorded bags    | `--source rosbag`                          | `RGBDFrame` msgs                                                                             |
| Replica          | `run_pipeline.py` + `configs/replica.yaml` | Demo path                                                                                    |
| LiDAR rigs       | `scripts/lidar_bag_to_frames.py`           | Projects PointCloud2 → synthetic depth (sqlite `.db3` only; mcap not supported by that tool) |




### Datasets / benchmarks


| Dataset                 | Role                                                                                           | Access                                     |
| ----------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------ |
| **FARM-Scenes**         | Primary large-scale release (7 scenes, 47,300 m², 3 platforms); GT + **prebuilt scene graphs** | Public HuggingFace                         |
| **ScanNet + ReferIt3D** | Indoor referring (NR3D/SR3D+)                                                                  | ScanNet ToS-gated                          |
| **HM3D + IRef-VLA**     | Large indoor referring                                                                         | HM3D ToS-gated; habitat-sim render on host |
| Replica                 | Config demo                                                                                    | Public                                     |


**Doc inconsistency:** `DATA.md` still claims grounding/QA harnesses are not included; `EVALUATION.md` + `src/scene_graph/eval/` **do ship the full harness**. Treat `DATA.md` as stale on that point.

### Assumptions baked in

1. Metric, calibrated RGB-D (or LiDAR-synthesized depth).
2. Known poses in a consistent world frame.
3. Primarily **pinhole** multi-camera / robot streams — not equirectangular 360° walks as first-class input.
4. Captions are required for eval-parity retrieval (“caption embedding channels carry most of the retrieval signal” — `EVALUATION.md`).
5. Open vocabulary (~1.5k YOLOE prompts), not a 43-label construction closed set — though construction scenes appear in FARM-Scenes (e.g. GrandTour warehouse / construction teaser).

---



## 4. Training / inference setup



### Training

**Inference-only.** No training loops, fine-tuning scripts, or `backward()` in application code. All backbones are frozen pretrained weights.

### Entry points


| Command                                                  | Purpose                               |
| -------------------------------------------------------- | ------------------------------------- |
| `./bootstrap_models.sh`                                  | Download YOLOE / MobileCLIP / SigLIP2 |
| `./run.sh build | shell | vllm | ros2 | stop`            | Docker lifecycle + vLLM servers       |
| `python -m scene_graph.offline.run`                      | Main offline mapping                  |
| `scripts/run_pipeline.py --config configs/replica.yaml`  | YAML offline orchestrator             |
| `scripts/view_scene_state.py`                            | Viser 3D viewer + Query panel         |
| `scripts/query_scene_graph.py`                           | CLI retrieval                         |
| `scripts/eval_farm_scenes.py`                            | FARM-Scenes predict + score           |
| `scripts/eval_referit3d_spatial.py` / `eval_iref_vla.py` | Academic predict                      |
| `scripts/eval_predictions.py`                            | Canonical scorer                      |




### Runtime / deps

- **Docker-only supported host path** (`pyproject.toml` note; README): CUDA 12.8+, ROS 2 Humble, PyTorch, YOLOE submodule.
- **Compute:** ~~50 GB GPU memory full stack (~~65 GB largest scenes); ~26 GB disk; tested RTX 5090 / PRO 6000 / Jetson Thor.
- **vLLM servers** (`./run.sh vllm`): Qwen3.5-9B `:8000`, Qwen3-Embedding-0.6B `:8002`, Qwen3-VL-Embedding-2B `:8006`.
- Optional gated DINOv3 ViT-S+/16 for paper-grade merging (bundled ViT-S/16 fragments room-scale scenes ~2×).



### Key config knobs

- Segmentation: YOLOE `conf_thres≈0.25`, `iou_thres≈0.5`, Mahalanobis depth filter, mask erosion.
- Offline: `--stride`, `--batch-size`, `--covisibility`, `--caption`, `--viser`.
- Locked eval: `unified_soft_w50`, `retrieval_mode=multi`, `candidate_pool_mode=active`, `geometry_mode=alias_expand`, top-100 preds.

---



## 5. Evaluation



### Protocol (locked)

Documented in `EVALUATION.md`:

- Predict: `parse_query` → `execute_spatial_query` with `unified_soft_w50` + multi RRF; no VLM rerank.
- Score top-10: **Acc@1@IoU ∈ {0.1, 0.25, 0.5}**, **Recall@K ∈ {1,3,5,10}**, **MRR**, **mean top-1 IoU**.
- IoU type: **3D-AABB** (FARM-Scenes) or **projected visible-mask** (ScanNet/HM3D; view picker `v1_largest_mask`, depth tol 0.15 m).
- Determinism: `recall@10` / `mean_top1_iou` stable; `acc@1`/`MRR` vary ±1–3 pts with LLM parser sampling.



### Reported numbers (from `EVALUATION.md`)

**FARM-Scenes (3D-AABB):**


| split     | utterances | acc@[1@0.25](mailto:1@0.25) | acc@[1@0.5](mailto:1@0.5) | recall@[10@0.25](mailto:10@0.25) | [MRR@0.25](mailto:MRR@0.25) |
| --------- | ---------- | --------------------------- | ------------------------- | -------------------------------- | --------------------------- |
| odin1     | 283        | 0.106                       | 0.057                     | 0.304                            | 0.175                       |
| grandtour | 598        | 0.099                       | 0.045                     | 0.333                            | 0.167                       |
| spot†     | 74         | 0.189                       | 0.135                     | 0.405                            | 0.268                       |


† `spot` withheld from public dataset (license).

**ReferIt3D:** 5-scene parity acc@[1@0.25](mailto:1@0.25) ≈ 0.160–0.173; 30-scene headline **acc@[1@mask-0.25](mailto:1@mask-0.25) = 0.256**.

**IRef-VLA:** 30-scene headline **acc@[1@mask-0.25](mailto:1@mask-0.25) = 0.051** (hard large-scale referring).

Absolute Acc@1 is modest — consistent with open-vocabulary 3D grounding on large multi-room / outdoor scenes, not closed-set detection.

### Engineering quality of eval

- Full predict → convert → score pipeline with curated UID lists.
- Scorer bit-exact vs paper prediction files (claimed for FARM-Scenes).
- Unit tests for spatial predicates / covisibility (`tests/`); broader coverage not comprehensive (`CLAUDE.md`).

---



## 6. Known weaknesses

1. **Requires poses + metric depth** — cannot ingest raw 360° video without an upstream SLAM/depth stack.
2. **Heavy compute / Docker lock-in** — ~50 GB VRAM; host `pip install` unsupported.
3. **DINOv3 backbone sensitivity** — bundled ViT-S/16 under-merges vs gated ViT-S+/16 on ScanNet-scale scenes.
4. **LLM parser non-determinism** — Acc@1/MRR jitter ±1–3 points.
5. **Caption dependency** — mapping without `--caption` underperforms retrieval benchmarks.
6. **AGPL-3.0** (YOLOE lineage) — licensing may block proprietary product adoption without careful legal review.
7. **Pickle** `scene_state.pt` — security warning; only trust local/known sources.
8. `DATA.md` **stale** vs shipped eval harness.
9. **Interactive vs paper methods diverge** (`joint_v1` vs `unified_soft_w50`).
10. **LiDAR→frames tooling gaps** (mcap unsupported in one converter).
11. **Absolute accuracy still low** on hard referring (especially IRef-VLA) — SOTA relative, not “solved.”
12. **Sparse TODOs** in code; incomplete/legacy Ollama paths remain in some configs (`configs/indoor.yaml`) alongside production vLLM.

---



## Divergence from Spatial GPT (preview for Phase 3)


| Axis             | Spatial GPT / VPA                                                                | FARM                                             |
| ---------------- | -------------------------------------------------------------------------------- | ------------------------------------------------ |
| Primary input    | Equirect 360° video                                                              | Posed RGB-D                                      |
| Pose source      | Stella SLAM (optional / Zone A)                                                  | External (required)                              |
| Object memory    | Ephemeral per-query detections + optional EDM tracks; no persistent 3D object DB | Persistent `scene_state.pt` Gaussians            |
| Spatial language | Weak (“where is X?” → frame evidence); no LeftOf/Between graph                   | First-class relational predicates over 3D memory |
| Detection        | Gemini closed vocab (~43)                                                        | YOLOE open vocab (~1.5k)                         |
| LLM              | Gemini cloud API                                                                 | Local Qwen via vLLM                              |
| Eval             | None for grounding                                                               | Full locked harness + published numbers          |
| Product layer    | Chat UI, workplan, PDFs, scan compare                                            | Research mapping + Viser + eval scripts          |


FARM’s teaser explicitly includes a **15,000 m² construction site**, so domain overlap exists; the **method** (object-level relational spatial memory) is what diverges.

---



## Audit notes / confidence

- Claims verified against `README.md`, `EVALUATION.md`, `DATA.md`, `CLAUDE.md`, `pyproject.toml`, `pipeline/steps.py`, `predicates.py`, and the explore agent’s file map.
- Full Docker build / FARM-Scenes download / live mapping was **not** executed in this workspace (multi-tens-of-GB GPU + models). Reported metrics are taken from repo docs, not re-measured here.
- Submodule `third_party/yoloe` may need `git submodule update --init` before a Docker build; clone used recursive contents where present.

