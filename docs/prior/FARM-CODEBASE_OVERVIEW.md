# Codebase Overview — FARM (Find Anything using Relational Spatial Memory)

This document is for understanding what FARM does, why it exists, how each piece of code works, what files to touch when something needs changing, and how to run everything end-to-end. No prior knowledge of the codebase is assumed. It is written in the same spirit as Spatial GPT’s `CODEBASE_OVERVIEW.md`.

**Repository:** `FARM-Project/` (https://github.com/GoldenGait/FARM-Project)  
**Paper:** [arXiv:2606.15476](https://arxiv.org/abs/2606.15476)  
**Python package import name:** `scene_graph`  
**License:** AGPL-3.0-or-later (driven by YOLOE / Ultralytics)

---

## Table of contents

1. [What this repository is](#1-what-this-repository-is)
2. [High-level architecture](#2-high-level-architecture)
3. [Repository layout at a glance](#3-repository-layout-at-a-glance)
4. [Setup and first run](#4-setup-and-first-run)
5. [Part A — Mapping: building the spatial memory](#part-a--mapping)
   - [A1. Input contract](#a1-input-contract)
   - [A2. Frame sources (offline)](#a2-frame-sources-offline)
   - [A3. Shared per-batch pipeline](#a3-shared-per-batch-pipeline)
   - [A4. Segmentation (YOLOE + DINOv3)](#a4-segmentation-yoloe--dinov3)
   - [A5. Filtering](#a5-filtering)
   - [A6. Neighbor lookup and correspondence](#a6-neighbor-lookup-and-correspondence)
   - [A7. Scene-state update, voxels, covisibility](#a7-scene-state-update-voxels-covisibility)
   - [A8. Captioning and embeddings](#a8-captioning-and-embeddings)
   - [A9. Pruning and regions](#a9-pruning-and-regions)
   - [A10. What `scene_state.pt` contains](#a10-what-scene_statept-contains)
6. [Part B — Retrieval: finding things with language](#part-b--retrieval)
   - [B1. Two retrieval APIs](#b1-two-retrieval-apis)
   - [B2. Query parsing](#b2-query-parsing)
   - [B3. Semantic retrieval (multi-channel RRF)](#b3-semantic-retrieval-multi-channel-rrf)
   - [B4. Spatial predicates](#b4-spatial-predicates)
   - [B5. Method profiles (`unified_soft_w50`, `joint_v1`, …)](#b5-method-profiles)
   - [B6. Viser Query panel and CLI](#b6-viser-query-panel-and-cli)
7. [Part C — Online ROS 2 path](#part-c--online-ros-2-path)
8. [Part D — Evaluation harness](#part-d--evaluation-harness)
9. [Configuration and environment reference](#9-configuration-and-environment-reference)
10. [Models, licenses, and compute](#10-models-licenses-and-compute)
11. [Data shapes quick reference](#11-data-shapes-quick-reference)
12. [Troubleshooting](#12-troubleshooting)
13. [Where to change what](#13-where-to-change-what)

---

## 1. What this repository is

FARM is a **robot / embodied spatial memory** system. Given a stream of posed RGB-D frames, it builds — in real time (~5–10 Hz) — a compact, **open-vocabulary, object-level 3D memory**. Each remembered object carries:

- 3D Gaussian geometry (mean + covariance)
- a sparse voxel cloud (for tight AABBs and evidence)
- a VLM caption and multi-modal embeddings
- viewpoint / mask observation history
- optional co-visibility links to other objects

You then **find things in that memory from natural language**, including relational constraints such as “the ladder *left of* the crates” or “the yellow forklift near the shelves.”

This is **not**:

- a SLAM system (poses and metric depth are **inputs**, not outputs)
- a construction-site report product (no workplan, PDFs, or scan-to-scan change analysis)
- a training framework (inference-only; all models are frozen pretrained weights)

It **is** the official implementation of the FARM paper: offline dataset mapping, live ROS 2 mapping, interactive 3D viewer with language retrieval, and the full benchmark harness used for the paper’s grounding numbers.

**Domain note:** The teaser and FARM-Scenes dataset include large outdoor / construction-like sites (e.g. a 15,000 m² scene, GrandTour warehouse). The method is general open-vocabulary 3D grounding, not construction-specific closed vocab.

---

## 2. High-level architecture

```
                     FARM-Project
                           │
         ┌─────────────────┴──────────────────┐
         │                                    │
  src/scene_graph/                      ros/
  (core library,                        (thin integration)
   zero ROS imports)                          │
         │                              mapping/ nodes
         │                              mapping_msgs/
         │                                    │
         └────────────┬───────────────────────┘
                      │
              StreamingMapper
         (_run_mapping_batch — ONE algorithm)
                      │
         ┌────────────┴────────────┐
         │                         │
   Offline ingress            Online ingress
   FrameSource iterators      frame_pub → RGBDFrame topics
   (sens / npz /              (+ odin1_depth_pub for LiDAR)
    frames-json / rosbag)
         │                         │
         └────────────┬────────────┘
                      ▼
              scene_state.pt
                      │
         ┌────────────┴────────────┐
         │                         │
   SceneGraphRetriever      parse_query +
   (embedding retrieve)     execute_spatial_query
                            (relational / paper path)
         │                         │
         └────────────┬────────────┘
                      ▼
            ranked object IDs + scores
            (Viser / CLI / eval scripts)
```

**Design principle:** Online and offline share the **same algorithm code**. `StreamingMapper` runs in both modes; only how frames arrive differs. Pure pipeline steps live in `src/scene_graph/pipeline/steps.py` so the ROS-free library and the ROS node stay aligned.

A secondary offline path — `scripts/run_pipeline.py` + YAML (`configs/replica.yaml`) — drives `PipelineOrchestrator` without initializing ROS, still calling the same step functions.

---

## 3. Repository layout at a glance

```
FARM-Project/
├── README.md                 quick start, architecture sketch, citation
├── DATA.md                   input formats, bring-your-own-data, LiDAR converter
├── EVALUATION.md             locked eval protocol + expected numbers
├── CLAUDE.md                 maintainer / agent guidance
├── THIRD_PARTY_NOTICES.md    per-model licenses
├── CITATION.cff
├── pyproject.toml            package `farm-scene-graph` → import `scene_graph`
├── run.sh                    build | shell | vllm | ros2 | stop
├── bootstrap_models.sh       YOLOE + MobileCLIP + SigLIP2 downloads
│
├── src/scene_graph/          ★ core library
│   ├── offline/              driver + frame_sources/
│   ├── pipeline/             PipelineOrchestrator + steps.py
│   ├── map_update/           Gaussians, union-find, cannot-link, covisibility, pruning
│   ├── segmentation/         YOLOE + DINOv3
│   ├── captioning/           async vLLM caption workers
│   ├── retrieval/            SceneGraphRetriever + spatial_reasoning/
│   ├── eval/                 ReferIt3D, IRef-VLA, FARM-Scenes scorers
│   ├── regions/              multi-room clustering + LLM labels
│   ├── storage/              HDF5 image store
│   ├── datasets/             Replica / ScanNet / NPZ loaders (YAML path)
│   ├── visualization/        Viser 3D viewer + Query panel
│   ├── debug/                JSONL pipeline tracer
│   ├── llm_utils/            LLMInterface / EmbedInterface (vLLM clients)
│   ├── config.py             PipelineConfig from YAML
│   ├── runtime_paths.py      model resolution (incl. DINOv3 backbone picker)
│   ├── scene_state_io.py     save/load scene_state.pt
│   └── camera_config.py      multi-camera topic wiring for ROS
│
├── ros/
│   ├── mapping/              ament-python pkg: StreamingMapper, frame_pub, launches
│   └── msgs/                 ament-cmake: RGBDFrame, DetectedObject, … msgs + Siglip2TextEmbed.srv
│
├── scripts/                  CLIs: view, query, map, eval, LiDAR→frames, …
├── configs/                  replica.yaml, indoor.yaml, yoloe_vocabulary.txt (~1516 prompts)
├── benchmarks/               curated utterance UID lists for paper subsets
├── models/                   dinov3-vits16 (committed); others downloaded
├── docker/                   Dockerfile (CUDA 12.8 + ROS 2 Humble), compose, entrypoint
├── tests/                    spatial predicates + covisibility
└── third_party/yoloe         submodule (AGPL) — required for Docker build
```

---

## 4. Setup and first run

### Prerequisites

- NVIDIA GPU with **CUDA 12.8+** drivers
- Docker + NVIDIA Container Toolkit + `docker compose`
- ~**50 GB** GPU memory for the full stack (mapping + vLLM + viewer); ~**65 GB** on largest scenes
- ~**26 GB** disk for models/data
- Host-side `pip install` of the package is **not supported** — everything runs inside the container

Tested on RTX 5090 / RTX PRO 6000 and Jetson Thor.

### Install (once)

```bash
git clone --recursive https://github.com/GoldenGait/FARM-Project.git
cd FARM-Project
# If you forgot --recursive:
git submodule update --init --recursive

./bootstrap_models.sh    # YOLOE + MobileCLIP + SigLIP2 (public URLs, no HF account)
./run.sh build           # builds scene_graph:latest
```

Optional paper-grade DINOv3 (gated; auto-preferred when present):

```bash
huggingface-cli download facebook/dinov3-vits16plus-pretrain-lvd1689m \
  --local-dir models/dinov3-vits16plus
```

Bundled `models/dinov3-vits16` works offline but can **fragment room-scale scenes ~2×** vs ViT-S+/16 (see `EVALUATION.md`).

### Quickstart — FARM-Scenes warehouse

```bash
# Host: download one scene + prebuilt graph (~350 MB)
hf download GoldenGait/FARM-Scenes --repo-type dataset \
  --include "scenes/grandtour/2024-11-25_warehouse/*" \
  --include "scene_graphs/grandtour/2024-11-25_warehouse.pt" \
  --local-dir /path/to/farm_scenes

./run.sh shell /path/to/farm_scenes    # mounts at /data inside container
```

**1. Explore a prebuilt memory**

```bash
python scripts/view_scene_state.py \
  --pt /data/scene_graphs/grandtour/2024-11-25_warehouse.pt \
  --cloud /data/scenes/grandtour/2024-11-25_warehouse/cloud.npz
# → http://localhost:8080  (allow 30–60s for point cloud load)
```

**2. Ask relational queries** (in the viewer’s Query panel, or CLI)

```bash
./run.sh vllm   # Qwen3.5-9B :8000, embeddings :8002 / :8006

python scripts/query_scene_graph.py \
  --pt /data/scene_graphs/grandtour/2024-11-25_warehouse.pt \
  --query "the ladder near the shelves"
```

**3. Rebuild the memory from raw RGB-D**

```bash
python -m scene_graph.offline.run \
  --source frames-json \
  --frames-json-dir /data/scenes/grandtour/2024-11-25_warehouse \
  --save-path /data/out/warehouse.pt \
  --covisibility --caption
# Add --viser to watch live (~6× slower); omit for full speed
```

**Synthetic smoke test (no external data):** see `DATA.md` — write a tiny NPZ with random RGB + flat depth, then map with `--source npz`. This checks plumbing, not retrieval quality.

---

<a name="part-a--mapping"></a>
## Part A — Mapping: building the spatial memory

---

### A1. Input contract

Every frame must provide:

| Field | Type | Meaning |
|-------|------|---------|
| RGB | `uint8` H×W×3 | Color image |
| Depth | `float32` H×W, **metres** | Metric depth; NaN/0 = invalid |
| Intrinsics | 3×3 (or fx/fy/cx/cy) | Pinhole camera |
| Pose | 4×4 `T_world_cam` | Camera-to-world |

**FARM does not estimate pose or run SLAM.** Wrong or missing poses break cross-frame association.

Two dict shapes are accepted by `FrameSource` / `_decode_batch` (`offline/frame_sources/base.py`):

1. **ROS-native:** `{"camera", "rgbd_msg": RGBDFrame, "received_time"}`
2. **Pre-decoded:** `{"camera", "rgb", "depth_f32", "T_world_cam", "rgb_instrinsics", "depth_instrinsics", "stamp_ns", "frame_id", "received_time"}`  
   (Note the historical spelling **`instrinsics`**.)

Anything expressible as `(rgb, metric depth, intrinsics, pose)` can be fed in — use a built-in source or write a ~30-line `FrameSource` subclass.

---

### A2. Frame sources (offline)

**Entry point:** `python -m scene_graph.offline.run`

| `--source` | Input | Typical use |
|------------|-------|-------------|
| `frames-json` | `frames.json` + JPEG + depth (`.npy` or uint16 PNG + `depth_encoding`) | FARM-Scenes, custom captures |
| `sens` | ScanNet `.sens` archive | Indoor ScanNet / ReferIt3D |
| `npz` | `.npz` with `images/depths/camtoworlds/K` | Habitat renders, custom |
| `rosbag` | Recorded `RGBDFrame` messages | Replay of online pipeline |

**YAML path (no StreamingMapper ROS init):**

```bash
python scripts/run_pipeline.py --config configs/replica.yaml
```

**LiDAR without depth images:** `scripts/lidar_bag_to_frames.py` projects `PointCloud2` into the camera and writes a `frames-json` scene (sqlite `.db3` bags; mcap not supported by that converter — see `DATA.md`).

**Important offline CLI flags**

| Flag | Effect |
|------|--------|
| `--stride` / `--start` / `--end` | Subsample frames |
| `--batch-size` | Frames per update (default 1) |
| `--save-path` | Output `scene_state.pt` |
| `--caption` | Async VLM captions (**needed for eval-parity retrieval**) |
| `--covisibility` | Build co-visibility graph |
| `--regions` | Multi-room region clustering |
| `--viser` | Live view on :8080 (~**6×** throughput cost) |
| `--keep-viser-after-run` | Leave viewer up after source ends |
| `--image-saving` | Copy frames into HDF5 (default: references only) |
| `--debug-trace-path` | Per-frame JSONL for `inspect_pipeline_trace.py` |
| `--extra-param KEY:=VALUE` | Pass-through StreamingMapper ROS params |

---

### A3. Shared per-batch pipeline

Both online and offline funnel into `StreamingMapper._run_mapping_batch`. Pure-function equivalents live in `pipeline/steps.py`.

```
1. _decode_batch          RGBDFrame → numpy / tensors (offline pre-decoded pass through)
2. _prepare_frames        colors, depths (resized to RGB size), intrinsics, world poses
3. _segment_batch         YOLOE masks → 3D Gaussians + DINOv3 features + voxels
4. _filter_segmentation   border / pixels / distance / label / IoU-duplicate filters
5. get_neighbors          feature cosine + Hellinger → candidate matches
6. find_object_correspondence   union-find (+ cannot-link blocks)
7. update_scene_graph_state     fuse Gaussians / voxels; create new objects
8. update_covisibility          adjacency for objects seen together
9. async captions               optional; does not block mapping much
10. prune                       deactivate walls / floors / clutter
11. snapshot                    serialize scene_state.pt
```

Default neighbor thresholds (`pipeline/steps.py`): `feature_sim_thresh=0.5`, `hellinger_thresh=0.8`.

---

### A4. Segmentation (YOLOE + DINOv3)

**Files:** `segmentation/yoloe.py`, `segmentation/dino.py`, `configs/yoloe_vocabulary.txt`

**YOLOE** (`yoloe-v8l` by default) is an open-vocabulary detector/segmenter. Prompts come from `configs/yoloe_vocabulary.txt` (~**1,516** class phrases spanning indoor and outdoor concepts).

For each mask:

1. Masked depth pixels unproject to world points (using pose + intrinsics).
2. Mahalanobis outlier rejection on the point set (`mahalanobis_thresh`, typical 2.0).
3. Fit a **3D Gaussian** (mean + packed `cov6` covariance).
4. Pool **DINOv3** features over the mask for cross-frame merging.
5. Accumulate a sparse **voxel** sample of supporting points (capped per object later).

**DINOv3 role:** merge / correspondence backbone only — not the open-vocab detector. `runtime_paths.resolve_dino_backbone()` auto-prefers gated `dinov3-vits16plus` if present, else bundled `dinov3-vits16`. Both are `hidden_size=384`.

**Replica YAML example knobs** (`configs/replica.yaml`):

```yaml
segmentation:
  model_id: "yoloe-v8l"
  vocab_file: "configs/yoloe_vocabulary.txt"
  imgsz: 640
  conf_thres: 0.25
  iou_thres: 0.5
  mask_erosion_px: 3
  mahalanobis_thresh: 2.0
```

---

### A5. Filtering

**Files:** `map_update/filtering.py`, orchestrator / StreamingMapper filter stage

Typical drops:

- detections touching the image border
- too few mask pixels
- too far from the camera
- uninformative labels (walls, floors, generic clutter — configurable)
- near-duplicate masks by IoU in the same frame

Surviving detections proceed to association.

---

### A6. Neighbor lookup and correspondence

**Files:** `map_update/get_neighbors.py`, `union_find.py`, `cannot_link.py`

**Neighbors:** For each new detection, find existing active objects whose:

- DINOv3 feature cosine similarity ≥ `feature_sim_thresh`, and
- Gaussian **Hellinger distance** ≤ `hellinger_thresh`

**Union-find:** Resolve which detections belong to the same world object (including chaining through neighbors). Returns winners / losers for fusion.

**Cannot-link:** If two objects were observed as **distinct detections in the same frame**, they must not be merged later — even if features look similar. This is a critical identity constraint for crowded scenes.

```
Same-frame detections A, B  →  cannot_link(A, B)
Later frames: union-find will refuse to join those lineages
```

---

### A7. Scene-state update, voxels, covisibility

**Files:** `map_update/object_update.py`, `covisibility.py`, `utils/geometry.py`

**Matched detections** fuse into the winning object’s Gaussian and sparse voxel cloud. **Unmatched** detections spawn new objects with fresh IDs.

**Voxel cloud (CSR-flat layout):**

- `object_voxel_keys_flat` / `object_voxel_keys_offsets` / `object_voxel_levels`
- Decoded at retrieval / viz time into tight AABBs (`voxel_cloud_aabb`, `voxel_keys_to_world`)
- Used by Viser boxes and by scoring / geometry helpers

**Covisibility:** Packed `u64` adjacency bitsets (default max ~1000 objects). When objects co-appear in a view (or camera neighborhood), edges are strengthened. Retrieval can optionally constrain target–anchor pairs to be within *K* hops (`covisibility_hops` on a `SpatialMethod`).

---

### A8. Captioning and embeddings

**Files:** `captioning/services.py` (`CaptionManager`), `worker.py`, `crop_util.py`, `structured.py`

When `--caption` / `caption_enabled:=true`:

1. Select a **best-view crop** per object (or update when a better view arrives).
2. Send crop asynchronously to **Qwen3.5-9B** via vLLM (`VLLM_BASE_URL`, default `:8000`).
3. Write back: caption text, category / attributes (structured fields), and embeddings:
   - caption text embedding (Qwen3-Embedding-0.6B, `:8002`)
   - SigLIP2 crop embedding (local)
   - Qwen3-VL-Embedding-2B crop embedding (`:8006`)

Captioning is async and **barely hurts mapping throughput**, unlike `--viser`.

**Eval note (`EVALUATION.md`):** caption embedding channels carry **most of the retrieval signal**. Reconstructing without `--caption` underperforms on paper benchmarks.

---

### A9. Pruning and regions

**Pruning** (`map_update/pruning.py`): periodically deactivate low-information objects (large planar surfaces, clutter labels) so the active pool stays queryable.

**Regions** (`regions/`, enabled with `--regions`): cluster objects into multi-room regions and optionally LLM-label them (`region_labels`, `region_centroids`, … on `SceneState`). Useful for “in the kitchen”-style `InRegion` predicates on house-scale scenes.

---

### A10. What `scene_state.pt` contains

**Schema:** `map_update/models.py` → `SceneState` (TypedDict; runtime is still a plain dict for compatibility).

**IO:** `scene_state_io.py` — PyTorch pickle. **Only open graphs you trust** (README security warning).

Core fields (conceptual):

| Group | Fields |
|-------|--------|
| Geometry | `means`, `cov6`, `count`, `active`, `object_id`, `class_ids` |
| Features | `features` (DINOv3 merge embeddings) |
| Identity | `id_redirect`, `loser_object_ids`, `cannot_link_object_ids` |
| Language | `object_caption`, category / attributes, caption + SigLIP2 + Qwen3-VL embeddings (+ histories) |
| Evidence | `object_image_ids`, `viewpoint_image_ids`, `object_mask_observations`, `images` |
| Voxels | `object_voxel_keys_flat`, `offsets`, `levels` |
| Graph | `covisibility_adj_u64`, weights, region_* |
| Runtime | `viewpoint_map`, `current_robot_position`, `is_locked` |

This file is the **entire spatial memory** — mapping writes it; retrieval and eval read it.

---

<a name="part-b--retrieval"></a>
## Part B — Retrieval: finding things with language

---

### B1. Two retrieval APIs

**1. Embedding-only retrieve** — good for simple “find a red toolbox”:

```python
from scene_graph.llm_utils import EmbedInterface
from scene_graph.retrieval.scene_graph_retriever import SceneGraphRetriever

retriever = SceneGraphRetriever.from_scene_state(
    "/data/out/warehouse.pt", embedder=EmbedInterface(verbose=False)
)
result = retriever.retrieve("a red toolbox on a workbench")
for cluster in result["clusters"]:
    print(cluster["cluster_score"], [c["object_id"] for c in cluster["candidate_objects"]])
```

**2. Full relational pipeline** (paper / Viser Query / `query_scene_graph.py`):

```
NL query
  → parse_query(llm) → QueryGraph(target_description, target_class, predicates[])
  → execute_spatial_query(...) → ranked ScoredCandidate list
```

Modules under `retrieval/spatial_reasoning/`:

| File | Role |
|------|------|
| `query_parser.py` | LLM → `QueryGraph` |
| `semantic_retrieval.py` | Multi-channel RRF candidate pool |
| `predicates.py` | Fast geometric (+ optional VLM) predicate scores |
| `executor.py` | Compose semantics + predicates |
| `joint_executor.py` | `joint_v1` interactive engine |
| `methods.py` | Named ablation / locked profiles |
| `models.py` | `Predicate`, `QueryGraph`, `ScoredCandidate`, … |
| `fuzzy_ops.py`, `calibration.py`, `prompts.py` | Soft logic, calibration, prompt text |

---

### B2. Query parsing

**File:** `retrieval/spatial_reasoning/query_parser.py`  
**Model:** Qwen3.5-9B via `LLMInterface`

`parse_query(query, llm)` returns a `QueryGraph`:

```python
@dataclass
class QueryGraph:
    target_description: str   # for embedding retrieval
    predicates: List[Predicate]
    reasoning: str = ""
    target_class: Optional[str] = None  # for soft class-match factor
```

**Valid predicate names:**

`Near`, `On`, `Above`, `Below`, `NextTo`, `Between`, `Inside`, `InRegion`, `HasAttribute`, `IsCategory`, `Closest`, `Farthest`, `LeftOf`, `RightOf`, `InFrontOf`, `Behind`

Post-parse fixup converts `Near` → `Closest` / `Farthest` when the raw query contains superlative wording.

If parsing fails or yields no predicates, callers may fall back to pure semantic lookup.

**Determinism:** Acc@1 / MRR in eval vary ±1–3 points across runs because of LLM sampling; Recall@10 and mean top-1 IoU are stable given fixed reconstructions.

---

### B3. Semantic retrieval (multi-channel RRF)

**File:** `semantic_retrieval.py`  
**Locked mode:** `retrieval_mode=multi`

Channels fused with Reciprocal Rank Fusion (RRF):

- caption-text embedding vs query
- caption-raw / related text channels
- SigLIP2 crop embedding
- Qwen3-VL crop embedding

`candidate_pool_mode=active` restricts to active (non-pruned) objects. `geometry_mode=alias_expand` expands geometric aliases for scoring. Eval keeps top-100 ranked predictions before scoring top-10.

---

### B4. Spatial predicates

**File:** `predicates.py` (`PredicateEvaluator`)

**Fast path:** metric / soft geometric scores from Gaussians, AABBs, shared viewpoints, vertical-axis inference (ScanNet Z-up vs HM3D Y-up via `_infer_vertical_axis`).

Examples:

- `Near` / `NextTo` — soft proximity
- `Above` / `Below` — along inferred up-axis
- `LeftOf` / `RightOf` / `InFrontOf` / `Behind` — **view-dependent**, using stored shared image views (image-plane / pose-projected), not a single global “left”
- `Closest` / `Farthest` — superlatives over the candidate set
- `Between`, `Inside`, `InRegion`, `HasAttribute`, `IsCategory` — geometric or semantic

**VLM path:** optional prompts when the fast path is undecidable or uncertain. The **locked paper method sets `force_no_vlm=True`**, so eval numbers are pure fast-path + soft composition.

---

### B5. Method profiles

**File:** `methods.py` — `SPATIAL_METHODS` registry

| Name | Role |
|------|------|
| **`unified_soft_w50`** | **Paper-locked** (2026-05-17): soft predicates, `predicate_weight=0.50`, `class_mismatch_floor=0.3`, `force_no_vlm=True` |
| `joint_v1` | Interactive Viser default: truncation-free anchors, vectorized predicates |
| `semantic_only` | Ignore spatial predicates |
| `hard_predicates` / `soft_predicates*` | Ablations over weight / hardness |
| `current` | Soft predicates + optional gated VLM |

Eval scripts default to `unified_soft_w50`. Set `VISER_SPATIAL_METHOD=unified_soft_w50` to force the viewer onto the paper protocol.

Soft class factor: if the candidate’s YOLOE category / caption does not match `target_class`, multiply similarity by `class_mismatch_floor` (0.3) instead of hard-pruning — important for paraphrased queries (“the broken one”).

---

### B6. Viser Query panel and CLI

**Viewer:** `scripts/view_scene_state.py` → `visualization/viser_visualizer.py`

- Serves object boxes, voxel evidence, captions on click, optional accumulated cloud
- **Query panel:** start vLLM backends, type a referential expression, color-code target (gold) / anchors (blue) / distractors, fly camera to top match
- Port **8080** (host networking in compose)

**CLI:** `scripts/query_scene_graph.py --pt … --query "…"`

---

<a name="part-c--online-ros-2-path"></a>
## Part C — Online ROS 2 path

**Packages:** `ros/mapping` (nodes + launch), `ros/msgs` (`mapping_msgs`)

### How frames enter

1. Each camera has a `frame_pub` that synchronizes RGB + depth + `camera_info` + TF → publishes `RGBDFrame` on `/mapping/rgbd_frame/<cam>`.
2. Topic wiring comes from `src/scene_graph/camera_config.py` (`CAMERA_CONFIG`).
3. Single `streaming_mapper` node batches by `camera_names` / `expected_batch` and runs `_run_mapping_batch`.

### Launch files (`ros/mapping/launch/`)

| Launch | Purpose |
|--------|---------|
| `scenegraph_validation_exploration.launch.py` | Default validation / exploration mapping |
| `mapping_five_cam.launch.py` | 5-camera Spot rig |
| `mapping_odin1.launch.py` | Fisheye + LiDAR: `odin1_depth_pub` projects cloud → synthetic depth, then standard RGBD topics |
| `visual_search_yoloe_text_prompt.launch.py` | Text-prompted YOLOE visual search demo |

### Convenience

```bash
./run.sh ros2 caption_enabled:=true
# starts vLLM + ros2 launch …
```

**Registered executables** (`ros/mapping/setup.py`): `streaming_mapper`, `frame_pub`, `odin1_depth_pub`, `visual_search_yoloe_text_prompt`.  
`tf_listener.py` is a helper import for `frame_pub`, not a standalone executable.

**Msgs** include: `RGBDFrame`, `FrameMetadata`, `DetectedObject(s)`, `LocalCaption(Array)`, `SceneGraphSnapshotSimple`, `VisualSearchResult(Array)`, plus `Siglip2TextEmbed.srv`.

**Note:** `streaming_mapper.py` uses **CRLF** line endings — keep them to avoid whole-file diffs.

---

<a name="part-d--evaluation-harness"></a>
## Part D — Evaluation harness

**Guide of record:** `EVALUATION.md`  
**Code:** `src/scene_graph/eval/` + `scripts/eval_*.py`

### Locked protocol (every headline number)

**Predict:** `parse_query` → `execute_spatial_query` with:

- `spatial_method=unified_soft_w50`
- `retrieval_mode=multi` (RRF)
- `candidate_pool_mode=active`
- `pre_filter_k=-1`
- no VLM rerank
- `geometry_mode=alias_expand`
- top-100 predictions kept

**Score (top-10):** Acc@1@IoU ∈ {0.1, 0.25, 0.5}, Recall@K ∈ {1,3,5,10}, MRR, mean top-1 IoU

| Benchmark | IoU type | Out of the box? |
|-----------|----------|-----------------|
| **FARM-Scenes** | 3D-AABB | Yes (public HF dataset + prebuilt graphs) |
| **ReferIt3D (ScanNet)** | Projected **visible-mask** IoU | After ScanNet ToS data |
| **IRef-VLA (HM3D)** | Projected **visible-mask** IoU | After HM3D ToS + host habitat-sim render |

Visible-mask scoring: `eval/visible_mask.py` + `view_selection.py` (`v1_largest_mask`, depth tolerance 0.15 m).

### Expected ballpark numbers (from `EVALUATION.md`)

**FARM-Scenes (3D-AABB):**

| split | utterances | acc@1@0.25 | recall@10@0.25 | MRR@0.25 |
|-------|-----------:|-----------:|---------------:|---------:|
| odin1 | 283 | 0.106 | 0.304 | 0.175 |
| grandtour | 598 | 0.099 | 0.333 | 0.167 |

**ReferIt3D 30-scene headline:** acc@1@mask-0.25 ≈ **0.256**  
**IRef-VLA 30-scene headline:** acc@1@mask-0.25 ≈ **0.051**

Absolute Acc@1 is modest — hard open-vocabulary 3D grounding on large scenes, not closed-set detection.

### Script map

```
Reconstruct          Predict                         Score
─────────────        ───────                         ─────
offline.run /        eval_farm_scenes.py             (same, --phase score)
run_scene_graph_*    eval_referit3d_spatial.py       convert_ours_to_canonical.py
                     eval_iref_vla.py                 → eval_predictions.py
                                                     score_largescale_predictions.py
```

Curated UID lists: `benchmarks/curated_utterances/` (`scannet_30.json`, `iref_vla_hm3d_30.json`, 5-scene parity slices, …).

### Unit tests

```bash
./scripts/in_docker.sh python -m pytest tests/
```

Covers spatial predicates and covisibility constraints. Broader coverage is not comprehensive — verify mapping changes with a short offline run (`--end 50` or synthetic NPZ).

---

## 9. Configuration and environment reference

### `run.sh`

| Command | Description |
|---------|-------------|
| `./run.sh build` | Build `scene_graph:latest` |
| `./run.sh shell [<dir>]` | Container shell; mount data at `/data` |
| `./run.sh vllm` | Start three vLLM servers in tmux |
| `./run.sh ros2 [args…]` | vLLM + online mapping launch |
| `./run.sh stop` | Stop vLLM tmux session |

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `SCENE_GRAPH_MODEL_DIR` | `./models` | Checkpoints root |
| `SCENE_GRAPH_MAPPING_DATA_DIR` | `~/.ros/scene_graph/mapping` | Default save dir |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | Caption + query parse (Qwen3.5-9B) |
| `VLLM_EMBED_BASE_URL` | `…:8002/v1` | Qwen3-Embedding-0.6B |
| `VLLM_QWEN3_VL_EMBED_BASE_URL` | `…:8006/v1` | Qwen3-VL-Embedding-2B |
| `GPU_VL8` / `GPU_EMB` / `GPU_VL_EMB` | `0` / `1` / `1` | GPU assignment per server |
| `LAM_SIGLIP2_CKPT` | (bootstrap path) | Local SigLIP2 weights |
| `VISER_SPATIAL_METHOD` | (`joint_v1` in UI) | Override interactive spatial method |
| `QWEN3_VL_EMBED_ENABLED` | — | Toggle VL embedding channel |

### Two Python installs (footgun)

1. Root `pyproject.toml` → `scene_graph` in the container venv  
2. `ros/mapping/setup.py` → colcon-built `mapping.*`  

Offline `run.py` prefers the **checked-out** `ros/mapping` source on `sys.path` so a stale colcon install under `/tmp/colcon_ws` does not silently win. Do **not** put a `package.xml` at `ros/` root — colcon would stop recursing and miss both packages.

---

## 10. Models, licenses, and compute

| Model | Role | How obtained | License note |
|-------|------|--------------|--------------|
| YOLOE v8l | Open-vocab detect/segment | `bootstrap_models.sh` + `third_party/yoloe` | **AGPL-3.0** (drives repo license) |
| MobileCLIP | YOLOE dependency | bootstrap / Apple CDN | Apple terms |
| DINOv3 ViT-S/16 | Merge features | **Committed** in `models/` | Meta DINOv3 license (in-repo) |
| DINOv3 ViT-S+/16 | Paper merge backbone | Gated HF download | Meta + HF access |
| SigLIP2 | Retrieval channel | `scripts/download_siglip2.py` | Google |
| Qwen3.5-9B | Caption + query parse | vLLM pull | Qwen license |
| Qwen3-Embedding / VL-Embed | Retrieval channels | vLLM | Qwen license |

Details: `THIRD_PARTY_NOTICES.md`. **Review before commercial use.**

**Compute reality:** full stack ≈ 50 GB VRAM; largest scenes ≈ 65 GB; disk ≈ 26 GB.

---

## 11. Data shapes quick reference

### Frame (pre-decoded)

```python
{
  "camera": "cam0",
  "rgb": np.ndarray,          # H,W,3 uint8
  "depth_f32": np.ndarray,    # H,W float32 metres
  "T_world_cam": np.ndarray,  # 4x4
  "rgb_instrinsics": ...,     # note spelling
  "depth_instrinsics": ...,
  "stamp_ns": int,
  "frame_id": str,
  "received_time": float,
}
```

### `QueryGraph` (parsed query)

```json
{
  "target_description": "yellow ladder",
  "target_class": "ladder",
  "predicates": [
    {"name": "Near", "args": ["shelves"], "kwargs": {}}
  ],
  "reasoning": "..."
}
```

### `ScoredCandidate` (retrieval hit)

```python
ScoredCandidate(
  object_index, object_id,
  predicate_results=[PredicateResult(name, score, status), ...],
  composite_score,
  matched_anchors={"shelves": anchor_object_id},
  target_similarity=0.42,
)
```

### NPZ chunk keys

`images (N,H,W,3)`, `depths (N,H,W) float32 m`, `camtoworlds (N,4,4)`, `K (3,3)`, optional `pose_convention` (`opengl`|`opencv`), `nominal_hz`.

### Eval metrics JSON (conceptual)

```json
{
  "acc_at_1@iou=0.25": 0.106,
  "recall_at_10@iou=0.25": 0.304,
  "mrr@iou=0.25": 0.175,
  "mean_top1_iou": 0.070
}
```

---

## 12. Troubleshooting

| Issue | Likely cause | Fix |
|-------|--------------|-----|
| Docker build fails on YOLOE | Submodule missing | `git submodule update --init --recursive` |
| Mapping runs but retrieval is weak | No captions | Re-run with `--caption` and `./run.sh vllm` |
| ScanNet objects over-fragmented | Bundled DINOv3 ViT-S/16 | Install gated `dinov3-vits16plus` |
| Viser feels extremely slow | Live streaming cost | Drop `--viser` for real runs (~6×) |
| Acc@1 jitter across eval runs | LLM parser sampling | Expected ±1–3 pts; trust Recall@10 |
| Offline ignores your mapper code edits | Stale colcon `mapping` | `offline/run.py` path hack; or rebuild colcon |
| OOM / won’t fit | Full stack VRAM | Split GPUs via `GPU_*`; don’t run viser+all servers+map on one 24 GB card |
| `scene_state.pt` won’t load / unsafe | Pickle | Only load trusted graphs |
| HM3D eval blocked | habitat-sim not in Docker | Render on host per `eval/iref_vla/README.md` |
| LiDAR converter fails on mcap | Tool limitation | Convert to sqlite `.db3` or write frames-json another way |
| Host `pip install -e .` | Unsupported | Use `./run.sh shell` |

---

## 13. Where to change what

| If you want to… | Touch… |
|-----------------|--------|
| Change association thresholds | `pipeline/steps.py` defaults; StreamingMapper params / `--extra-param` |
| Change open-vocab classes | `configs/yoloe_vocabulary.txt` |
| Change Gaussian / voxel fusion | `map_update/object_update.py` |
| Change cannot-link / merge rules | `cannot_link.py`, `union_find.py` |
| Change predicate geometry | `retrieval/spatial_reasoning/predicates.py` |
| Add a spatial method ablation | `methods.py` (`SPATIAL_METHODS`) |
| Change query parse prompt | `retrieval/spatial_reasoning/prompts.py` |
| Change RRF / embedding channels | `semantic_retrieval.py`, `llm_utils/` |
| Add a dataset format | New `offline/frame_sources/*.py` + wire in `offline/run.py` |
| Change ROS camera topics | `camera_config.py` + launch files |
| Change eval protocol / metrics | `EVALUATION.md` + `eval/unified_scoring.py` / bench runners |
| Debug a single frame’s decisions | `--debug-trace-path` + `scripts/inspect_pipeline_trace.py` |
| Change Viser UX / Query panel | `visualization/viser_visualizer.py` |

---

## Mental model (one paragraph)

FARM turns a **posed RGB-D stream** into a **persistent object database** (`scene_state.pt`) by detecting open-vocab masks (YOLOE), lifting them to 3D Gaussians, merging them with DINOv3 + Hellinger + union-find under cannot-link constraints, and enriching them with captions and embeddings. Language queries are decomposed by an LLM into a **target + spatial predicates**, candidates are ranked by **multi-modal embedding fusion**, and predicates are scored with **fast geometric (and optionally VLM) checks**. Online ROS and offline datasets are the same mapper with different plumbing. Evaluation is first-class: locked Acc@IoU / Recall@K / MRR on FARM-Scenes, ReferIt3D, and IRef-VLA.

For format contracts and bring-your-own-data recipes, prefer **`DATA.md`**. For reproducing paper numbers, prefer **`EVALUATION.md`**. For day-to-day agent/maintainer shortcuts, prefer **`CLAUDE.md`**.
