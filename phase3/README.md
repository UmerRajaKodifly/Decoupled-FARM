# Phase 3 — Associate / Fuse / Map

Reads Phase 2 detection packs in keyframe order, filters detections, resolves
correspondences via union-find, and fuses them into a growing global
`SceneState`.  Output is a single `scene_state.pt` that becomes the input to
Phase 4 (captioning).

---

## Role in the pipeline

```
Phase 2 packs          Phase 3                  Phase 4
(detections_kf*.pt) ──► filter → associate    ──► captions / VLM
                             ↓ resolve              embeddings
                         update + covis
                             ↓
                       scene_state.pt
```

Phase 3 is a FARM-faithful implementation.  It calls FARM's:

| FARM function | What it does |
|---|---|
| `filtering.py::normalize_seg_outputs` | dtype/device coercion |
| `filtering.py::filter_detections_by_num_pixels` | drop tiny masks |
| `filtering.py::filter_detections_touching_image_border` | drop border slivers |
| `filtering.py::filter_detections_by_distance` | drop out-of-range objects |
| `filtering.py::filter_detections_duplicates_iou` | per-face IoU dedup |
| `steps.py::find_neighbors_for_detections` | cosine + Hellinger neighbor search |
| `steps.py::resolve_correspondence` | union-find + cannot-link enforcement |
| `object_update.py::update_scene_graph_state` | Gaussian / voxel merge |
| `covisibility.py::update_covisibility_from_visible_indices` | kNN covis edges |
| `cannot_link.py::add_same_frame_cannot_links_from_detection_assignments` | post-update cannot-links |

---

## Inputs

| Source | Path | Description |
|---|---|---|
| Phase 2 packs | `phase2-detect-segment-embed/output/detections_kf*.pt` | One file per keyframe; contains `means`, `cov6`, `features`, `masks`, `scores`, `class_ids`, `batch_ids`, `num_pixels`, `boxes_xyxy`, `det_points_flat`, `det_points_offsets`, `poses_world`, `intrinsics`, `face_meta`, `kf_id`, `vocab` |

Each pack covers all 4 cubemap faces of one keyframe.

---

## Directory layout

```
phase3-associate-fuse-map/
├── README.md           ← this file
├── filter.py           ← face-aware border + distance + pixel + IoU filtering
├── associate.py        ← neighbor lookup + union-find correspondence
├── update.py           ← Gaussian/voxel/covis fusion into SceneState
├── run_phase3.py       ← main loop + CLI + checkpointing
├── validate_phase3.py  ← metrics, plots, PASS/WARN/FAIL gate
└── output/
    ├── scene_state.pt              ← final product
    ├── run_stats.json              ← per-keyframe counters
    ├── scene_state_ckpt_kf*.pt     ← intermediate checkpoints (every 50 kf)
    └── validation/
        ├── object_count_growth.png
        ├── merge_rate.png
        ├── world_xy_scatter.png
        ├── overlays_3d.html
        ├── class_breakdown.png
        ├── feature_consistency.png
        ├── metrics.json
        └── summary.txt
```

---

## 360-specific adaptations

| Concern | FARM default | This implementation |
|---|---|---|
| Border filter | `min_kept_num_pixels=4000` | `min_kept_num_pixels=1000` — cube seams legitimately clip large objects |
| `detection_image_ids` | per-camera image id | `global_id = kf_index * 4 + face_index` — cannot-link is per face; two faces may independently match the same world object |

---

## Running Phase 3

```bash
conda activate farm-phase2

cd /home/kodifly/Desktop/farm-git/pipeline/phase3-associate-fuse-map

python run_phase3.py \
  --det-dir ../phase2-detect-segment-embed/output \
  --output-dir ./output \
  --device cuda
```

All thresholds have sane defaults but are tunable:

```
  --feature-sim-thresh 0.5   cosine similarity floor
  --hellinger-thresh   0.8   Hellinger distance ceiling
  --max-merge-dist     1.0   world-space merge kill distance (m)
  --min-pixels         50    minimum mask pixels
  --min-dist           0.3   minimum detection distance from camera (m)
  --max-dist          80.0   maximum detection distance from camera (m)
  --checkpoint-every   50    save an intermediate ckpt every N keyframes
```

Expected runtime: ~1–4 s/keyframe on a single GPU for a 360-frame sequence,
depending on the number of detected objects.

---

## Validating Phase 3 outputs

After `scene_state.pt` is written:

```bash
python validate_phase3.py \
  --output-dir ./output \
  --vocab-file ../phase2-detect-segment-embed/vocab/construction_vocab.txt
```

Check `output/validation/summary.txt` for the PASS/WARN/FAIL gate report.

### Gate to Phase 4

| Check | Criterion |
|---|---|
| `merge_occurred` | Active objects < 95% of total Phase 2 detections |
| `object_count` | Active objects ≤ 2000 |
| `merge_activity` | At least one merge occurred in some keyframe |
| `scatter_coherence` | World XY footprint < 500 m (no depth scale blowup) |

Any FAIL blocks Phase 4.  WARNs are informational.

---

## Phase 4 contract

Phase 4 (captioning) requires:

| Field | Location in `scene_state.pt` |
|---|---|
| Object means (world) | `state["means"]` — `(N, 3)` float32 |
| Object Gaussians | `state["cov6"]` — `(N, 6)` float32 |
| DINOv3 features | `state["features"]` — `(N, 384)` float32 |
| Class IDs | `state["class_ids"]` — `(N,)` int64 |
| Active flags | `state["active"]` — `(N,)` bool |
| Covisibility graph | `state["covisibility_adj"]` — bitmask tensor |
| Vocabulary | `state["vocab"]` — list[str] (written from pack) |

Phase 4 will load this file directly and enqueue best-view crops from the
stored frame references for captioning by a VLM.
