# COLMAP-MVS alternate depth flow

Primary depth remains **`dl` / `dl_depth_v1`**. This document is the explicit
comparison path only. Default `--depth-source` is `dl` everywhere.

## Switch

```bash
python -m farm_object_map e2e --video CLIP.mp4 --outputs-root outputs --depth-source colmap_mvs --gpu-index 0
python -m farm_object_map e2e --video CLIP.mp4 --outputs-root outputs --depth-source dl
```

Layout:

```
outputs/<video_stem>/dl/
outputs/<video_stem>/colmap_mvs/
```

Downstream mapping is unchanged: YOLOE mask → `DepthSource.depth_for_frame` →
unproject → `T_world_cam` → Gaussian / voxels + FARM association.

## Test run (2026-08-05)

- **Video:** `test-videos/export_video(2).mp4` (Insta360 ONE R equirect, 3840×1920, 171 s)
- **Subset:** first **8 frames @ 2 fps** (≈4 s), panorama cubemap
  `cubemap-nosfm-top-and-bottom` → 48 registered faces (top/bottom masked in SfM;
  32 equatorial views densified)
- **Work dir:** `outputs/export_video_2/colmap_mvs/`
- **DL side-by-side:** **not run** — `dl_depth_v1` still not deployable

### GPU evidence (this run, not Phase 0)

**COLMAP binary (dense stereo argv0):**
`/home/kodifly/tools/colmap-4.1.0/bin/colmap`

```
COLMAP 4.1.0 (Commit fa8e3b3 on 2026-06-26 with CUDA)
```

**Bundle adjustment — device index 0, Caspar selected:**

```
utils.colmap_sfm INFO ✓ Incremental SfM bundle adjustment: CASPAR global BA + GPU Ceres local BA (gpu_index=0)
utils.colmap_sfm INFO Camera model(s) ['PINHOLE'] are CASPAR-compatible; using CASPAR for global BA.
```

No Ceres-fallback warning. Log: `outputs/export_video_2/colmap_mvs/sfm/logs/bundle_adjustment.log`

**Dense stereo — `--PatchMatchStereo.gpu_index 0`, CUDA sweeps, no CPU fallback:**

```
I20260805 13:04:44.850654 ... patch_match_options.cc:48] gpu_index: 0
I20260805 13:04:45.034359 ... cudacc.cc:51] Initialization: 0.0629s
I20260805 13:04:45.487281 ... cudacc.cc:51]  Sweep 1: 0.4528s
```

`nvidia-smi -L`: `GPU 0: NVIDIA GeForce RTX 3070`. Fallback warnings: none.
Log: `outputs/export_video_2/colmap_mvs/dense/logs/patch_match_stereo.log`

### Depth contract

32 `DepthMap` npz files under `depth_npz/` with `units="sfm"`, `source="colmap_mvs"`.

### Mapping (FARM association)

32 / 48 undistorted views had MVS depth (16 top/bottom rig faces skipped — no
`geometric.bin`). Mapping used only those 32.

| Metric | colmap_mvs | dl |
|---|---|---|
| Object count | **13** | pending |
| Labels | utility shed×3, precast×2, rebar cage, skid steer, excavated ground, perimeter wall, hoarding, aggregate stockpile, mobile crane, foundation rebar | — |
| Mean voxels / object | **592** | — |
| IoU-track count (parallel baseline) | 0 on this short clip | — |

FARM still emitted Gaussians/voxels; the wrapper’s `seg["n"]` drop-log stayed 0
(key is not “valid depth pixels”). Use voxel counts until that FARM field is
clarified.

Same-object Gaussian mean/cov diff vs DL is **pending** until `dl_depth_v1` lands.
Use `python -m farm_object_map compare-flows --dl-summary ... --mvs-summary ...`.
