# COLMAP + DA3 depth pipeline

Equirect video or frames → panorama SfM (COLMAP) → DA3 face depths → point cloud.

| Role | Owns |
|------|------|
| **COLMAP panorama SfM** | Relative face poses, intrinsics, sparse structure. |
| **DA3** | Dense per-face depth (metric or relative / pose-conditioned). |
| **Scale `alpha`** | Optional: `d ≈ alpha · d_sparse` scales COLMAP `t` (skip with `scale_align.mode: none`). |

## Quick start

```bash
./colmap_depth_pipeline/scripts/run_e2e.sh /path/to/clip.mp4 /path/to/run_out
./colmap_depth_pipeline/scripts/run_e2e.sh /path/to/frames /path/to/run_out --max-frames 100

# Reuse SfM, re-run depth only
./colmap_depth_pipeline/scripts/run_e2e.sh clip.mp4 out --skip-sfm --window-size 4
```

Depth-only (existing SfM):

```bash
python colmap_depth_pipeline/scripts/run_pipeline.py \
  --colmap_dir /path/to/run_out/sfm \
  --pano_dir /path/to/run_out/frames \
  --out_dir /path/to/run_out/depth \
  --export_ply
```

Re-export PLY without re-running DA3:

```bash
python colmap_depth_pipeline/scripts/export_pointcloud.py \
  --colmap_dir run_out/sfm --out_dir run_out/depth
```

## Layout

```
run_out/
  frames/                 equirect frames
  sfm/                    COLMAP panorama project
  depth/
    face_depth/face{id}/  DA3 depths (*.npy, *_raw.npy)
    face_conf/face{id}/
    face_sky/face{id}/
    face_depth_vis/face{id}/
    pointcloud.ply       COLMAP K + poses unproject
    manifest.json
```

## Config

- `configs/default.yaml` — DA3METRIC + metric alpha
- `configs/recon.yaml` — pose-conditioned DA3-LARGE, relative units

## Notes

- Requires `ffmpeg` on `PATH` for video extract. SfM uses the Docker image in config.
- Sparse depth for scale fit uses observed COLMAP keypoints per face.
- Overlapping DA3 windows blend by confidence (higher conf wins).
