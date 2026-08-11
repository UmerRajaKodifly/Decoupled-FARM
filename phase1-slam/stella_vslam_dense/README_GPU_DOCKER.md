# Stella VSLAM Dense - GPU Docker Run Guide

This guide provides step-by-step instructions and example commands to build, start, and run **Stella VSLAM Dense** with full GPU (CUDA) acceleration inside Docker. It covers running on **monocular equirectangular (360°)** inputs from either a **video file** or a **directory of frames/images**.

---

## 📋 Prerequisites

Before starting, ensure your host machine has:
1. **NVIDIA GPU Driver** installed.
2. **Docker** installed.
3. **NVIDIA Container Toolkit** installed (to pass GPU capability to Docker containers).
   - If not installed, you can run:
     ```bash
     ./scripts/ubuntu/install_nvidia_docker.sh
     ```

---

## 🛠️ Step 1: Build the Docker Image

Build the Docker image with the interactive web viewer (Viser) support:

```bash
docker build -t stella_vslam_dense -f Dockerfile.viser .
```

---

## 🚀 Step 2: Start the GPU Docker Container

To run with CUDA acceleration, you must start the container with `--gpus all`. 

> 💡 **CRITICAL FIX:** If your host Nvidia driver is newer than the container's CUDA version constraints, the container may fail to initialize CUDA (raising `ScorePlaneDepth no CUDA-capable device is detected`). You can bypass this check by adding the environment variable `-e NVIDIA_DISABLE_REQUIRE=true`.

### Option A: Start Interactively (Foreground)
Perfect for testing and running commands manually inside the container:
```bash
docker run -it --rm \
  --gpus all \
  -e NVIDIA_DISABLE_REQUIRE=true \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8080:8080 \
  --name=stella_vslam_dense \
  -v /home/kodifly/stella_data/:/data \
  -v /home/kodifly/stella_vslam_dense:/stella \
  stella_vslam_dense
```

### Option B: Start as a Background Daemon (Detached)
Recommended for long-running scripts or scripting pipelines. This mounts the workspace code directory directly to `/stella` so any edits to config/YAML or scripts on your host are immediately reflected inside the container:
```bash
docker run -d \
  --gpus all \
  -e NVIDIA_DISABLE_REQUIRE=true \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8080:8080 \
  --name=stella_vslam_dense \
  -v /home/kodifly/stella_data/:/data \
  -v /home/kodifly/stella_vslam_dense:/stella \
  stella_vslam_dense \
  tail -f /dev/null
```

---

## 🎞️ Step 3: Run Stella VSLAM Dense

Put your dataset (videos/frames) and vocabulary files into the mapped directory on your host (e.g. `/home/kodifly/stella_data/` which appears as `/data/` inside the container).

Download the vocabulary file if you haven't already:
```bash
wget -L "https://github.com/stella-cv/FBoW_orb_vocab/raw/main/orb_vocab.fbow" -O /home/kodifly/stella_data/orb_vocab.fbow
```

---

### 🎥 Mode 1: Running on an Input Video File

Run SLAM on an equirectangular video file. 

```bash
# If running inside Option A (interactive terminal):
./run_video_slam.py \
  -v /data/orb_vocab.fbow \
  -c /stella/example/dense/dense_hd.yaml \
  -m /data/export_video.mp4 \
  --frame-step 1 \
  -o /data/out.db \
  -p /data/out.ply \
  -k /data/keyframes/ \
  --eval-log-dir /data/traj \
  --auto-term

# If running from Host using Option B (detached daemon):
docker exec -it stella_vslam_dense \
  python3 /stella/tools/run_video_slam.py \
  -v /data/orb_vocab.fbow \
  -c /stella/example/dense/dense_hd.yaml \
  -m /data/export_video.mp4 \
  --frame-step 1 \
  -o /data/out.db \
  -p /data/out.ply \
  -k /data/keyframes/ \
  --eval-log-dir /data/traj \
  --auto-term
```

---

### 📂 Mode 2: Running on a Directory of Frames (Image Sequence)

Our custom wrapper in `run_video_slam.py` natively supports passing a **directory containing sorted image files** (e.g. `.png`, `.jpg`, `.jpeg`). It automatically processes them sequentially.

```bash
# If running inside Option A (interactive terminal):
./run_video_slam.py \
  -v /data/orb_vocab.fbow \
  -c /stella/example/dense/dense_hd.yaml \
  -m /data/my_frame_directory/ \
  --frame-step 1 \
  -o /data/out.db \
  -p /data/out.ply \
  -k /data/keyframes/ \
  --eval-log-dir /data/traj \
  --auto-term

# If running from Host using Option B (detached daemon):
docker exec -it stella_vslam_dense \
  python3 /stella/tools/run_video_slam.py \
  -v /data/orb_vocab.fbow \
  -c /stella/example/dense/dense_hd.yaml \
  -m /data/my_frame_directory/ \
  --frame-step 1 \
  -o /data/out.db \
  -p /data/out.ply \
  -k /data/keyframes/ \
  --eval-log-dir /data/traj \
  --auto-term
```

---

## 🎛️ Key Tuning Arguments

| Argument | Description | Recommendation |
|---|---|---|
| `--frame-step <N>` | Process every N-th frame. | Set to `1` for maximum density/quality; `2` or `3` for speed. |
| `--auto-term` | Automatically shutdown and save outputs when the video ends. | Highly recommended for non-interactive/headless runs. |
| `--disable-viewer` | Run headless without launching the Web UI. | Reduces CPU/RAM overhead on remote servers. |
| `--eval-log-dir <path>` | Directory to save camera poses and tracking times. | Automatically creates the directory and writes TUM trajectory files. |

---

## 💻 Monitoring & Viewer

Open your browser and navigate to:
🔗 **`http://localhost:8080`**

You will see a live 3D visualizer showing:
- Real-time video/frame feed and extracted features.
- Monocular camera path & frustum trajectory.
- Sparse landmarks (yellow/green) and **dense PatchMatch 3D points** generated on-the-fly.

---

## 💾 Where are the Outputs Saved?

Once the run completes (or you press **Terminate SLAM** in the viewer), the following files are saved in `/home/kodifly/stella_data/`:

1. **Dense 3D Point Cloud (`out.ply`)**:
   - High-fidelity point cloud containing reconstructed coordinates and RGB colors.
2. **Camera Poses (`traj/keyframe_trajectory.txt` & `traj/frame_trajectory.txt`)**:
   - Contains 3D translations and quaternion rotations for each frame in standard **TUM format**:
     `timestamp tx ty tz qx qy qz qw`
3. **Map Database (`out.db`)**:
   - SQLite3 database containing keyframe nodes, 3D points, and graph covisibility data.
4. **Keyframe Images (`keyframes/`)**:
   - Folder containing keyframe images.

---

## 🧹 Post-Processing: Denoising Point Clouds

To clean up noise, floating artifacts, or outliers from the generated point cloud:

```bash
python3 denoise_pcd.py /home/kodifly/stella_data/out.ply /home/kodifly/stella_data/out_clean.ply
```
*(Uses statistical outlier removal to filter out points with distance standard deviation greater than `3.5` using `30` nearest neighbors).*
