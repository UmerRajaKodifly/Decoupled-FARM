# What “FARM vs greedy IoU object counts” means

Short version: **same masks, two identity rules, two integers.** It is a sanity
check on association, not a benchmark score.

## Setup

1. Run YOLOE once per frame (FARM `YOLOESegmenter`, construction-site vocab).
2. Feed those detections into **both**:
   - `--association farm` → FARM `get_neighbors` + union-find + cannot-link
   - `--association greedy_iou` (or the parallel IoU tracker inside the farm run)
3. After the sequence:
   - \(N_{\mathrm{farm}}\) = active objects in FARM scene state
   - \(N_{\mathrm{iou}}\) = distinct IoU track IDs ever created

The farm runtime already counts IoU tracks in parallel during a farm run
(`n_iou_tracks` in `summary.json`), so one mapping pass is enough.

## How to read the numbers

| Pattern | Likely meaning |
|---|---|
| \(N_{\mathrm{iou}} \gg N_{\mathrm{farm}}\) | 2D tracks fragment (orbit, cubemap face changes, occlusion). FARM 3D+DINO is merging them. **This is the hoped-for signal.** |
| \(N_{\mathrm{farm}} \approx N_{\mathrm{iou}}\) | Little multi-view linking — short clip, tiny motion, or both rules under-merging. |
| \(N_{\mathrm{farm}} \gg N_{\mathrm{iou}}\) | FARM is failing to associate (scale wrong, Gaussians degenerate, DINO off / wrong backbone, Hellinger always high). |

## What it is not

- Not mAP / mask IoU vs ground truth.
- Not “FARM detected more objects” in the YOLOE sense — detection set is shared.
- Not runnable without metric-consistent depth. Hellinger² uses 3D means/covs.
  Dummy constant depth or unscaled SfM depth makes \(N_{\mathrm{farm}}\)
  meaningless.

## Why cubemap makes this comparison more interesting

A crane that spans two 90° faces in one timestamp, or moves from `front` to
`right` on the next timestamp, will look like **two 2D tracks** to greedy IoU
(different images, maybe non-overlapping boxes). FARM should still merge them
if world Gaussians + DINO agree. That gap is exactly \(N_{\mathrm{iou}} -
N_{\mathrm{farm}}\).
