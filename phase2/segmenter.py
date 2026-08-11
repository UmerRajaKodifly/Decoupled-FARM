"""YOLOE open-vocabulary instance segmenter for Phase 2.

Thin wrapper around FARM's YOLOESegmenter that:
  - wires FARM's model weights (yoloe-v8l-seg.pt / yoloe-v8l-seg-pf.pt)
  - injects the construction-site vocabulary (vocab/construction_vocab.txt)
  - forces DINOv3 features (use_dino_features=True), matching FARM's live path
  - exposes a call signature identical to FARM's YOLOESegmenter so that
    Phase 3 can consume the output dict unchanged

Why YOLOE prompt-free?
----------------------
YOLOE's prompt-free (pf) checkpoint has the vocabulary fused directly into the
model heads at build time via MobileCLIP text embeddings.  At inference no text
encoder runs, so detection is fast and fully deterministic.  The base checkpoint
is needed only once at init to extract the vocab embeddings.

Output dict schema (identical to FARM)
---------------------------------------
  means       (M, 3)  float32  world frame (after transform_to_world is applied
                               by the runner — segmenter itself emits camera frame)
  cov6        (M, 6)  float32  packed 3×3 covariance
  features    (M, D)  float32  DINOv3 mask-pooled, L2-normalised (D=384)
  masks       list of (H_i, W_i) bool tensors, aligned to original images
  scores      (M,)    float32  YOLOE confidence
  class_ids   (M,)    int64    index into construction_vocab.txt
  labels      list[str]        class name per detection
  batch_ids   (M,)    int64    which face in the input batch
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import torch

# ---------------------------------------------------------------------------
# Resolve FARM paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
_COMMON = _HERE.parent.parent / "common"
if _COMMON.is_dir() and str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))
from paths import apply_model_env, ensure_sys_path, resolve_models_dir  # noqa: E402

_FARM_ROOT = ensure_sys_path(_HERE)
_MODELS_DIR = apply_model_env(resolve_models_dir(_FARM_ROOT))
_DEFAULT_VOCAB = (
    Path(os.environ.get("CONSTRUCTION_VOCAB", "")).resolve()
    if os.environ.get("CONSTRUCTION_VOCAB")
    else (_HERE.parent.parent / "vocab" / "construction_vocab.txt")
)
if not _DEFAULT_VOCAB.is_file():
    _DEFAULT_VOCAB = _HERE.parent / "vocab" / "construction_vocab.txt"
_MOBILECLIP_CKPT = _MODELS_DIR / "mobileclip" / "mobileclip_blt.pt"


def _resolve_yoloe_weights(model_id: str, prompt_free: bool) -> Path:
    suffix = "-seg-pf.pt" if prompt_free else "-seg.pt"
    local = _MODELS_DIR / "yoloe" / f"{model_id}{suffix}"
    if local.exists():
        return local
    raise FileNotFoundError(
        f"YOLOE weights not found at {local}. "
        f"Run ./bootstrap_models.sh from the repo root first."
    )


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

class ConstructionSegmenter:
    """Open-vocabulary instance segmenter for construction-site faces.

    Parameters
    ----------
    vocab_file : Path | str
        Path to the construction vocabulary text file.
        Defaults to vocab/construction_vocab.txt next to this file.
    model_id : str
        YOLOE checkpoint base name (default 'yoloe-v8l').
    imgsz : int
        YOLOE inference resolution (default 640 — matches FARM default).
    conf : float
        NMS confidence threshold (FARM default ~0.35-0.40).
    iou : float
        NMS IoU threshold (FARM default 0.5).
    device : str
        'cuda' or 'cpu'.
    mask_erosion_px : int
        Erode instance masks by this many pixels before depth unprojection.
        Prevents edge-leakage at object boundaries. FARM default = 3.
    mahalanobis_thresh : float
        Outlier rejection threshold for 3-D Gaussian fitting. FARM default = 2.0.
    """

    def __init__(
        self,
        vocab_file: Path | str | None = None,
        model_id: str = "yoloe-v8l",
        imgsz: int = 640,
        conf: float = 0.35,
        iou: float = 0.5,
        device: str = "cuda",
        mask_erosion_px: int = 3,
        mahalanobis_thresh: float = 2.0,
    ) -> None:
        vocab_file = Path(vocab_file) if vocab_file else _DEFAULT_VOCAB
        if not vocab_file.exists():
            raise FileNotFoundError(f"Vocab file not found: {vocab_file}")
        if not _MOBILECLIP_CKPT.is_file():
            raise FileNotFoundError(
                f"MobileCLIP weights not found at {_MOBILECLIP_CKPT}. "
                f"Run FARM-Project/bootstrap_models.sh first."
            )

        from scene_graph.segmentation.yoloe import YOLOESegmenter

        self._segmenter = YOLOESegmenter(
            model_id=model_id,
            vocab_file=vocab_file,
            imgsz=imgsz,
            conf_thres=conf,
            iou_thres=iou,
            device=device,
            use_dino_features=True,   # DINOv3 mask-pooled features; required for Phase 3
            mask_erosion_px=mask_erosion_px,
            mahalanobis_thresh=mahalanobis_thresh,
        )
        self.feature_dim = self._segmenter.feature_dim
        self.names = self._segmenter.names
        self.device = torch.device(device)

    def __call__(
        self,
        colors: List[torch.Tensor] | torch.Tensor,
        depths: List[torch.Tensor] | torch.Tensor,
        intrinsics: List[torch.Tensor] | torch.Tensor,
    ) -> dict:
        """Run YOLOE + DINOv3 on a batch of face images.

        Parameters
        ----------
        colors : list of (H, W, 3) uint8 tensors, length B
        depths : list of (H, W) float32 tensors in metric metres, length B
        intrinsics : list of (3, 3) float32 tensors (pinhole K), length B

        Returns
        -------
        dict with keys:
            means       (M, 3)  camera-frame 3-D centres
            cov6        (M, 6)  camera-frame packed covariances
            features    (M, D)  DINOv3 mask-pooled features
            masks       list[Tensor]  per-image (H_i, W_i) bool masks
            scores      (M,)
            class_ids   (M,)
            labels      list[str]
            batch_ids   (M,)
        """
        return self._segmenter(colors, depths, intrinsics)
