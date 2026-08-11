"""DINOv3 mask-pooled feature extractor for Phase 2.

Thin wrapper around FARM's DINOFeaturesExtractor that:
  - resolves local model weights from FARM-Project/models/dinov3-vits16/
  - exposes a single extract() function matching the interface expected by
    segmenter.py and the output schema consumed by Phase 3.

The mask-pooling logic is ported verbatim from FARM's
YOLOESegmenter._compute_dino_mask_embeddings (yoloe.py lines 698-760).
Output features are L2-normalised 384-D float32 tensors, one per detection.

Why DINOv3?
-----------
DINOv3 features are view-invariant appearance descriptors. Phase 3's neighbor
search uses cosine similarity between these features to decide whether a new
detection is the same object as an existing map entry seen from a different
viewpoint. This requires features that are stable across viewpoints and
illumination — exactly what DINOv3 patch tokens provide.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Locate FARM model weights
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
_COMMON = _HERE.parent.parent / "common"
if _COMMON.is_dir():
    import sys as _sys
    if str(_COMMON) not in _sys.path:
        _sys.path.insert(0, str(_COMMON))
from paths import ensure_sys_path, resolve_models_dir  # noqa: E402

_FARM_ROOT = ensure_sys_path(_HERE)
_MODELS_DIR = resolve_models_dir(_FARM_ROOT)
_DINO_WEIGHTS_DIR = _MODELS_DIR / "dinov3-vits16"
_DINO_PLUS_WEIGHTS_DIR = _MODELS_DIR / "dinov3-vits16plus"

HIDDEN_SIZE = 384  # ViT-S/16 patch-token dimension


def _resolve_dino_weights() -> tuple[str, str | None]:
    """Return (model_id_string, local_weights_path_or_None).

    Auto-prefers ViT-S+/16 (paper backbone) over ViT-S/16 (bundled),
    matching FARM's runtime_paths.resolve_dino_backbone behaviour.
    """
    if _DINO_PLUS_WEIGHTS_DIR.exists() and any(_DINO_PLUS_WEIGHTS_DIR.iterdir()):
        return "facebook/dinov3-vits16plus-pretrain-lvd1689m", str(_DINO_PLUS_WEIGHTS_DIR)
    if _DINO_WEIGHTS_DIR.exists() and any(_DINO_WEIGHTS_DIR.iterdir()):
        return "facebook/dinov3-vits16-pretrain-lvd1689m", str(_DINO_WEIGHTS_DIR)
    # Fallback: let HF download (requires internet)
    return "facebook/dinov3-vits16-pretrain-lvd1689m", None


# ---------------------------------------------------------------------------
# Lazy import of FARM's DINOFeaturesExtractor
# ---------------------------------------------------------------------------

def _get_farm_dino_class():
    """Import FARM's extractor, adding its src/ to sys.path if needed."""
    try:
        from scene_graph.segmentation.dino import DINOFeaturesExtractor
        return DINOFeaturesExtractor
    except ImportError:
        ensure_sys_path(_HERE)
        from scene_graph.segmentation.dino import DINOFeaturesExtractor
        return DINOFeaturesExtractor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Phase2DinoExtractor:
    """Mask-pooled DINOv3 feature extractor matching FARM's live pipeline.

    Parameters
    ----------
    device : str or torch.device
        'cuda' or 'cpu'.
    load_size : int
        Image size DINOv3 is run at (FARM default 512).
    """

    def __init__(
        self,
        device: str | torch.device = "cuda",
        load_size: int = 512,
    ) -> None:
        self.device = torch.device(device)
        self.load_size = load_size
        self.hidden_size = HIDDEN_SIZE

        model_id, weights_path = _resolve_dino_weights()
        DINOFeaturesExtractor = _get_farm_dino_class()

        self._extractor = DINOFeaturesExtractor(
            model=model_id,
            weights_path=weights_path,
            load_size=load_size,
            fp16=self.device.type == "cuda",
            device=self.device,
        )
        # Update hidden_size from actual model if available
        hs = getattr(self._extractor, "hidden_size", None)
        if hs is not None:
            self.hidden_size = int(hs)

    def extract(
        self,
        colors: List[torch.Tensor],      # list of (H, W, 3) uint8 or float tensors
        masks: torch.Tensor,             # (M, H_lb, W_lb) bool — letterboxed masks
        batch_ids: torch.Tensor,         # (M,) int — which image each detection belongs to
    ) -> torch.Tensor:
        """Return (M, D) L2-normalised mask-pooled DINOv3 features.

        Ported directly from FARM's YOLOESegmenter._compute_dino_mask_embeddings.

        Parameters
        ----------
        colors : list of (H, W, 3) tensors, length B
            RGB frames in the batch (values 0-255 uint8 or 0-1 float).
        masks : (M, H, W) bool
            Instance masks at the same spatial resolution as the images
            (or at DINOv3 token-grid resolution after interpolation inside).
        batch_ids : (M,) int
            Which element of `colors` each detection belongs to.
        """
        M = masks.shape[0]
        B = len(colors)

        if M == 0 or B == 0:
            return torch.zeros((0, self.hidden_size), device=self.device, dtype=torch.float32)

        # Forward DINOv3 on all images; get per-image token grids [H', W', C]
        grids = self._extractor(colors)   # list of [H', W', C]
        if not grids:
            return torch.zeros((M, self.hidden_size), device=self.device, dtype=torch.float32)

        try:
            grid_tensor = torch.stack(
                [g.to(self.device) if g.device != self.device else g for g in grids],
                dim=0,
            )  # (B, H', W', C)
        except Exception:
            return torch.zeros((M, self.hidden_size), device=self.device, dtype=torch.float32)

        _B, gH, gW, gC = grid_tensor.shape
        embeddings = torch.zeros((M, gC), device=self.device, dtype=torch.float32)

        batch_ids_dev = batch_ids.to(self.device, dtype=torch.long)
        valid_det = (batch_ids_dev >= 0) & (batch_ids_dev < _B)
        if not valid_det.any():
            return embeddings

        valid_idx = valid_det.nonzero(as_tuple=False).squeeze(1)
        batch_ids_valid = batch_ids_dev[valid_idx]

        # Resize masks to token grid in one batched call
        masks_valid = masks[valid_idx].to(device=self.device, dtype=torch.float32)
        masks_resized = F.interpolate(
            masks_valid.unsqueeze(1),
            size=(gH, gW),
            mode="nearest",
        ).squeeze(1)  # (M_valid, H', W')

        # Mask-pool: for each detection average the DINOv3 tokens under its mask
        for local_i, (glob_i, b_idx) in enumerate(zip(valid_idx.tolist(), batch_ids_valid.tolist())):
            feat_map = grid_tensor[b_idx]              # (H', W', C)
            mask_i = masks_resized[local_i]            # (H', W')
            w_sum = mask_i.sum()
            if w_sum.item() <= 0:
                continue
            # feat_map: (H', W', C) — apply mask, average over spatial dims
            feat_masked = feat_map * mask_i.unsqueeze(-1)  # (H', W', C)
            embeddings[glob_i] = feat_masked.sum(dim=(0, 1)) / w_sum.clamp_min(1e-6)

        # L2-normalise (matches FARM's DINOFeaturesExtractor output normalisation)
        norms = embeddings.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return (embeddings / norms).float()
