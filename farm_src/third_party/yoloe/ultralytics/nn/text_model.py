import os
from abc import abstractmethod
from pathlib import Path

import mobileclip
import torch
import torch.nn as nn
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import smart_inference_mode

class TextModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def tokenize(texts):
        pass
    
    @abstractmethod
    def encode_text(texts, dtype):
        pass

def _find_repo_root(start: Path) -> Path | None:
    """Walk up the directory tree looking for a git root."""
    for parent in start.parents:
        if (parent / ".git").exists():
            return parent
    return None


def _resolve_mobileclip_checkpoint(size: str) -> Path:
    """Return a best-effort path to the requested MobileCLIP checkpoint."""
    env_specific = os.environ.get(f"MOBILECLIP_{size.upper()}_CKPT")
    candidates = []
    if env_specific:
        candidates.append(Path(env_specific).expanduser())

    env_generic = os.environ.get("MOBILECLIP_CHECKPOINT")
    if env_generic:
        candidates.append(Path(env_generic).expanduser())

    weights_dir = os.environ.get("MOBILECLIP_WEIGHTS_DIR")
    if weights_dir:
        candidates.append(Path(weights_dir).expanduser() / f"mobileclip_{size}.pt")

    # SCENE_GRAPH_MODEL_DIR — standard model root for scene_graph project
    sg_model_dir = os.environ.get("SCENE_GRAPH_MODEL_DIR")
    if sg_model_dir:
        candidates.append(Path(sg_model_dir).expanduser() / "mobileclip" / f"mobileclip_{size}.pt")

    current = Path(__file__).resolve()
    for parent in current.parents:
        ros_candidate = (
            parent
            / "install"
            / "mapping"
            / "share"
            / "mapping"
            / "models"
            / f"mobileclip_{size}.pt"
        )
        if ros_candidate.is_file():
            candidates.append(ros_candidate)
            break

    repo_root = _find_repo_root(Path(__file__).resolve())
    if repo_root:
        candidates.append(repo_root / "models" / "mobileclip" / f"mobileclip_{size}.pt")

    candidates.append(Path.home() / "tiamat" / "models" / "mobileclip" / f"mobileclip_{size}.pt")
    candidates.append(Path(f"mobileclip_{size}.pt"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    LOGGER.warning("MobileCLIP checkpoint not found in expected locations, falling back to %s", candidates[-1])
    return candidates[-1]


class MobileCLIP(TextModel):
    
    config_size_map = {
        "s0": "s0",
        "s1": "s1",
        "s2": "s2",
        "b": "b",
        "blt": "b"
    }
    
    def __init__(self, size, device):
        super().__init__()
        config = self.config_size_map[size]
        checkpoint_path = _resolve_mobileclip_checkpoint(size)
        self.model = mobileclip.create_model_and_transforms(
            f'mobileclip_{config}',
            pretrained=str(checkpoint_path),
            device=device,
        )[0]
        self.tokenizer = mobileclip.get_tokenizer(f'mobileclip_{config}')
        self.to(device)
        self.device = device
        self.eval()
    
    def tokenize(self, texts):
        text_tokens = self.tokenizer(texts).to(self.device)
        # max_len = text_tokens.argmax(dim=-1).max().item() + 1
        # text_tokens = text_tokens[..., :max_len]
        return text_tokens

    @smart_inference_mode()
    def encode_text(self, texts, dtype=torch.float32):
        text_features = self.model.encode_text(texts).to(dtype)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features

def build_text_model(variant, device=None):
    LOGGER.info(f"Build text model {variant}")
    base, size = variant.split(":")
    if base == 'clip':
        return CLIP(size, device)
    elif base == 'mobileclip':
        return MobileCLIP(size, device)
    else:
        print("Variant not found")
        assert(False)
