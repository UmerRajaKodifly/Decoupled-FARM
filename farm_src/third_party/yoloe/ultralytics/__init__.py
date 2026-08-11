# Ultralytics YOLO 🚀, AGPL-3.0 license

__version__ = "8.3.39"

import os

# YOLOE uses custom heads that mutate tensors in-place; disable torch.inference_mode to avoid
# RuntimeError: "Inference tensors do not track version counter" during warmup/predict calls.
os.environ.setdefault("ULTRALYTICS_DISABLE_INFERENCE_MODE", "1")

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

try:  # Prefer upstream Ultralytics models when available
    from ultralytics.models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOE
except ImportError:
    # Fallback to the vendored Ultralytics fork. Register it as `ultralytics` so that
    # absolute imports inside the fork keep resolving locally.
    import sys

    for name in list(sys.modules):
        if name == "ultralytics" or name.startswith("ultralytics."):
            sys.modules.pop(name, None)
    sys.modules["ultralytics"] = sys.modules[__name__]
    from .models import NAS, RTDETR, SAM, YOLO, FastSAM, YOLOE
from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

settings = SETTINGS
__all__ = (
    "__version__",
    "ASSETS",
    "YOLO",
    "YOLOE",
    "NAS",
    "SAM",
    "FastSAM",
    "RTDETR",
    "checks",
    "download",
    "settings",
)
