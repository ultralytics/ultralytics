# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Re-save yolo27 DEIM checkpoints after the DEIM class rename.

The rename commit moved ``DeimDecoder`` -> ``DEIMDecoder`` and folded the D-FINE decoder classes into
``ultralytics.nn.modules.deim_transformer`` (``dfine_transformer.py`` was deleted). Pickled checkpoints store class
module paths, so pre-rename ``.pt`` files fail to load with the new code. This script loads each checkpoint with a
temporary alias shim, fixes the embedded model YAML, and re-saves it; afterwards the files load with the new code
directly.

Usage:
    python scripts/convert_yolo27_deim_weights.py weights/27/det/yolo27m.pt [more.pt ...]
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import torch


def _install_legacy_aliases() -> None:
    """Map pre-rename module paths onto the renamed DEIM classes so old pickles resolve."""
    from ultralytics.nn.modules import deim_transformer, head

    legacy_transformer = types.ModuleType("ultralytics.nn.modules.dfine_transformer")
    for old, new in {
        "DFineTransformerDecoder": "DEIMTransformerDecoder",
        "DeimTransformerDecoder": "DEIMTransformerDecoder",
        "DeimTransformerDecoderLayer": "DEIMTransformerDecoderLayer",
        "DeimGate": "DEIMGate",
        "MSDeformableAttention": "MSDeformableAttention",
        "Integral": "Integral",
        "LQE": "LQE",
        "DEIMRMSNorm": "DEIMRMSNorm",
        "DEIMSwiGLUFFN": "DEIMSwiGLUFFN",
    }.items():
        setattr(legacy_transformer, old, getattr(deim_transformer, new))
    sys.modules[legacy_transformer.__name__] = legacy_transformer

    from ultralytics.models.utils import loss as models_loss

    legacy_loss = types.ModuleType("ultralytics.models.utils.loss_dfine")
    legacy_loss.DfineLoss = models_loss.DEIMLoss
    sys.modules[legacy_loss.__name__] = legacy_loss

    head.DeimDecoder = head.DEIMDecoder


def convert(path: Path) -> None:
    """Load ``path`` through the legacy aliases, update the embedded YAML head name, and re-save in place."""
    from ultralytics.nn.tasks import torch_safe_load

    ckpt, _ = torch_safe_load(str(path))
    for m in (ckpt.get("model"), ckpt.get("ema")):
        if m is not None and hasattr(m, "yaml"):
            for section in ("backbone", "head"):
                for layer in m.yaml.get(section, []):
                    if len(layer) > 2 and layer[2] == "DeimDecoder":
                        layer[2] = "DEIMDecoder"
    torch.save(ckpt, path)
    print(f"converted {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", nargs="+", type=Path, help="Checkpoint paths to convert in place.")
    args = parser.parse_args()
    _install_legacy_aliases()
    for w in args.weights:
        convert(w)
