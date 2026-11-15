# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch

from ultralytics.nn.modules import Depth, SemanticSegment
from ultralytics.utils.export.tensorflow import tf_wrapper


def stedgeai_wrapper(model: torch.nn.Module) -> torch.nn.Module:
    """Patch model heads to emit outputs that survive whole-graph INT8 quantization on STM32 devices.

    Detect and Pose heads emit box and keypoint coordinates normalized to [0, 1] by the input size, so they share one
    INT8 scale with the class scores. Semantic heads stop at the [B, nc, H/8, W/8] logits and Depth heads at the
    calibrated [B, 1, H/4, W/4] log-depth, which keeps the on-device output small and the INT8 depth range linear in
    log space; post-processing upsamples both and applies exp to the log-depth.

    Args:
        model (torch.nn.Module): Model whose heads are patched in place.

    Returns:
        (torch.nn.Module): The same model with patched head methods.
    """
    for m in model.modules():
        if isinstance(m, SemanticSegment):
            m.forward = lambda x, m=m: m.classifier(x[0])
        elif isinstance(m, Depth):
            m.forward = lambda x, m=m: m.cal_a * m.log_depth(x) + m.cal_b
    return tf_wrapper(model)
