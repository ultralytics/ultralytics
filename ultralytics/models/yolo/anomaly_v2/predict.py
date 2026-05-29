# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 predictor.

By default mask is disabled (no prior available outside training/val), so the
v2 model is exactly equivalent to a vanilla YOLO detector on inference inputs
that don't have labels (passthrough via 2*sigmoid(0)=1).

Optional prompt injection (Phase 0+ exploration):
  - ``predictor.bbox_prompt = (bboxes, batch_idx)`` -> renderer turns them into
    a mask using the model's configured mask_mode (rect/gauss).
  - ``predictor.external_mask = mask_tensor`` -> a pre-computed mask is fed
    directly to the heatmap encoder, bypassing the renderer. Useful for
    hand-drawn masks or downstream MemoryBank priors.

Both prompts are persistent on the predictor instance until cleared; they
affect every subsequent forward until set to ``None``. ``external_mask`` takes
precedence over ``bbox_prompt`` if both are set.
"""

from __future__ import annotations

import torch

from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils.torch_utils import unwrap_model


class AnomalyV2Predictor(DetectionPredictor):
    """YOLO Anomaly v2 predictor with optional prompt-style mask injection."""

    # Persistent prompt slots; set externally to override the mask-off default.
    bbox_prompt: tuple[torch.Tensor, torch.Tensor] | None = None
    external_mask: torch.Tensor | None = None

    def preprocess(self, im):
        """Set mask state on the model based on configured prompt."""
        m = unwrap_model(self.model) if self.model is not None else None
        if m is not None and hasattr(m, "disable_mask_once"):
            device = next(m.parameters()).device
            if self.external_mask is not None:
                if not hasattr(m, "set_external_mask_once"):
                    raise RuntimeError(
                        "Model does not support external_mask; rebuild from this branch."
                    )
                m.set_external_mask_once(self.external_mask.to(device))
            elif self.bbox_prompt is not None:
                bb, bi = self.bbox_prompt
                m.set_mask_input(bb.to(device), bi.to(device))
            else:
                m.disable_mask_once()
        return super().preprocess(im)
