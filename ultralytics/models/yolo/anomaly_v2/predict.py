# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 predictor.

By default mask is disabled (no prior available outside training/val), so the
v2 model is exactly equivalent to a vanilla YOLO detector on inference inputs
that don't have labels (passthrough via 2*sigmoid(0)=1).

Future Phase 0+ work may attach an external prior (MemoryBank, hand-drawn mask)
here -- the model already accepts that via ``set_mask_input``.
"""

from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils.torch_utils import unwrap_model


class AnomalyV2Predictor(DetectionPredictor):
    """Standard YOLO detection predictor; v2 model is mask-off (passthrough) by default."""

    def preprocess(self, im):
        """Before each prediction batch, ensure the model is in mask-off mode."""
        m = unwrap_model(self.model) if self.model is not None else None
        if m is not None and hasattr(m, "disable_mask_once"):
            m.disable_mask_once()
        return super().preprocess(im)
