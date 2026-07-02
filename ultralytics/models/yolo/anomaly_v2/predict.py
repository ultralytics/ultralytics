# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 predictor with unified prior-mode routing.

``YOLOAnomalyPredictor`` extends ``DetectionPredictor`` with prior-injection in
``preprocess`` / ``inference`` via ``YOLOAnomalyPredictorBase``.

Prior modes selectable via ``predictor.prior_mode``:

  - ``"none"``     — passthrough (vanilla YOLO, no fusion bias).
  - ``"heatmap"``  — feature-side memory-bank anomaly map.

Explicit masks (prompts, external heatmaps, etc.) are supplied via the
``prior_mask`` argument and fused directly; no one-shot buffer API is used.
"""

from __future__ import annotations

import torch

from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG

from ._util import resolve_v2_model


class YOLOAnomalyPredictorBase:
    """Shared prior-mode injection for the Detect-head anomaly predictor."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        # Pop custom keys from overrides before the base predictor validates all keys.
        prior_mode = None
        prior_mask = None
        if isinstance(overrides, dict):
            prior_mode = overrides.pop("prior_mode", None)
            prior_mask = overrides.pop("prior_mask", None)
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.prior_mode = prior_mode
        self.prior_mask = prior_mask

    def preprocess(self, im):
        """Set heatmap prior mode on the model before forward."""
        m = resolve_v2_model(self.model)
        if m is not None and hasattr(m, "set_prior_mode"):
            # "heatmap" enables the internal memory-bank path; everything else
            # (none / mask / anomaly_model) is handled via explicit prior_mask.
            m.set_prior_mode("heatmap" if self.prior_mode == "heatmap" else None)
        return super().preprocess(im)

    def inference(self, im, *args, **kwargs):
        """Run inference, passing any explicit prior_mask to the model."""
        prior_mask = getattr(self, "prior_mask", None)
        if prior_mask is not None:
            m = resolve_v2_model(self.model)
            device = next(m.parameters()).device if m is not None else prior_mask.device
            kwargs["prior_mask"] = prior_mask.to(device=device, dtype=torch.float32)
        return super().inference(im, *args, **kwargs)


class YOLOAnomalyPredictor(YOLOAnomalyPredictorBase, DetectionPredictor):
    """YOLO Anomaly v2 predictor (Detect head) with configurable prior mode."""
