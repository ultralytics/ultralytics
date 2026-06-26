# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 predictors with unified prior-mode routing.

Two predictors share the prior-injection ``preprocess`` via ``YOLOAnomalyPredictorBase``:

  - ``YOLOAnomalyPredictor``    — Detect head (boxes).
  - ``YOLOAnomalySegPredictor`` — Segment head (boxes + per-instance masks); reuses
    ``SegmentationPredictor``'s proto/mask decoding.

``YOLOA.task_map`` picks the right one from the loaded checkpoint's head type, mirroring
how ``AnomalyV2Trainer.get_model`` selects ``YOLOAnomalyV2Model`` vs ``YOLOAnomalyV2SegModel``.

Prior modes selectable via ``predictor.prior_mode``:

  - ``"none"``     — passthrough (vanilla YOLO, no fusion bias).
  - ``"segment"``  — SegBranch sigmoid output as prior.
  - ``"heatmap"``  — feature-side anomaly map (producer set by ``_heatmap_producer``:
                     bank / learned / both / cached). Legacy ``"heatmap_learned"`` /
                     ``"heatmap_fused"`` / ``"cached"`` are translated by ``set_prior_mode``.
  - ``"mask"``     — external_mask provided by caller.

Legacy ``external_mask`` / ``bbox_prompt`` attributes still work when
``prior_mode`` is ``None`` (backward compat).
"""

from __future__ import annotations

import torch

from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.utils import DEFAULT_CFG

from ._util import resolve_v2_model


class YOLOAnomalyPredictorBase:
    """Shared prior-mode injection for the Detect- and Segment-head anomaly predictors."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        # Pop custom keys from overrides before the base predictor validates all keys.
        prior_mode = overrides.pop("prior_mode", None) if isinstance(overrides, dict) else None
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        self.prior_mode = prior_mode
        self.bbox_prompt: tuple[torch.Tensor, torch.Tensor] | None = None
        self.external_mask: torch.Tensor | None = None

    def preprocess(self, im):
        """Set prior mode and optional mask on the model before forward."""
        m = resolve_v2_model(self.model)
        if m is not None and hasattr(m, "set_prior_mode"):
            m.set_prior_mode(self.prior_mode)
            device = next(m.parameters()).device
            # Legacy prompt handling (backward compat)
            if self.prior_mode is None or self.prior_mode == "mask":
                if self.external_mask is not None:
                    if hasattr(m, "set_external_mask_once"):
                        m.set_external_mask_once(self.external_mask.to(device))
                    elif hasattr(m, "_external_mask_buf"):
                        m._external_mask_buf = self.external_mask.to(device)
                elif self.bbox_prompt is not None:
                    bb, bi = self.bbox_prompt
                    m.set_mask_input(bb.to(device), bi.to(device))
                elif self.prior_mode is None:
                    # Legacy: no prompt → passthrough
                    m.disable_mask_once()
        return super().preprocess(im)


class YOLOAnomalyPredictor(YOLOAnomalyPredictorBase, DetectionPredictor):
    """YOLO Anomaly v2 predictor (Detect head) with configurable prior mode."""


class YOLOAnomalySegPredictor(YOLOAnomalyPredictorBase, SegmentationPredictor):
    """YOLO Anomaly v2 predictor (Segment head): boxes + per-instance masks with prior mode."""
