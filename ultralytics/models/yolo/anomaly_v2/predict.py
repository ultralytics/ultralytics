# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 predictor with unified prior-mode routing.

Four prior modes selectable via ``predictor.prior_mode``:

  - ``"none"``        — passthrough (vanilla YOLO, no fusion bias).
  - ``"segment"``     — SegBranch sigmoid output as prior.
  - ``"heatmap"``     — BackboneMemoryBank output as prior.
  - ``"seg_heatmap"`` — average of segment + memory-bank heatmaps.
  - ``"mask"``        — external_mask provided by caller.

Legacy ``external_mask`` / ``bbox_prompt`` attributes still work when
``prior_mode`` is ``None`` (backward compat).
"""

from __future__ import annotations

import torch

from ultralytics.models.yolo.detect import DetectionPredictor

from ._util import resolve_v2_model


class AnomalyV2Predictor(DetectionPredictor):
    """YOLO Anomaly v2 predictor with configurable prior mode."""

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        # Pop custom keys from overrides before super validates all keys
        if isinstance(overrides, dict):
            prior_mode = overrides.pop("prior_mode", None)
        else:
            prior_mode = None
        # Don't pass cfg=None explicitly — let DetectionPredictor's DEFAULT_CFG default kick in
        init_kw = {"overrides": overrides, "_callbacks": _callbacks}
        if cfg is not None:
            init_kw["cfg"] = cfg
        super().__init__(**init_kw)
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
