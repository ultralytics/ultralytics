# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 predictor with optional heatmap prior."""

from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionPredictor

from ._util import resolve_v2_model


class YOLOAnomalyPredictor(DetectionPredictor):
    """YOLO Anomaly v2 predictor.

    Uses the memory-bank heatmap prior when a non-empty bank is available.
    Otherwise it falls back to regular detection inference.
    """

    def preprocess(self, im):
        """Enable the heatmap prior only when a built memory bank exists."""
        m = resolve_v2_model(self.model)
        if m is not None:
            mb = getattr(m, "memory_bank", None)
            has_bank = (
                mb is not None
                and getattr(mb, "memory_bank", None) is not None
                and mb.memory_bank.shape[0] > 0
            )
            m.use_heatmap_prior = bool(has_bank)
        return super().preprocess(im)
