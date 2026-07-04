# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 predictor with optional heatmap prior."""

from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils.anomaly_v2 import PriorContext, PriorMode, prior_context_from_overrides

from ._util import resolve_v2_model


def _ctx_with_mode(base: PriorContext, mode: PriorMode) -> PriorContext:
    """Return a fresh PriorContext with ``mode`` replaced, copying only known fields."""
    return PriorContext(
        mode=mode,
        heatmap_norm=base.heatmap_norm,
        heatmap_smooth_kernel=base.heatmap_smooth_kernel,
        heatmap_edge_weight=base.heatmap_edge_weight,
        heatmap_edge_p=base.heatmap_edge_p,
        heatmap_edge_m=base.heatmap_edge_m,
        heatmap_edge_sigma=base.heatmap_edge_sigma,
    )


class YOLOAnomalyPredictor(DetectionPredictor):
    """YOLO Anomaly v2 predictor.

    Uses the memory-bank heatmap prior when a non-empty bank is available.
    Otherwise it falls back to regular detection inference.
    """

    def preprocess(self, im):
        """Enable the heatmap prior only when a built memory bank exists."""
        m = resolve_v2_model(self.model)
        if m is not None and hasattr(m, "set_prior_context"):
            base = m.prior_context if isinstance(m.prior_context, PriorContext) else PriorContext()
            mb = getattr(m, "memory_bank", None)
            has_bank = (
                mb is not None
                and getattr(mb, "memory_bank", None) is not None
                and mb.memory_bank.shape[0] > 0
            )
            if has_bank:
                ctx = _ctx_with_mode(base, PriorMode.HEATMAP)
                ctx = prior_context_from_overrides(dict(vars(self.args)), ctx)
                m.set_prior_context(ctx)
            else:
                m.set_prior_context(_ctx_with_mode(base, PriorMode.NONE))
        return super().preprocess(im)
