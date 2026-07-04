# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Shared anomaly-v2 helpers that must not depend on the high-level YOLOA wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ultralytics.utils import LOGGER


class PriorMode(str, Enum):
    """Anomaly-v2 prior sources."""

    NONE = "none"
    HEATMAP = "heatmap"


@dataclass
class PriorContext:
    """Everything the model needs to resolve a fusion prior for one forward pass."""

    mode: PriorMode = PriorMode.NONE
    heatmap_norm: str = "none"  # "none" | "minmax" | "gaussian" | "mean"
    heatmap_smooth_kernel: int = 5
    heatmap_edge_weight: bool = False
    heatmap_edge_p: float = 4.0
    heatmap_edge_m: float = 4.4
    heatmap_edge_sigma: float = 1.0

    def is_heatmap(self) -> bool:
        """True when the prior should be built from the memory bank."""
        return self.mode == PriorMode.HEATMAP


# Bank-build knobs that live in the fit config. bb_* override the model yaml's v2_cfg defaults;
# imgsz / max_images are build inputs (not part of the model yaml).
FIT_KEYS = (
    "imgsz",
    "max_images",
    "bb_layers",
    "bb_max_bank_size",
    "bb_K",
    "bb_temperature",
    "bb_calibration_target_score",
    "bb_calibration_target_quantile",
    "bb_hmap_stretch_strength",
    "bb_holdout_max",
)

# fit-key -> BackboneMemoryBank attribute it sets (bb_layers handled separately: it re-taps).
_BB_TO_MB = {
    "bb_max_bank_size": "max_bank_size",
    "bb_K": "K",
    "bb_temperature": "temperature",
    "bb_calibration_target_score": "calibration_target_score",
    "bb_calibration_target_quantile": "calibration_target_quantile",
    "bb_hmap_stretch_strength": "hmap_stretch_strength",
    "bb_holdout_max": "holdout_max",
}


def apply_bb_overrides(model, fit_args: dict) -> None:
    """Apply bb_* fit overrides onto a YOLOAnomalyV2Model + its memory bank before the bank is built.

    Shared by :meth:`YOLOA.fit` and the trainer's MVTec OOD eval so both build the bank with the
    same fit-config knobs (bb_K / bb_temperature / calibration / bb_max_bank_size / bb_layers re-tap),
    keeping in-training and post-training evaluation identical. Only keys present (non-None) in
    ``fit_args`` override the model-baked v2_cfg defaults.
    """
    mb = model.memory_bank
    new_layers = fit_args.get("bb_layers")
    if new_layers is not None and list(new_layers) != list(model._bb_layers or []):
        new_layers = list(new_layers)
        LOGGER.warning(
            f"apply_bb_overrides: bb_layers {model._bb_layers} -> {new_layers} (rebuilds bank; "
            f"deviating from the training layer set may shift fusion behavior)"
        )
        model._bb_layers = new_layers
        mb._bb_layer_indices = new_layers
    for fk, mbk in _BB_TO_MB.items():
        if fit_args.get(fk) is not None:
            setattr(mb, mbk, fit_args[fk])


# Canonical prior-shaping keys plus short aliases used by YOLOA.
_INFER_ALIASES = {
    "heat_norm": "heatmap_norm",
    "heat_smooth_kernel": "heatmap_smooth_kernel",
    "heat_edge": "heatmap_edge_weight",
    "heat_edge_p": "heatmap_edge_p",
    "heat_edge_m": "heatmap_edge_m",
    "heat_edge_sigma": "heatmap_edge_sigma",
}


def prior_context_from_overrides(overrides: dict[str, Any], defaults: PriorContext | None = None) -> PriorContext:
    """Build a :class:`PriorContext` from predictor/validator override kwargs.

    Pops the anomaly-specific heatmap-shaping keys so the remaining kwargs are safe for the base
    predictor/validator config. Accepts short aliases (``heat_edge`` -> ``heatmap_edge_weight``).
    """
    ctx = defaults or PriorContext()

    def _pop(key: str, default: Any) -> Any:
        if key in overrides:
            return overrides.pop(key)
        return overrides.pop(_INFER_ALIASES.get(key), default)

    return PriorContext(
        mode=ctx.mode,
        heatmap_norm=_pop("heatmap_norm", ctx.heatmap_norm),
        heatmap_smooth_kernel=_pop("heatmap_smooth_kernel", ctx.heatmap_smooth_kernel),
        heatmap_edge_weight=_pop("heatmap_edge_weight", ctx.heatmap_edge_weight),
        heatmap_edge_p=_pop("heatmap_edge_p", ctx.heatmap_edge_p),
        heatmap_edge_m=_pop("heatmap_edge_m", ctx.heatmap_edge_m),
        heatmap_edge_sigma=_pop("heatmap_edge_sigma", ctx.heatmap_edge_sigma),
    )
