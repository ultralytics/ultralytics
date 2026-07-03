# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Shared anomaly-v2 helpers that must not depend on the high-level YOLOA wrapper."""

from __future__ import annotations

from ultralytics.utils import LOGGER


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
