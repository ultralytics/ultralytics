#!/usr/bin/env python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Convert an old anomaly-v2 checkpoint to the current format.

This script exists because the anomaly_v2 code no longer carries __setstate__/
__getstate__ workarounds in the model classes. Run it once per old weight file
to migrate deprecated attributes and bake the current defaults into the saved
object, then use the converted file with the current code.

Example:
    python scripts/convert_anomaly_v2_ckpt.py \
        --input best_old.pt \
        --output best_converted.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from ultralytics.nn.modules.anomaly_v2 import AnomalyMemoryBank, BboxMaskRenderer, HeatmapProcessor
from ultralytics.nn.tasks import YOLOAnomalyV2Model


def _migrate_bbox_mask_renderer(m: BboxMaskRenderer) -> None:
    """Backfill sigma_lo/sigma_hi for checkpoints saved before the [lo, hi] range knob."""
    if not hasattr(m, "sigma_lo") or not hasattr(m, "sigma_hi"):
        sf = float(getattr(m, "sigma_factor", 0.25))
        m.sigma_lo = m.sigma_hi = sf
    if not hasattr(m, "sigma_factor"):
        m.sigma_factor = m.sigma_lo


def _migrate_anomaly_memory_bank(m: AnomalyMemoryBank) -> None:
    """Migrate an old BackboneMemoryBank pickle to the current AnomalyMemoryBank layout."""
    # Drop deprecated projection state from very old checkpoints.
    if "proj_dim" in m.__dict__:
        delattr(m, "proj_dim")
    if "_proj_weight" in m._buffers:
        del m._buffers["_proj_weight"]

    # Rename old tensor attribute to the new registered buffer.
    if hasattr(m, "memory_bank"):
        bank_tensor = m.memory_bank
    else:
        bank_tensor = torch.empty(0, 0)
    if "bank" not in m._buffers:
        m.register_buffer("bank", bank_tensor if isinstance(bank_tensor, torch.Tensor) else torch.empty(0, 0))
    else:
        m.bank = bank_tensor
    if hasattr(m, "memory_bank") and "memory_bank" not in m._buffers:
        delattr(m, "memory_bank")

    # Rename scalar / transient attributes.
    if hasattr(m, "feature_dim") and not hasattr(m, "dim"):
        m.dim = m.feature_dim
    if hasattr(m, "update") and not hasattr(m, "building"):
        m.building = m.update
    if hasattr(m, "_bank_chunks") and not hasattr(m, "_chunks"):
        m._chunks = m._bank_chunks

    if not hasattr(m, "_compactness"):
        m._compactness = None
    if not hasattr(m, "_threshold"):
        m._threshold = None

    # Drop deprecated / removed attributes.
    for attr in (
        "calibration_target_quantile",
        "hmap_stretch_strength",
        "holdout_max",
        "max_bank_size",
        "score_chunk_elems",
        "_calibrated",
        "feature_dim",
        "update",
        "_bank_chunks",
    ):
        if attr in m.__dict__:
            delattr(m, attr)


def _migrate_heatmap_processor(m: HeatmapProcessor) -> None:
    """Ensure HeatmapProcessor owns the current default knobs."""
    defaults = HeatmapProcessor(mask_size=getattr(m, "mask_size", 80)).__dict__
    for k in ("norm", "smooth_kernel", "edge_weight", "edge_p", "edge_m", "edge_sigma"):
        setattr(m, k, defaults[k])
    # Transient cache should not be saved.
    m._edge_weight_cache = None


def _migrate_v2_model(m: YOLOAnomalyV2Model) -> None:
    """Apply the migration that used to live in YOLOAnomalyV2Model.__setstate__."""
    # Ensure submodules are migrated.
    for module in m.modules():
        if isinstance(module, BboxMaskRenderer):
            _migrate_bbox_mask_renderer(module)
        elif isinstance(module, AnomalyMemoryBank):
            _migrate_anomaly_memory_bank(module)
        elif isinstance(module, HeatmapProcessor):
            _migrate_heatmap_processor(module)

    # Rename legacy _bb_layers -> bb_layers.
    if "_bb_layers" in m.__dict__ and "bb_layers" not in m.__dict__:
        m.bb_layers = m.__dict__.pop("_bb_layers")

    # Drop transient / deprecated attributes.
    for attr in (
        "_bb_feats",
        "_heatmap_bank_warned",
        "prior_context",
        "_use_heatmap_prior",
        "use_heatmap_prior",
        "heatmap_norm",
        "heatmap_smooth_kernel",
        "heatmap_edge_weight",
        "heatmap_edge_p",
        "heatmap_edge_m",
        "heatmap_edge_sigma",
        "_edge_weight_cache",
        "mask_renderer",
        "mask_augmenter",
        "fit_args",
    ):
        if attr in m.__dict__:
            delattr(m, attr)

    # Ensure v2.3+ attributes exist.
    for attr, default in (
        ("p_drop", 0.5),
        ("seg_target_polygon", False),
        ("softmax_temperature", 1.0),
        ("fit_data", None),
        ("fusion_mid", 32),
    ):
        if not hasattr(m, attr):
            setattr(m, attr, default)

    # Ensure the heatmap processor exists and is reset to defaults.
    if not hasattr(m, "heatmap_processor"):
        m.heatmap_processor = HeatmapProcessor(mask_size=getattr(m, "mask_size", 80))
    _migrate_heatmap_processor(m.heatmap_processor)

    # Reset bank defaults so pickled bank-build hyperparameters never override the code defaults.
    mb = getattr(m, "memory_bank", None)
    if mb is not None:
        defaults = AnomalyMemoryBank().__dict__
        for k in ("bank_size", "K", "temperature", "target_score", "stretch", "score_chunk"):
            setattr(mb, k, defaults[k])
        mb.bank = torch.empty(0, 0, device=mb.bank.device)
        mb.dim = None
        mb._compactness = None
        mb._threshold = None
        mb.building = True
        mb._chunks = []


def convert_checkpoint(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Load an old checkpoint, migrate it, and save the migrated checkpoint."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, YOLOAnomalyV2Model):
        model = ckpt
        ckpt = {"model": model, "updates": None, "optimizer": None, "args": None}
    else:
        model = ckpt.get("model")

    if not isinstance(model, YOLOAnomalyV2Model):
        raise TypeError(f"expected YOLOAnomalyV2Model, got {type(model).__name__}")

    model.eval()
    _migrate_v2_model(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_path)
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an old anomaly-v2 checkpoint to the current format.")
    parser.add_argument("--input", "-i", required=True, type=Path, help="Path to the old checkpoint.")
    parser.add_argument("--output", "-o", required=True, type=Path, help="Path to write the converted checkpoint.")
    args = parser.parse_args()

    convert_checkpoint(args.input, args.output)
    print(f"Converted {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
