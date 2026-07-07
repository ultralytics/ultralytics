#!/usr/bin/env python
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Convert an old anomaly checkpoint to the current AnomalyDetect-head format.

Older anomaly checkpoints stored ``HeatmapBiasFusion`` and ``HeatmapProcessor`` as
model-level modules alongside a standard ``Detect`` head. Current code moves both
modules *inside* a dedicated ``AnomalyDetect`` head. This script rebuilds a fresh
model with the new architecture and migrates the weights/state.

Example:
    python tools/convert_anomaly_ckpt.py \
        --input best_old.pt \
        --output best_converted.pt
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from ultralytics.nn.modules.anomaly import AnomalyMemoryBank, BboxMaskRenderer
from ultralytics.nn.tasks import YOLOAnomalyModel, temporary_modules

# Old checkpoints pickle modules/classes that were later renamed. Alias them
# temporarily during torch.load so the old pickle can unpickle; the migration
# below then normalizes attributes and bakes the current names into the saved
# object so the aliases are no longer needed at load time.
_LEGACY_ALIASES = {
    "modules": {
        "ultralytics.nn.modules.anomaly_v2": "ultralytics.nn.modules.anomaly",
        # "ultralytics.nn.modules.anomaly_v2_prior_augment": "ultralytics.nn.modules.anomaly",
        "ultralytics.nn.modules.anomaly_v2_prior_augment": "ultralytics.nn.modules.anomaly_prior_augment",
        "ultralytics.models.yolo.anomaly_v2": "ultralytics.models.yolo.anomaly",
    },
    "attributes": {
        "ultralytics.nn.tasks.YOLOAnomalyV2Model": "ultralytics.nn.tasks.YOLOAnomalyModel",
        "ultralytics.nn.modules.anomaly.BackboneMemoryBank": "ultralytics.nn.modules.anomaly.AnomalyMemoryBank",
    },
}


def _load_with_aliases(input_path: Path) -> Any:
    """Load a checkpoint, applying legacy aliases so old pickles can unpickle."""
    with temporary_modules(
        modules=_LEGACY_ALIASES["modules"], attributes=_LEGACY_ALIASES["attributes"]
    ):
        return torch.load(input_path, map_location="cpu", weights_only=False)


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


def _migrate_modules(m: YOLOAnomalyModel) -> None:
    """Apply module-level migrations to memory bank and bbox renderer."""
    for module in m.modules():
        if isinstance(module, BboxMaskRenderer):
            _migrate_bbox_mask_renderer(module)
        elif isinstance(module, AnomalyMemoryBank):
            _migrate_anomaly_memory_bank(module)

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


def _build_new_model(old_model: YOLOAnomalyModel) -> YOLOAnomalyModel:
    """Create a fresh YOLOAnomalyModel using the old YAML but with AnomalyDetect head."""
    yaml_cfg = deepcopy(old_model.yaml)

    # Normalize legacy ``anomaly_v2`` key to ``anomaly``.
    if "anomaly_v2" in yaml_cfg and "anomaly" not in yaml_cfg:
        yaml_cfg["anomaly"] = yaml_cfg.pop("anomaly_v2")

    # Infer the old fusion mid-channel from the pickled state so we preserve weights.
    old_state = old_model.state_dict()
    fusion_key = "heatmap_bias_fusion.conv.0.weight"
    inferred_c_mid = old_state.get(fusion_key, torch.empty(32, 1, 3, 3)).shape[0]

    for entry in yaml_cfg.get("head", []):
        if len(entry) >= 3 and entry[2] == "Detect":
            entry[2] = "AnomalyDetect"
            args = list(entry[3]) if len(entry) > 3 else []
            if not args:
                args = [yaml_cfg.get("nc", 80)]
            if len(args) < 2:
                args.append(inferred_c_mid)
            entry[3] = args

    new_model = YOLOAnomalyModel(
        cfg=yaml_cfg,
        ch=getattr(old_model, "ch", 3),
        nc=yaml_cfg.get("nc", 80),
        verbose=False,
    )

    # Carry over model-level attributes not baked into the YAML.
    for attr in ("p_drop", "seg_target_polygon", "mask_size", "bb_layers", "names"):
        if hasattr(old_model, attr):
            setattr(new_model, attr, getattr(old_model, attr))

    return new_model


def _remap_state_dict(old_state: dict[str, torch.Tensor], new_model: YOLOAnomalyModel) -> dict[str, torch.Tensor]:
    """Remap old state-dict keys into the new AnomalyDetect structure."""
    head_idx = len(new_model.model) - 1
    new_state: dict[str, torch.Tensor] = {}

    for key, value in old_state.items():
        if key.startswith("heatmap_bias_fusion."):
            new_key = f"model.{head_idx}.heatmap_bias_fusion.{key[len('heatmap_bias_fusion.'):]}"
        elif key.startswith("heatmap_processor."):
            new_key = f"model.{head_idx}.heatmap_processor.{key[len('heatmap_processor.'):]}"
        elif key == "memory_bank.memory_bank":
            # Very old checkpoints used ``memory_bank`` as the buffer name.
            new_key = "memory_bank.bank"
        else:
            new_key = key
        new_state[new_key] = value

    return new_state


def _fix_task(ckpt: dict[str, Any]) -> None:
    """Bake the current 'detect' task into the checkpoint args."""
    model = ckpt.get("model")
    if model is not None:
        model.task = "detect"
        if hasattr(model, "args") and isinstance(model.args, dict):
            model.args["task"] = "detect"

    train_args = ckpt.get("train_args")
    if isinstance(train_args, dict):
        train_args["task"] = "detect"


def convert_checkpoint(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Load an old checkpoint, migrate it to the AnomalyDetect-head format, and save."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    ckpt = _load_with_aliases(input_path)
    if isinstance(ckpt, YOLOAnomalyModel):
        old_model = ckpt
        ckpt = {"model": old_model, "updates": None, "optimizer": None, "args": None}
    else:
        old_model = ckpt.get("model")

    if not isinstance(old_model, YOLOAnomalyModel):
        raise TypeError(f"expected YOLOAnomalyModel, got {type(old_model).__name__}")

    old_model.eval()
    _migrate_modules(old_model)

    new_model = _build_new_model(old_model)

    # Copy weights with remapping for the heatmap modules that moved into the head.
    old_state = old_model.state_dict()
    remapped_state = _remap_state_dict(old_state, new_model)
    new_state = new_model.state_dict()

    matched = {}
    for key, value in remapped_state.items():
        if key in new_state and new_state[key].shape == value.shape:
            matched[key] = value
        else:
            print(f"  skipping incompatible/missing key: {key}")

    new_model.load_state_dict(matched, strict=False)

    # Replace the model in the checkpoint dict.
    ckpt["model"] = new_model
    _fix_task(ckpt)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_path)
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an old anomaly checkpoint to the current AnomalyDetect format.")
    parser.add_argument("--input", "-i", required=True, type=Path, help="Path to the old checkpoint.")
    parser.add_argument("--output", "-o", required=True, type=Path, help="Path to write the converted checkpoint.")
    args = parser.parse_args()

    convert_checkpoint(args.input, args.output)
    print(f"Converted {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
