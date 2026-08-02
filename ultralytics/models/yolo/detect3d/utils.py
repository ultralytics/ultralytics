# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Small runtime helpers shared by Detect3D train, validation, and prediction paths."""

from __future__ import annotations

import torch

from ultralytics.nn.modules.head import Detect3D


def find_detect3d_head(model: torch.nn.Module) -> Detect3D | None:
    """Return a native PyTorch Detect3D head through YOLO/AutoBackend wrappers, if one is available."""
    candidate = model
    visited: set[int] = set()
    for _ in range(5):
        if candidate is None or id(candidate) in visited:
            break
        visited.add(id(candidate))
        if isinstance(candidate, Detect3D):
            return candidate
        if hasattr(candidate, "_orig_mod"):
            candidate = candidate._orig_mod
            continue
        child = getattr(candidate, "model", None)
        if isinstance(child, (torch.nn.Sequential, torch.nn.ModuleList, list, tuple)):
            return child[-1] if child and isinstance(child[-1], Detect3D) else None
        candidate = child
    return None


def set_detect3d_quality_power(model: torch.nn.Module, power: float) -> bool:
    """Apply q3d score calibration to a native Detect3D model; return False for immutable exported backends."""
    head = find_detect3d_head(model)
    if head is None:
        return False
    head.set_quality3d_power(power)
    return True
