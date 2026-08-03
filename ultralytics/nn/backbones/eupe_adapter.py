"""DEIMv2 EUPE backbone adapter."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["EUPEConvNeXt"]

_FP16_LN_PATCH_FLAG = "_fp16_safe_layernorm_patched"


def _import_eupe_backbones(repo_dir: str | None = None) -> ModuleType:
    """Import EUPE hub backbones from the installed package or a local repository checkout."""
    try:
        return importlib.import_module("eupe.hub.backbones")
    except ModuleNotFoundError:
        pass
    # Local checkouts: explicit repo_dir, then EUPE_REPO_DIR, then the sibling and ~/git/{ultralytics,EUPE} layouts.
    candidates = (
        repo_dir,
        os.getenv("EUPE_REPO_DIR"),
        Path(__file__).resolve().parents[3].parent / "EUPE",
        "~/git/EUPE",
    )
    for value in candidates:
        path = Path(value).expanduser() if value else None
        if path and (path / "eupe" / "hub" / "backbones.py").is_file():
            sys.path.insert(0, str(path))
            return importlib.import_module("eupe.hub.backbones")
    raise ModuleNotFoundError("EUPE not found, install the package or set EUPE_REPO_DIR to a local EUPE checkout.")


_import_eupe_backbones()  # resolve EUPE on sys.path at import so pickled EUPEConvNeXt checkpoints unpickle


def _fp16_safe_layernorm(self, x: torch.Tensor) -> torch.Tensor:
    """FP16-safe replacement for EUPE's ConvNeXt ``channels_first`` LayerNorm.

    EUPE's ConvNeXt copies the original Meta implementation, whose channels_first LayerNorm computes mean and variance
    manually in the input dtype. ConvNeXt activations routinely exceed FP16's ~65504 ceiling once squared, so
    model.half() silently corrupts the stem and downsample norms (inf/NaN, not a crash). F.layer_norm accumulates the
    reduction in FP32 internally even for FP16 input, so routing through it is overflow-safe with no explicit upcast (an
    explicit .float() is pure overhead for FP16-identical output).
    """
    if self.data_format == "channels_last":
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    out = F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps)
    return out.permute(0, 3, 1, 2)


def _patch_eupe_layernorms(module: nn.Module) -> None:
    """Patch EUPE ConvNeXt LayerNorm classes for FP16-safe reduction, at class level so pickled models get it too."""
    for m in module.modules():
        cls = type(m)  # EUPE's ConvNeXt LayerNorm is the only module exposing data_format and normalized_shape
        if hasattr(m, "data_format") and hasattr(m, "normalized_shape") and not getattr(cls, _FP16_LN_PATCH_FLAG, 0):
            cls.forward = _fp16_safe_layernorm
            setattr(cls, _FP16_LN_PATCH_FLAG, True)


class EUPEConvNeXt(nn.Module):
    """EUPE ConvNeXt backbone returning native pyramid features.

    Attributes:
        eupe (nn.Module): ConvNeXt trunk built from an EUPE hub factory.
        out_indices (list[int]): ConvNeXt stage indices to return, 1/2/3 giving strides 8/16/32.
        out_channels (list[int]): Channel count of each returned feature map.

    Examples:
        >>> backbone = EUPEConvNeXt("eupe_convnext_small", pretrained=False)
        >>> p3, p4, p5 = backbone(torch.zeros(1, 3, 640, 640))
    """

    def __init__(
        self,
        name: str = "eupe_convnext_tiny",
        pretrained: bool = True,
        out_indices: tuple[int, ...] | list[int] = (1, 2, 3),
        finetune: bool = True,
        weights: str | None = None,
        repo_dir: str | None = None,
    ):
        """Initialize an EUPE ConvNeXt backbone.

        Args:
            name (str): EUPE hub factory name, i.e. 'eupe_convnext_small'.
            pretrained (bool): Load EUPE pretrained weights.
            out_indices (tuple | list): ConvNeXt stage indices to return.
            finetune (bool): Train the backbone, otherwise freeze it in eval mode.
            weights (str, optional): Weights alias ('LVD1689M', 'SAT493M') or a path/URL to a checkpoint.
            repo_dir (str, optional): Local EUPE repository to import when the package is not installed.
        """
        super().__init__()
        backbones = _import_eupe_backbones(repo_dir)
        kwargs = {"weights": getattr(backbones.Weights, weights.upper(), weights)} if weights else {}
        self.eupe = getattr(backbones, name)(pretrained=pretrained, **kwargs)
        self.out_indices = list(out_indices)
        self.out_channels = [self.eupe.embed_dims[i] for i in self.out_indices]
        if not finetune:
            self.eupe.eval().requires_grad_(False)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass producing the selected ConvNeXt feature maps."""
        # Patch lazily: a model restored from a pickled .pt never re-runs __init__, so forward is the reliable hook.
        if not getattr(type(self), _FP16_LN_PATCH_FLAG, False):
            _patch_eupe_layernorms(self.eupe)
            setattr(type(self), _FP16_LN_PATCH_FLAG, True)
        return list(self.eupe.get_intermediate_layers(x, n=self.out_indices, reshape=True))
