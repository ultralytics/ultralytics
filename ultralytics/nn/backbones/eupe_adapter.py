"""DEIMv2 EUPE backbone adapter."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from pathlib import Path
from types import ModuleType
from urllib.error import HTTPError

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER

from .dinov3_adapter import SpatialPriorModulev2

__all__ = ["EUPEConvNeXt", "EUPESTAs"]


def _candidate_eupe_repos(repo_dir: str | None = None) -> list[Path]:
    """Return likely local EUPE repository locations."""
    candidates = []
    for value in (repo_dir, os.getenv("EUPE_REPO_DIR")):
        if value:
            candidates.append(Path(value).expanduser())

    # Common local layout for this workspace: /Users/esat/git/{ultralytics,EUPE}.
    repo_root = Path(__file__).resolve().parents[3]
    candidates.append(repo_root.parent / "EUPE")
    return candidates


def _import_eupe_backbones(repo_dir: str | None = None) -> ModuleType:
    """Import EUPE hub backbones from the environment or a local checkout."""
    try:
        return importlib.import_module("eupe.hub.backbones")
    except ModuleNotFoundError as original_error:
        for path in _candidate_eupe_repos(repo_dir):
            if (path / "eupe" / "hub" / "backbones.py").is_file():
                path_str = str(path)
                if path_str not in sys.path:
                    sys.path.insert(0, path_str)
                try:
                    return importlib.import_module("eupe.hub.backbones")
                except ModuleNotFoundError:
                    continue
        raise ModuleNotFoundError(
            "EUPE package was not found. Install EUPE or set EUPE_REPO_DIR to the local EUPE repository."
        ) from original_error


def _normalize_weights(backbones: ModuleType, weights: str | None):
    """Convert optional weight aliases to EUPE hub values."""
    weights = weights or os.getenv("DEIMV2_EUPE_WEIGHTS")
    if weights in {"", "none", "None", "null", "NULL"}:
        return None
    if weights is None:
        return None

    weights_enum = getattr(backbones, "Weights", None)
    if weights_enum is not None:
        upper = str(weights).upper()
        if hasattr(weights_enum, upper):
            return getattr(weights_enum, upper)
    return weights


def _make_eupe_model(
    name: str,
    pretrained: bool,
    weights: str | None,
    repo_dir: str | None,
    check_hash: bool = False,
) -> nn.Module:
    """Build an EUPE backbone, falling back to random init if pretrained loading fails."""
    backbones = _import_eupe_backbones(repo_dir)
    if not hasattr(backbones, name):
        raise ValueError(f"Unknown EUPE backbone '{name}'.")

    factory = getattr(backbones, name)
    normalized_weights = _normalize_weights(backbones, weights)
    kwargs = {}
    if normalized_weights is not None:
        kwargs["weights"] = normalized_weights
    if "check_hash" in inspect.signature(factory).parameters:
        kwargs["check_hash"] = check_hash

    if not pretrained:
        return factory(pretrained=False, **kwargs)

    try:
        model = factory(pretrained=True, **kwargs)
        LOGGER.info(f"EUPE pretrained weights loaded for {name}.")
        return model
    except HTTPError as e:
        LOGGER.warning(f"EUPE pretrained weights were not loaded for {name}: {e}. Backbone will train from scratch.")
    except Exception as e:
        LOGGER.warning(f"EUPE pretrained weights were not loaded for {name}: {e}. Backbone will train from scratch.")

    return factory(pretrained=False, **kwargs)


class EUPESTAs(nn.Module):
    """EUPE ViT + STA fusion backbone returning three pyramid features."""

    def __init__(
        self,
        name: str = "eupe_vits16",
        pretrained: bool = True,
        interaction_indexes: tuple[int, ...] | list[int] = (5, 8, 11),
        finetune: bool = True,
        patch_size: int = 16,
        use_sta: bool = True,
        conv_inplane: int = 32,
        hidden_dim: int = 224,
        weights: str | None = None,
        repo_dir: str | None = None,
        check_hash: bool = False,
    ):
        super().__init__()
        self.interaction_indexes = list(interaction_indexes)
        self.patch_size = patch_size
        self._last_load_source = "none"

        self.eupe = _make_eupe_model(name, pretrained, weights, repo_dir, check_hash)
        self._sanitize_qkv_bias_mask()

        if not finetune:
            self.eupe.eval()
            self.eupe.requires_grad_(False)

        self.use_sta = use_sta
        if use_sta:
            self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        else:
            conv_inplane = 0

        embed_dim = self.eupe.embed_dim
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(embed_dim + conv_inplane * 2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Conv2d(embed_dim + conv_inplane * 4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Conv2d(embed_dim + conv_inplane * 4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            ]
        )
        self.norms = nn.ModuleList(
            [nn.SyncBatchNorm(hidden_dim), nn.SyncBatchNorm(hidden_dim), nn.SyncBatchNorm(hidden_dim)]
        )
        self.out_channels = [hidden_dim, hidden_dim, hidden_dim]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass producing P3, P4 and P5 features."""
        h_base = x.shape[2] // self.patch_size
        w_base = x.shape[3] // self.patch_size
        bs = x.shape[0]

        all_layers = self.eupe.get_intermediate_layers(x, n=self.interaction_indexes, return_class_token=True)
        if len(all_layers) == 1:
            all_layers = [all_layers[0], all_layers[0], all_layers[0]]

        sem_feats = []
        num_scales = len(all_layers) - 2
        for i, sem_layer in enumerate(all_layers):
            feat, _ = sem_layer
            sem = feat.transpose(1, 2).view(bs, -1, h_base, w_base).contiguous()
            resize_h = int(h_base * 2 ** (num_scales - i))
            resize_w = int(w_base * 2 ** (num_scales - i))
            sem = F.interpolate(sem, size=[resize_h, resize_w], mode="bilinear", align_corners=False)
            sem_feats.append(sem)

        if self.use_sta:
            detail_feats = self.sta(x)
            fused_feats = [torch.cat([s, d], dim=1) for s, d in zip(sem_feats, detail_feats)]
        else:
            fused_feats = sem_feats

        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        c4 = self.norms[2](self.convs[2](fused_feats[2]))
        return [c2, c3, c4]

    def _sanitize_qkv_bias_mask(self):
        """Ensure masked K bias stays zero even if a checkpoint omits bias_mask."""
        for blk in getattr(self.eupe, "blocks", []):
            qkv = getattr(getattr(blk, "attn", None), "qkv", None)
            if qkv is None or getattr(qkv, "bias", None) is None:
                continue
            bias = qkv.bias
            c = bias.numel() // 3
            if hasattr(qkv, "bias_mask"):
                with torch.no_grad():
                    qkv.bias_mask[:c].fill_(1.0)
                    qkv.bias_mask[c : 2 * c].fill_(0.0)
                    qkv.bias_mask[2 * c :].fill_(1.0)
                    bias[c : 2 * c].zero_()
                    bias.copy_(torch.nan_to_num(bias))


_FP16_LN_PATCH_FLAG = "_fp16_safe_layernorm_patched"


def _fp16_safe_layernorm(self, x: torch.Tensor) -> torch.Tensor:
    """FP16-safe replacement for EUPE's ConvNeXt ``channels_first`` LayerNorm.

    EUPE's ConvNeXt copies the original Meta implementation, whose ``channels_first``
    LayerNorm computes mean/variance manually in the input dtype. ConvNeXt activations
    routinely exceed FP16's ~65504 ceiling once squared, so ``model.half()`` silently
    corrupts the stem and downsample norms (inf/NaN, not a crash) and tanks accuracy.

    ``F.layer_norm`` accumulates the reduction in FP32 internally even for FP16 input,
    so routing through it is overflow-safe with no explicit upcast (an explicit
    ``.float()`` is pure overhead — ~30% slower in benchmarks for FP16-identical output).
    """
    if self.data_format == "channels_last":
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    out = F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps)
    return out.permute(0, 3, 1, 2)


def _patch_eupe_layernorms(module: nn.Module) -> int:
    """Idempotently patch EUPE ConvNeXt LayerNorm classes in ``module`` for FP16-safe reduction.

    Patches at the class level (not per instance) so the fix also applies to models loaded
    from a pickled ``.pt`` checkpoint, where ``EUPEConvNeXt.__init__`` is never re-run.
    """
    patched = 0
    for m in module.modules():
        # EUPE's custom ConvNeXt LayerNorm exposes ``data_format`` + ``normalized_shape``.
        if hasattr(m, "data_format") and hasattr(m, "normalized_shape"):
            cls = type(m)
            if not getattr(cls, _FP16_LN_PATCH_FLAG, False):
                cls.forward = _fp16_safe_layernorm
                setattr(cls, _FP16_LN_PATCH_FLAG, True)
                patched += 1
    return patched


class EUPEConvNeXt(nn.Module):
    """EUPE ConvNeXt backbone returning native pyramid features."""

    def __init__(
        self,
        name: str = "eupe_convnext_tiny",
        pretrained: bool = True,
        out_indices: tuple[int, ...] | list[int] = (1, 2, 3),
        finetune: bool = True,
        weights: str | None = None,
        repo_dir: str | None = None,
    ):
        super().__init__()
        self.eupe = _make_eupe_model(name, pretrained, weights, repo_dir)
        _patch_eupe_layernorms(self.eupe)
        self.out_indices = list(out_indices)

        if not finetune:
            self.eupe.eval()
            self.eupe.requires_grad_(False)

        embed_dims = getattr(self.eupe, "embed_dims", None)
        self.out_channels = [embed_dims[i] for i in self.out_indices] if embed_dims is not None else []

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass producing selected ConvNeXt feature maps."""
        # Lazily ensure the FP16-safe LayerNorm patch is applied. Needed because models loaded
        # from a pickled .pt checkpoint never re-run __init__, so this is the reliable hook.
        if not getattr(type(self), _FP16_LN_PATCH_FLAG, False):
            if _patch_eupe_layernorms(self.eupe):
                setattr(type(self), _FP16_LN_PATCH_FLAG, True)
        return list(self.eupe.get_intermediate_layers(x, n=self.out_indices, reshape=True))
