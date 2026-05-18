"""DEIMv2 DINOv3 backbone adapter."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER

from .dinov3 import DinoVisionTransformer, WindowedDinoVisionTransformer
from .dinov3.vision_transformer import configs as dinov3_configs

__all__ = ["DINOv3", "DINOv3STAs"]


# Official DINOv3 checkpoint hash suffixes (from facebookresearch/dinov3 hub/backbones.py).
DINOV3_HASH_SUFFIX = {
    "vits16": "08c60483",
    "vits16plus": "4057cbaa",
    "vitb16": "73cec8be",
    "vitl16": "8aa4cbdd",
    "vith16plus": "7c1da9a5",
    "vit7b16": "a955f4ea",
}


class SpatialPriorModulev2(nn.Module):
    """Lite spatial prior branch used by DEIMv2."""

    def __init__(self, inplanes: int = 16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
        )
        self.conv3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
        )
        self.conv4 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        return c2, c3, c4


class DINOv3STAs(nn.Module):
    """DEIMv2 DINOv3 + STA fusion backbone returning three pyramid features."""

    def __init__(
        self,
        name: str = "dinov3_vits16",
        pretrained: bool = True,
        interaction_indexes: tuple[int, ...] | list[int] = (5, 8, 11),
        finetune: bool = True,
        patch_size: int = 16,
        use_sta: bool = True,
        conv_inplane: int = 32,
        hidden_dim: int = 224,
        qk_layernorm: bool = False,
        num_windows: int = 1,
        global_block_indexes: list[int] | None = None,
    ):
        super().__init__()
        if num_windows > 1:
            self.dinov3 = WindowedDinoVisionTransformer(
                name=name,
                qk_layernorm=qk_layernorm,
                num_windows=num_windows,
                # Default: interaction_indexes are the global attention layers
                global_block_indexes=global_block_indexes or list(interaction_indexes),
            )
        else:
            self.dinov3 = DinoVisionTransformer(name=name, qk_layernorm=qk_layernorm)
        self.interaction_indexes = list(interaction_indexes)
        self.patch_size = patch_size
        self._last_load_source = "none"

        loaded = False
        if pretrained:
            loaded = self._autoload_pretrained(name=name)

        if loaded:
            LOGGER.info(f"DINOv3 pretrained weights loaded from {self._last_load_source}.")
        elif pretrained:
            LOGGER.warning("DINOv3 backbone weights were not loaded. Backbone will train from scratch.")

        # Ensure K-bias masking is valid even when checkpoint lacks explicit bias_mask.
        self._sanitize_qkv_bias_mask()

        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

        self.use_sta = use_sta
        if use_sta:
            self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        else:
            conv_inplane = 0

        embed_dim = self.dinov3.embed_dim
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
        h_base = x.shape[2] // 16
        w_base = x.shape[3] // 16
        bs = x.shape[0]

        all_layers = self.dinov3.get_intermediate_layers(x, n=self.interaction_indexes, return_class_token=True)
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

    def _autoload_pretrained(self, name: str) -> bool:
        """Auto-load pretrained weights from official DINOv3 URLs only."""
        return self._load_from_official_dinov3(name=name)

    def _load_from_official_dinov3(self, name: str) -> bool:
        """Download/load DINOv3 checkpoint from official Meta URL layout."""
        explicit_url = os.getenv("DEIMV2_DINOV3_URL")
        if explicit_url:
            return self._load_from_url(explicit_url)

        url = self._make_official_dinov3_url(name=name)
        if not url:
            return False
        return self._load_from_url(url)

    def _make_official_dinov3_url(self, name: str) -> str | None:
        """Build official checkpoint URL from local DINOv3 config."""
        cfg = dinov3_configs.get(name)
        if not cfg:
            return None

        patch_size = cfg.get("patch_size")
        weights = cfg.get("weights")
        if not patch_size or weights is None:
            return None

        weight_name = str(getattr(weights, "value", weights)).lower()
        # Use the config key-derived arch slug to preserve official ordering, e.g. vits16plus not vitsplus16.
        model_arch = name.removeprefix("dinov3_")
        hash_suffix = DINOV3_HASH_SUFFIX.get(model_arch)
        if not hash_suffix:
            return None

        # Match official hub naming:
        #   {base}/dinov3_{arch}/dinov3_{arch}_pretrain_{weights}-{hash}.pth
        model_dir = f"dinov3_{model_arch}"
        checkpoint = f"dinov3_{model_arch}_pretrain_{weight_name}-{hash_suffix}"
        base_url = os.getenv("DINOV3_BASE_URL", "https://dl.fbaipublicfiles.com/dinov3").rstrip("/")
        return f"{base_url}/{model_dir}/{checkpoint}.pth"

    def _load_from_url(self, url: str) -> bool:
        """Download checkpoint URL into torch cache and load."""
        try:
            model_dir = os.getenv("TORCH_MODEL_ZOO", os.path.join(torch.hub.get_dir(), "checkpoints"))
            filename = Path(urlparse(url).path).name
            cache_path = Path(model_dir).expanduser() / filename if filename else None

            state = None
            from_cache = False
            if cache_path and cache_path.exists():
                LOGGER.info(f"Found cached DINOv3 checkpoint: {cache_path}")
                try:
                    state = torch.load(cache_path, map_location="cpu")
                    from_cache = True
                except Exception as e:
                    LOGGER.warning(f"Failed reading cached checkpoint {cache_path}: {e}. Re-downloading.")

            if state is None:
                LOGGER.info(f"Downloading DINOv3 checkpoint: {url}")
                state = torch.hub.load_state_dict_from_url(
                    url=url,
                    map_location="cpu",
                    model_dir=model_dir,
                    progress=True,
                    check_hash=False,
                    file_name=filename or None,
                )
                if cache_path:
                    LOGGER.info(f"Downloaded DINOv3 checkpoint to: {cache_path}")

            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            msg = self.dinov3.load_state_dict(state, strict=False)
            self._log_load_result(msg.missing_keys, msg.unexpected_keys)
            if from_cache and cache_path:
                self._last_load_source = f"cache: {cache_path}"
                LOGGER.info(f"Loaded DINOv3 from cache: {cache_path}")
            else:
                self._last_load_source = f"url: {url}"
                LOGGER.info(f"Loaded DINOv3 from official URL: {url}")
            return True
        except HTTPError as e:
            if e.code == 403:
                LOGGER.warning(
                    "Official DINOv3 URL is access-gated (HTTP 403). "
                    "Set DEIMV2_DINOV3_URL to your granted download URL from DINOv3."
                )
            else:
                LOGGER.warning(f"Official DINOv3 download/load failed ({url}): {e}")
            return False
        except Exception as e:
            LOGGER.warning(f"Official DINOv3 download/load failed ({url}): {e}")
            return False

    def _sanitize_qkv_bias_mask(self):
        """Ensure qkv.bias_mask exists and masks K branch with zeros (q/v ones)."""
        for blk in self.dinov3.blocks:
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

    @staticmethod
    def _log_load_result(missing: list[str], unexpected: list[str]):
        if missing:
            LOGGER.info(f"DINOv3 load missing keys: {len(missing)}")
        if unexpected:
            LOGGER.info(f"DINOv3 load unexpected keys: {len(unexpected)}")


class DINOv3(DINOv3STAs):
    """Native DINOv3 backbone wrapper returning reshaped intermediate ViT features without STA."""

    def __init__(
        self,
        name: str = "dinov3_vits16",
        pretrained: bool = True,
        out_indices: tuple[int, ...] | list[int] = (11,),
        finetune: bool = True,
        patch_size: int = 16,
        qk_layernorm: bool = False,
        num_windows: int = 1,
        global_block_indexes: list[int] | None = None,
    ):
        nn.Module.__init__(self)
        if num_windows > 1:
            self.dinov3 = WindowedDinoVisionTransformer(
                name=name,
                qk_layernorm=qk_layernorm,
                num_windows=num_windows,
                global_block_indexes=global_block_indexes or list(out_indices),
            )
        else:
            self.dinov3 = DinoVisionTransformer(name=name, qk_layernorm=qk_layernorm)
        self.out_indices = list(out_indices)
        self.patch_size = patch_size
        self._last_load_source = "none"

        loaded = False
        if pretrained:
            loaded = self._autoload_pretrained(name=name)

        if loaded:
            LOGGER.info(f"DINOv3 pretrained weights loaded from {self._last_load_source}.")
        elif pretrained:
            LOGGER.warning("DINOv3 backbone weights were not loaded. Backbone will train from scratch.")

        self._sanitize_qkv_bias_mask()

        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

        self.out_channels = [self.dinov3.embed_dim for _ in self.out_indices]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return native DINOv3 intermediate patch features as B,C,H,W maps."""
        return list(self.dinov3.get_intermediate_layers(x, n=self.out_indices, reshape=True))
