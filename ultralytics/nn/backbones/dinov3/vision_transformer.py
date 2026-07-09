# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from functools import partial
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn.init
from torch import Tensor, nn

from .layers import LayerScale, Mlp, PatchEmbed, RopePositionEmbedding, SelfAttentionBlock, SwiGLUFFN
from .utils import named_apply


configs = {
    "dinov3_vits16": {
        "patch_size": 16,
        "in_chans": 3,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "ffn_ratio": 4,
        "ffn_layer": "mlp",
    },
    "dinov3_vits16plus": {
        "patch_size": 16,
        "in_chans": 3,
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "ffn_ratio": 6,
        "ffn_layer": "swiglu",
    },
    "dinov3_vitb16": {
        "patch_size": 16,
        "in_chans": 3,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "ffn_ratio": 4,
        "ffn_layer": "mlp",
    },
    "dinov3_vitl16": {
        "patch_size": 16,
        "in_chans": 3,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "ffn_ratio": 4,
        "ffn_layer": "mlp",
    },
    "dinov3_vith16plus": {
        "patch_size": 16,
        "in_chans": 3,
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 20,
        "ffn_ratio": 6.0,
        "ffn_layer": "swiglu",
    },
    "dinov3_vit7b16": {
        "patch_size": 16,
        "in_chans": 3,
        "embed_dim": 4096,
        "depth": 40,
        "num_heads": 32,
        "ffn_ratio": 3,
        "qkv_bias": False,
        "ffn_layer": "swiglu64",
    },
}

# Shared hyperparameters across all six ViT variants. Kept as module-level constants because they were
# duplicated verbatim in every config entry and never varied.
_ROPE_BASE = 100
_ROPE_NORMALIZE_COORDS = "separate"
_ROPE_RESCALE_COORDS = 2
_LAYERSCALE_INIT = 1.0e-5
_N_STORAGE_TOKENS = 4
_MASK_K_BIAS = True

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu64": partial(SwiGLUFFN, align_to=64),
}


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()


class DinoVisionTransformer(nn.Module):
    def __init__(self, name):
        super().__init__()

        cfg = configs[name]
        patch_size = cfg["patch_size"]
        in_chans = cfg["in_chans"]
        embed_dim = cfg["embed_dim"]
        depth = cfg["depth"]
        num_heads = cfg["num_heads"]
        ffn_ratio = cfg["ffn_ratio"]
        qkv_bias = cfg.get("qkv_bias", True)
        ffn_layer = cfg["ffn_layer"]

        norm_layer_cls = partial(nn.LayerNorm, eps=1e-5)
        ffn_layer_cls = ffn_layer_dict[ffn_layer]

        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.n_storage_tokens = _N_STORAGE_TOKENS

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.storage_tokens = nn.Parameter(torch.empty(1, _N_STORAGE_TOKENS, embed_dim))
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim))

        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=_ROPE_BASE,
            normalize_coords=_ROPE_NORMALIZE_COORDS,
            rescale_coords=_ROPE_RESCALE_COORDS,
            dtype=torch.float32,
        )

        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=True,
                    ffn_bias=True,
                    norm_layer=norm_layer_cls,
                    act_layer=nn.GELU,
                    ffn_layer=ffn_layer_cls,
                    init_values=_LAYERSCALE_INIT,
                    mask_k_bias=_MASK_K_BIAS,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer_cls(embed_dim)

        self.init_weights()

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int, int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                self.storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )
        return x, (H, W)

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: Union[int, Sequence]) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            rope_sincos = self.rope_embed(H=H, W=W)
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,
        return_class_token: bool = False,
    ) -> Tuple:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)
