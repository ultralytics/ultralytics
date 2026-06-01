# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
import logging
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn.init
from torch import Tensor, nn

from .layers import LayerScale, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, SelfAttentionBlock, SwiGLUFFN
from .utils import named_apply


class Weights(Enum):
    LVD1689M = "LVD1689M"
    SAT493M = "SAT493M"

configs = {
    'dinov3_vits16': {
        'img_size': 224,
        'patch_size': 16,
        'in_chans': 3,
        'pos_embed_rope_base': 100,
        'pos_embed_rope_normalize_coords': "separate",
        'pos_embed_rope_rescale_coords': 2,
        'pos_embed_rope_dtype': "fp32",
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'ffn_ratio': 4,
        'qkv_bias': True,
        'drop_path_rate': 0.0,
        'layerscale_init': 1.0e-05,
        'norm_layer': "layernormbf16",
        'ffn_layer': "mlp",
        'ffn_bias': True,
        'proj_bias': True,
        'n_storage_tokens': 4,
        'mask_k_bias': True,
        'pretrained': True,
        'weights': Weights.LVD1689M,
        'compact_arch_name': "vits",
        'check_hash': False,
    },

    'dinov3_vits16plus': {
        "img_size": 224,
        "patch_size": 16,
        "in_chans": 3,
        "pos_embed_rope_base": 100,
        "pos_embed_rope_normalize_coords": "separate",
        "pos_embed_rope_rescale_coords": 2,
        "pos_embed_rope_dtype": "fp32",
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "ffn_ratio": 6,
        "qkv_bias": True,
        "drop_path_rate": 0.0,
        "layerscale_init": 1.0e-05,
        "norm_layer": "layernormbf16",
        "ffn_layer": "swiglu",
        "ffn_bias": True,
        "proj_bias": True,
        "n_storage_tokens": 4,
        "mask_k_bias": True,
        "pretrained": True,
        "weights": Weights.LVD1689M,
        "compact_arch_name": "vitsplus",
        "check_hash": False,
    },

    'dinov3_vitb16': {
        'img_size': 224,
        'patch_size': 16,
        'in_chans': 3,
        'pos_embed_rope_base': 100,
        'pos_embed_rope_normalize_coords': "separate",
        'pos_embed_rope_rescale_coords': 2,
        'pos_embed_rope_dtype': "fp32",
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'ffn_ratio': 4,
        'qkv_bias': True,
        'drop_path_rate': 0.0,
        'layerscale_init': 1.0e-05,
        'norm_layer': "layernormbf16",
        'ffn_layer': "mlp",
        'ffn_bias': True,
        'proj_bias': True,
        'n_storage_tokens': 4,
        'mask_k_bias': True,
        'pretrained': True,
        'weights': Weights.LVD1689M,
        'compact_arch_name': "vitb",
        'check_hash': False,
    },

    'dinov3_vitl16': {
        'img_size': 224,
        'patch_size': 16,
        'in_chans': 3,
        'pos_embed_rope_base': 100,
        'pos_embed_rope_normalize_coords': "separate",
        'pos_embed_rope_rescale_coords': 2,
        'pos_embed_rope_dtype': "fp32",
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'ffn_ratio': 4,
        'qkv_bias': True,
        'drop_path_rate': 0.0,
        'layerscale_init': 1.0e-05,
        'norm_layer': "layernormbf16",
        'ffn_layer': "mlp",
        'ffn_bias': True,
        'proj_bias': True,
        'n_storage_tokens': 4,
        'mask_k_bias': True,
        'pretrained': True,
        'weights': Weights.LVD1689M,
        'compact_arch_name': "vitl",
        'check_hash': False,
    },

    'dinov3_vith16plus': {
        'img_size': 224,
        'patch_size': 16,
        'in_chans': 3,
        'pos_embed_rope_base': 100,
        'pos_embed_rope_normalize_coords': "separate",
        'pos_embed_rope_rescale_coords': 2,
        'pos_embed_rope_dtype': "fp32",
        'embed_dim': 1280,
        'depth': 32,
        'num_heads': 20,
        'ffn_ratio': 6.0,
        'qkv_bias': True,
        'drop_path_rate': 0.0,
        'layerscale_init': 1.0e-05,
        'norm_layer': "layernormbf16",
        'ffn_layer': "swiglu",
        'ffn_bias': True,
        'proj_bias': True,
        'n_storage_tokens': 4,
        'mask_k_bias': True,
        'pretrained': True,
        'weights': Weights.LVD1689M,
        'compact_arch_name': "vithplus",
        'check_hash': False,
    },

    'dinov3_vit7b16': {
        'img_size': 224,
        'patch_size': 16,
        'in_chans': 3,
        'pos_embed_rope_base': 100,
        'pos_embed_rope_normalize_coords': "separate",
        'pos_embed_rope_rescale_coords': 2,
        'pos_embed_rope_dtype': "fp32",
        'embed_dim': 4096,
        'depth': 40,
        'num_heads': 32,
        'ffn_ratio': 3,
        'qkv_bias': False,
        'drop_path_rate': 0.0,
        'layerscale_init': 1.0e-05,
        'norm_layer': "layernormbf16",
        'ffn_layer': "swiglu64",
        'ffn_bias': True,
        'proj_bias': True,
        'n_storage_tokens': 4,
        'mask_k_bias': True,
        'pretrained': True,
        'weights': Weights.LVD1689M,
        'compact_arch_name': "vit7b",
        'check_hash': False,
    }
}

logger = logging.getLogger("dinov3")

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
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
    if isinstance(module, RMSNorm):
        module.reset_parameters()


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        name,
    ):
        super().__init__()

        img_size                        = configs[name]['img_size']
        patch_size                      = configs[name]['patch_size']
        in_chans                        = configs[name]['in_chans']
        pos_embed_rope_min_period       = None
        pos_embed_rope_max_period       = None
        pos_embed_rope_shift_coords     = None
        pos_embed_rope_jitter_coords    = None
        pos_embed_rope_rescale_coords   = None
        pos_embed_rope_base             = configs[name]['pos_embed_rope_base']
        pos_embed_rope_normalize_coords = configs[name]['pos_embed_rope_normalize_coords']
        pos_embed_rope_rescale_coords   = configs[name]['pos_embed_rope_rescale_coords']
        pos_embed_rope_dtype            = configs[name]['pos_embed_rope_dtype']
        embed_dim                       = configs[name]['embed_dim']
        depth                           = configs[name]['depth']
        num_heads                       = configs[name]['num_heads']
        ffn_ratio                       = configs[name]['ffn_ratio']
        qkv_bias                        = configs[name]['qkv_bias']
        drop_path_rate                  = configs[name]['drop_path_rate']
        layerscale_init                 = configs[name]['layerscale_init']
        norm_layer                      = configs[name]['norm_layer']
        ffn_layer                       = configs[name]['ffn_layer']
        ffn_bias                        = configs[name]['ffn_bias']
        proj_bias                       = configs[name]['proj_bias']
        n_storage_tokens                = configs[name]['n_storage_tokens']
        mask_k_bias                     = configs[name]['mask_k_bias']
        pretrained                      = configs[name]['pretrained']
        weights                         = configs[name]['weights']
        compact_arch_name               = configs[name]['compact_arch_name']
        check_hash                      = configs[name]['check_hash']
        untie_cls_and_patch_norms       = False
        untie_global_and_local_cls_norm = False
        device                          = None

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new")
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new")
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

        self.init_weights()

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, (H, W)

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(self, *args, is_training: bool = False, **kwargs) -> List[Dict[str, Tensor]] | Tensor:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])
