# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from ultralytics.nn.modules.transformer import MLP


class LinearPresenceHead(nn.Sequential):
    """Linear presence head for predicting the presence of classes in an image."""

    def __init__(self, d_model):
        """Initializes the LinearPresenceHead."""
        # a hack to make `LinearPresenceHead` compatible with old checkpoints
        super().__init__(nn.Identity(), nn.Identity(), nn.Linear(d_model, 1))

    def forward(self, hs, prompt, prompt_mask):
        """Forward pass of the presence head."""
        return super().forward(hs)


class MaskPredictor(nn.Module):
    """Predicts masks from object queries and pixel embeddings."""

    def __init__(self, hidden_dim, mask_dim):
        """Initializes the MaskPredictor."""
        super().__init__()
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(self, obj_queries, pixel_embed):
        """Predicts masks from object queries and pixel embeddings."""
        if len(obj_queries.shape) == 3:
            if pixel_embed.ndim == 3:
                # batch size was omitted
                mask_preds = torch.einsum("bqc,chw->bqhw", self.mask_embed(obj_queries), pixel_embed)
            else:
                mask_preds = torch.einsum("bqc,bchw->bqhw", self.mask_embed(obj_queries), pixel_embed)
        else:
            # Assumed to have aux masks
            if pixel_embed.ndim == 3:
                # batch size was omitted
                mask_preds = torch.einsum("lbqc,chw->lbqhw", self.mask_embed(obj_queries), pixel_embed)
            else:
                mask_preds = torch.einsum("lbqc,bchw->lbqhw", self.mask_embed(obj_queries), pixel_embed)

        return mask_preds


class SegmentationHead(nn.Module):
    """Segmentation head that predicts masks from backbone features and object queries."""

    def __init__(
        self,
        hidden_dim,
        upsampling_stages,
        use_encoder_inputs=False,
        aux_masks=False,
        no_dec=False,
        pixel_decoder=None,
        act_ckpt=False,
        shared_conv=False,
        compile_mode_pixel_decoder=None,
    ):
        """Initializes the SegmentationHead."""
        super().__init__()
        self.use_encoder_inputs = use_encoder_inputs
        self.aux_masks = aux_masks
        if pixel_decoder is not None:
            self.pixel_decoder = pixel_decoder
        else:
            self.pixel_decoder = PixelDecoder(
                hidden_dim,
                upsampling_stages,
                shared_conv=shared_conv,
                compile_mode=compile_mode_pixel_decoder,
            )
        self.no_dec = no_dec
        if no_dec:
            self.mask_predictor = nn.Conv2d(hidden_dim, 1, kernel_size=3, stride=1, padding=1)
        else:
            self.mask_predictor = MaskPredictor(hidden_dim, mask_dim=hidden_dim)

        self.act_ckpt = act_ckpt

        # used to update the output dictionary
        self.instance_keys = ["pred_masks"]

    def _embed_pixels(self, backbone_feats: list[torch.Tensor], encoder_hidden_states) -> torch.Tensor:
        """Embeds pixels using the pixel decoder."""
        if self.use_encoder_inputs:
            backbone_visual_feats = [bb_feat.clone() for bb_feat in backbone_feats]
            # Extract visual embeddings
            encoder_hidden_states = encoder_hidden_states.permute(1, 2, 0)
            spatial_dim = math.prod(backbone_feats[-1].shape[-2:])
            encoder_visual_embed = encoder_hidden_states[..., :spatial_dim].reshape(-1, *backbone_feats[-1].shape[1:])

            backbone_visual_feats[-1] = encoder_visual_embed
            if self.act_ckpt:
                pixel_embed = checkpoint.checkpoint(self.pixel_decoder, backbone_visual_feats, use_reentrant=False)
            else:
                pixel_embed = self.pixel_decoder(backbone_visual_feats)
        else:
            backbone_feats = [x for x in backbone_feats]
            pixel_embed = self.pixel_decoder(backbone_feats)
            if pixel_embed.shape[0] == 1:
                # For batch_size=1 training, we can avoid the indexing to save memory
                pixel_embed = pixel_embed.squeeze(0)
            else:
                pixel_embed = pixel_embed[[0], ...]
        return pixel_embed

    def forward(
        self,
        backbone_feats: list[torch.Tensor],
        obj_queries: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the SegmentationHead."""
        if self.use_encoder_inputs:
            assert encoder_hidden_states is not None

        pixel_embed = self._embed_pixels(backbone_feats=backbone_feats, encoder_hidden_states=encoder_hidden_states)

        if self.no_dec:
            mask_pred = self.mask_predictor(pixel_embed)
        elif self.aux_masks:
            mask_pred = self.mask_predictor(obj_queries, pixel_embed)
        else:
            mask_pred = self.mask_predictor(obj_queries[-1], pixel_embed)

        return {"pred_masks": mask_pred}


class PixelDecoder(nn.Module):
    """Pixel decoder module that upsamples backbone features."""

    def __init__(
        self,
        hidden_dim,
        num_upsampling_stages,
        interpolation_mode="nearest",
        shared_conv=False,
        compile_mode=None,
    ):
        """Initializes the PixelDecoder."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_upsampling_stages = num_upsampling_stages
        self.interpolation_mode = interpolation_mode
        conv_layers = []
        norms = []
        num_convs = 1 if shared_conv else num_upsampling_stages
        for _ in range(num_convs):
            conv_layers.append(nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1))
            norms.append(nn.GroupNorm(8, self.hidden_dim))

        self.conv_layers = nn.ModuleList(conv_layers)
        self.norms = nn.ModuleList(norms)
        self.shared_conv = shared_conv
        self.out_dim = self.conv_layers[-1].out_channels
        if compile_mode is not None:
            self.forward = torch.compile(self.forward, mode=compile_mode, dynamic=True, fullgraph=True)
            # Needed to make checkpointing happy. But we don't know if the module is checkpointed, so we disable it by default.
            torch._dynamo.config.optimize_ddp = False

    def forward(self, backbone_feats: list[torch.Tensor]):
        """Forward pass of the PixelDecoder."""
        prev_fpn = backbone_feats[-1]
        fpn_feats = backbone_feats[:-1]
        for layer_idx, bb_feat in enumerate(fpn_feats[::-1]):
            curr_fpn = bb_feat
            prev_fpn = curr_fpn + F.interpolate(prev_fpn, size=curr_fpn.shape[-2:], mode=self.interpolation_mode)
            if self.shared_conv:
                # only one conv layer
                layer_idx = 0
            prev_fpn = self.conv_layers[layer_idx](prev_fpn)
            prev_fpn = F.relu(self.norms[layer_idx](prev_fpn))

        return prev_fpn


class UniversalSegmentationHead(SegmentationHead):
    """This module handles semantic+instance segmentation."""

    def __init__(
        self,
        hidden_dim,
        upsampling_stages,
        pixel_decoder,
        aux_masks=False,
        no_dec=False,
        act_ckpt=False,
        presence_head: bool = False,
        dot_product_scorer=None,
        cross_attend_prompt=None,
    ):
        """Initializes the UniversalSegmentationHead."""
        super().__init__(
            hidden_dim=hidden_dim,
            upsampling_stages=upsampling_stages,
            use_encoder_inputs=True,
            aux_masks=aux_masks,
            no_dec=no_dec,
            pixel_decoder=pixel_decoder,
            act_ckpt=act_ckpt,
        )
        self.d_model = hidden_dim

        if dot_product_scorer is not None:
            assert presence_head, "Specifying a dot product scorer without a presence head is likely a mistake"

        self.presence_head = None
        if presence_head:
            self.presence_head = (
                dot_product_scorer if dot_product_scorer is not None else LinearPresenceHead(self.d_model)
            )

        self.cross_attend_prompt = cross_attend_prompt
        if self.cross_attend_prompt is not None:
            self.cross_attn_norm = nn.LayerNorm(self.d_model)

        self.semantic_seg_head = nn.Conv2d(self.pixel_decoder.out_dim, 1, kernel_size=1)
        self.instance_seg_head = nn.Conv2d(self.pixel_decoder.out_dim, self.d_model, kernel_size=1)

    def forward(
        self,
        backbone_feats: list[torch.Tensor],
        obj_queries: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        prompt: torch.Tensor = None,
        prompt_mask: torch.Tensor = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of the UniversalSegmentationHead."""
        assert encoder_hidden_states is not None
        bs = encoder_hidden_states.shape[1]

        if self.cross_attend_prompt is not None:
            tgt2 = self.cross_attn_norm(encoder_hidden_states)
            tgt2 = self.cross_attend_prompt(
                query=tgt2,
                key=prompt.to(tgt2.dtype),
                value=prompt.to(tgt2.dtype),
                key_padding_mask=prompt_mask,
                need_weights=False,
            )[0]
            encoder_hidden_states = tgt2 + encoder_hidden_states

        presence_logit = None
        if self.presence_head is not None:
            pooled_enc = encoder_hidden_states.mean(0)
            presence_logit = (
                self.presence_head(
                    pooled_enc.view(1, bs, 1, self.d_model),
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                )
                .squeeze(0)
                .squeeze(1)
            )

        pixel_embed = self._embed_pixels(backbone_feats=backbone_feats, encoder_hidden_states=encoder_hidden_states)

        instance_embeds = self.instance_seg_head(pixel_embed)

        if self.no_dec:
            mask_pred = self.mask_predictor(instance_embeds)
        elif self.aux_masks:
            mask_pred = self.mask_predictor(obj_queries, instance_embeds)
        else:
            mask_pred = self.mask_predictor(obj_queries[-1], instance_embeds)

        return {
            "pred_masks": mask_pred,
            "semantic_seg": self.semantic_seg_head(pixel_embed),
            "presence_logit": presence_logit,
        }
