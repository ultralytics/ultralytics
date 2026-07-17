# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""SAM3 ONNX/TensorRT export pipeline.

Exports SAM3SemanticModel as 3 separate ONNX modules:
  1. Vision Encoder  - ViT backbone + FPN neck
  2. Text Encoder    - CLIP text encoder + projection to 256-dim
  3. Decoder         - DETR encoder-decoder + mask heads

TensorRT FP16 compatibility fixes applied during export:
  - ViT attention: separate Q/K/V projections, SDPA with pre-computed scale,
    rotate_half RoPE, static window partition shapes
  - DETR attention: nn.MultiheadAttention replaced with manual attention using
    pre-computed scale constant (eliminates dynamic Sqrt ops)
  - GELU: nn.GELU(approximate='tanh') traces to native ONNX Gelu op (opset 20)
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import LOGGER


# ---------------------------------------------------------------------------
# TRT-compatible sine position encoding
# ---------------------------------------------------------------------------


def _compute_sine_pos_enc(shape, device, dtype=torch.float32, num_pos_feats=128, temperature=10000):
    """Compute 2D sine position encoding compatible with TensorRT (no cumsum)."""
    _, _, height, width = shape
    scale = 2 * math.pi

    y = torch.arange(1, height + 1, dtype=dtype, device=device).view(1, height, 1).expand(1, height, width)
    x = torch.arange(1, width + 1, dtype=dtype, device=device).view(1, 1, width).expand(1, height, width)

    y = y / (height + 1e-6) * scale
    x = x / (width + 1e-6) * scale

    dim_t = torch.arange(num_pos_feats, dtype=dtype, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x[:, :, :, None] / dim_t
    pos_y = y[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# TRT-friendly nn.MultiheadAttention replacement
# ---------------------------------------------------------------------------


class _MHAWithPrecomputedScale(nn.Module):
    """Drop-in replacement for nn.MultiheadAttention that uses pre-computed scale.

    Eliminates dynamic Sqrt ops in the ONNX graph. TRT's FP16 attention kernel
    handles this pattern correctly.
    """

    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = mha.embed_dim // mha.num_heads
        self.scale = self.head_dim**-0.5
        self.batch_first = mha.batch_first

        self.in_proj_weight = mha.in_proj_weight
        self.in_proj_bias = mha.in_proj_bias
        self.out_proj = mha.out_proj

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None, need_weights=False):
        """Manual multi-head attention with pre-computed scale (no Sqrt in ONNX)."""
        if self.batch_first:
            # (batch, seq, dim)
            bsz, tgt_len, _ = query.shape
            src_len = key.shape[1]
        else:
            # (seq, batch, dim)
            tgt_len, bsz, _ = query.shape
            src_len = key.shape[0]

        # Project Q, K, V using in_proj_weight (same as nn.MHA)
        w = self.in_proj_weight
        b = self.in_proj_bias
        d = self.embed_dim
        q = F.linear(query, w[:d], b[:d] if b is not None else None)
        k = F.linear(key, w[d : 2 * d], b[d : 2 * d] if b is not None else None)
        v = F.linear(value, w[2 * d :], b[2 * d :] if b is not None else None)

        # Reshape to (batch, heads, seq, head_dim)
        if self.batch_first:
            q = q.reshape(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.reshape(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.reshape(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            q = q.reshape(tgt_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            k = k.reshape(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            v = v.reshape(src_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        # Attention with pre-computed scale
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # (tgt_len, src_len) -> broadcast to (1, 1, tgt_len, src_len)
                attn_weights = attn_weights + attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # (bsz*num_heads, tgt_len, src_len) -> (bsz, num_heads, tgt_len, src_len)
                attn_weights = attn_weights + attn_mask.view(bsz, self.num_heads, tgt_len, src_len)

        if key_padding_mask is not None:
            # key_padding_mask: (bsz, src_len), True = padded position to mask
            if key_padding_mask.dtype == torch.bool:
                kpm = torch.zeros_like(key_padding_mask, dtype=attn_weights.dtype)
                kpm = kpm.masked_fill(key_padding_mask, float("-inf"))
            else:
                kpm = key_padding_mask
            attn_weights = attn_weights + kpm.view(bsz, 1, 1, src_len)

        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Reshape back to input format
        if self.batch_first:
            # (batch, heads, tgt_len, head_dim) -> (batch, tgt_len, embed_dim)
            out = (attn_weights @ v).transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        else:
            # (batch, heads, tgt_len, head_dim) -> (tgt_len, batch, embed_dim)
            out = (attn_weights @ v).permute(2, 0, 1, 3).reshape(tgt_len, bsz, self.embed_dim)
        out = self.out_proj(out)

        return out, attn_weights if need_weights else None


def _replace_mha_modules(model):
    """Replace nn.MultiheadAttention in DETR + geometry encoder + segmentation head with TRT-friendly version.

    Only replaces modules under model.transformer.*, model.geometry_encoder.*, and
    model.segmentation_head.*. Skips language backbone (CLIP) which has different conventions.
    """
    count = 0
    targets = []
    if hasattr(model, "transformer"):
        targets.append(model.transformer)
    if hasattr(model, "geometry_encoder") and model.geometry_encoder is not None:
        targets.append(model.geometry_encoder)
    if hasattr(model, "segmentation_head") and model.segmentation_head is not None:
        targets.append(model.segmentation_head)

    for target in targets:
        for module in target.modules():
            for name, child in module.named_children():
                if isinstance(child, nn.MultiheadAttention):
                    setattr(module, name, _MHAWithPrecomputedScale(child))
                    count += 1
    return count


# ---------------------------------------------------------------------------
# ONNX Wrapper: Vision Encoder
# ---------------------------------------------------------------------------


class _ViTBlockONNX(nn.Module):
    """Single ViT block rewritten for TRT FP16 precision.

    Performs attention inline with separate Q/K/V + SDPA(scale=) + rotate_half RoPE.
    This produces an ONNX graph that TRT's FP16 kernels handle accurately
    (cosine 0.9999 per block vs 0.994 from the default Block.forward path).
    """

    def __init__(self, block):
        super().__init__()
        self.norm1 = block.norm1
        self.norm2 = block.norm2
        self.q_proj = block.attn.q_proj
        self.k_proj = block.attn.k_proj
        self.v_proj = block.attn.v_proj
        self.proj = block.attn.proj
        self.num_heads = block.attn.num_heads
        self.scale = block.attn.scale
        self.ls1 = block.ls1
        self.ls2 = block.ls2
        self.mlp = block.mlp
        self.window_size = block.window_size
        self.use_rope = block.attn.use_rope
        if self.use_rope and hasattr(block.attn, "freqs_cos"):
            self.register_buffer("freqs_cos", block.attn.freqs_cos)
            self.register_buffer("freqs_sin", block.attn.freqs_sin)

    @staticmethod
    def _rotate_half(x):
        x = x.unflatten(-1, (-1, 2))
        a, b = x.unbind(-1)
        return torch.stack((-b, a), dim=-1).flatten(-2)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        B, H, W, C = x.shape

        # Window partition (static shapes, explicit F.pad for TRT)
        if self.window_size > 0:
            ws = self.window_size
            pad_h = (ws - H % ws) % ws
            pad_w = (ws - W % ws) % ws
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            Hp, Wp = H + pad_h, W + pad_w
            x = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)

        # Inline attention: separate Q/K/V + RoPE + SDPA(scale=)
        Bw, Hw, Ww = x.shape[0], x.shape[1], x.shape[2]
        L = Hw * Ww
        q = self.q_proj(x).reshape(Bw, L, self.num_heads, -1).transpose(1, 2)
        k = self.k_proj(x).reshape(Bw, L, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(x).reshape(Bw, L, self.num_heads, -1).transpose(1, 2)

        if self.use_rope:
            cos = self.freqs_cos.unsqueeze(0).unsqueeze(0)
            sin = self.freqs_sin.unsqueeze(0).unsqueeze(0)
            q = q.float() * cos + self._rotate_half(q.float()) * sin
            k = k.float() * cos + self._rotate_half(k.float()) * sin

        x = F.scaled_dot_product_attention(q, k, v.float(), scale=self.scale)
        x = x.to(shortcut.dtype).transpose(1, 2).reshape(Bw, Hw, Ww, -1)
        x = self.proj(x)
        x = self.ls1(x)

        # Window unpartition
        if self.window_size > 0:
            x = x.view(B, Hp // ws, Wp // ws, ws, ws, -1)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
            if Hp > H or Wp > W:
                x = x[:, :H, :W, :].contiguous()

        x = shortcut + x
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class SAM3VisionEncoderONNX(nn.Module):
    """ONNX wrapper for SAM3 vision encoder with TRT FP16 compatibility.

    Uses _ViTBlockONNX for each ViT block to produce a TRT-friendly ONNX graph.
    Pre-computes position embeddings and FPN sine position encoding as buffers.

    Outputs both SAM3 FPN features (for DETR decoder) and SAM2 FPN features
    (for point prompt mask decoder). The SAM3 backbone has a DUAL neck with
    separate learned weights: ``convs`` for SAM3 and ``sam2_convs`` for SAM2.
    """

    def __init__(self, model, imgsz=1008, sam2_convs=None):
        super().__init__()
        neck = model.backbone.vision_backbone
        trunk = neck.trunk

        self.patch_embed = trunk.patch_embed
        self.ln_pre = trunk.ln_pre
        self.ln_post = trunk.ln_post
        self.fpn_convs = neck.convs

        # SAM2 neck (separate learned weights for point-prompt mask decoder)
        self.has_sam2_neck = sam2_convs is not None
        if sam2_convs is not None:
            self.sam2_convs = sam2_convs

        patch_size = trunk.patch_size
        self.h_patches = imgsz // patch_size
        self.w_patches = imgsz // patch_size
        self.hidden_size = trunk.blocks[0].mlp.fc1.in_features
        self.full_attn_ids = trunk.full_attn_ids
        self.pretrain_use_cls_token = trunk.pretrain_use_cls_token

        # Wrap each block with the TRT-friendly inline attention
        self.blocks = nn.ModuleList([_ViTBlockONNX(blk) for blk in trunk.blocks])

        # Pre-compute ViT position embeddings
        if trunk.pos_embed is not None:
            pos_embed = trunk.pos_embed.data.clone()
            pos_embed_spatial = pos_embed[:, 1:] if self.pretrain_use_cls_token else pos_embed
            num_positions = pos_embed_spatial.shape[1]
            pretrain_size = int(num_positions**0.5)
            pos_embed_2d = pos_embed_spatial.reshape(1, pretrain_size, pretrain_size, self.hidden_size).permute(
                0, 3, 1, 2
            )
            rh = self.h_patches // pretrain_size + 1
            rw = self.w_patches // pretrain_size + 1
            pos_embed_2d = pos_embed_2d.tile([1, 1, rh, rw])[:, :, : self.h_patches, : self.w_patches]
            pos_embed_flat = pos_embed_2d.permute(0, 2, 3, 1).reshape(
                1, self.h_patches * self.w_patches, self.hidden_size
            )
            self.register_buffer("vit_pos_embed", pos_embed_flat)
        else:
            self.vit_pos_embed = None

        # Pre-compute FPN sine position encoding for level 2
        fpn_hidden_size = 256
        num_pos_feats = fpn_hidden_size // 2
        fpn_pos = _compute_sine_pos_enc(
            shape=(1, fpn_hidden_size, self.h_patches, self.w_patches),
            device=torch.device("cpu"),
            dtype=torch.float32,
            num_pos_feats=num_pos_feats,
        )
        self.register_buffer("fpn_pos_2", fpn_pos)

    def forward(self, images: torch.Tensor):
        """Forward: patch embed -> pos embed -> ViT blocks -> dual FPN -> outputs.

        Returns SAM3 FPN features (for DETR decoder) and, if available,
        SAM2 FPN features (for point-prompt mask decoder).
        """
        batch_size = images.shape[0]

        x = self.patch_embed(images)
        x_flat = x.flatten(1, 2)
        if self.vit_pos_embed is not None:
            x_flat = x_flat + self.vit_pos_embed
        x = x_flat.view(batch_size, self.h_patches, self.w_patches, -1)

        x = self.ln_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == self.full_attn_ids[-1]:
                x = self.ln_post(x)

        feats = x.permute(0, 3, 1, 2)

        # SAM3 FPN (for DETR bbox/text decoder)
        fpn_feat_0 = self.fpn_convs[0](feats)
        fpn_feat_1 = self.fpn_convs[1](feats)
        fpn_feat_2 = self.fpn_convs[2](feats)

        if self.has_sam2_neck:
            # SAM2 FPN (for point-prompt mask decoder — separate learned weights)
            sam2_feat_0 = self.sam2_convs[0](feats)
            sam2_feat_1 = self.sam2_convs[1](feats)
            sam2_feat_2 = self.sam2_convs[2](feats)
            return (
                fpn_feat_0,
                fpn_feat_1,
                fpn_feat_2,
                self.fpn_pos_2.expand(batch_size, -1, -1, -1),
                sam2_feat_0,
                sam2_feat_1,
                sam2_feat_2,
            )

        return fpn_feat_0, fpn_feat_1, fpn_feat_2, self.fpn_pos_2.expand(batch_size, -1, -1, -1)


# ---------------------------------------------------------------------------
# ONNX Wrapper: Text Encoder
# ---------------------------------------------------------------------------


class SAM3TextEncoderONNX(nn.Module):
    """ONNX wrapper for SAM3 text encoder (CLIP encoder + projection to 256-dim)."""

    def __init__(self, model):
        super().__init__()
        self.language_backbone = model.backbone.language_backbone

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode pre-tokenized text. Returns (text_features, text_mask)."""
        text_attention_mask = (tokens != 0).bool()
        _, text_memory = self.language_backbone.encoder(tokens)
        text_memory = text_memory.transpose(0, 1)
        text_features = self.language_backbone.resizer(text_memory)
        return text_features, text_attention_mask


# ---------------------------------------------------------------------------
# ONNX Wrapper: SAM Prompt Encoder (for point prompts)
# ---------------------------------------------------------------------------


class SAM3PromptEncoderONNX(nn.Module):
    """ONNX wrapper for SAM prompt encoder (points → sparse/dense embeddings).

    Reimplements point embedding inline to avoid advanced boolean indexing
    (point_embedding[labels == X] = ...) which traces poorly to ONNX.
    Uses torch.where with label-based selection instead.

    Inputs:
        point_coords: [B, N, 2] float32 — point coordinates in pixel space
        point_labels: [B, N] int32 — 1=foreground, 0=background

    Outputs:
        sparse_embeddings: [B, N+1, 256] — point embeddings (N points + 1 padding)
        dense_embeddings:  [B, 256, 72, 72] — spatial embedding (no-mask default)
        dense_pe:          [1, 256, 72, 72] — positional encoding for mask decoder
    """

    def __init__(self, tracker_model):
        super().__init__()
        pe = tracker_model.sam_prompt_encoder
        self.pe_layer = pe.pe_layer
        self.input_image_size = pe.input_image_size
        self.embed_dim = pe.embed_dim
        self.image_embedding_size = pe.image_embedding_size

        # Copy label embedding weights
        self.register_buffer("embed_bg", pe.point_embeddings[0].weight)  # label=0 (background)
        self.register_buffer("embed_fg", pe.point_embeddings[1].weight)  # label=1 (foreground)
        self.register_buffer("embed_pad", pe.not_a_point_embed.weight)  # label=-1 (padding)
        self.register_buffer("no_mask_embed", pe.no_mask_embed.weight)
        self.register_buffer("dense_pe", pe.get_dense_pe())

    def forward(self, point_coords: torch.Tensor, point_labels: torch.Tensor):
        B, N, _ = point_coords.shape

        # Add padding point (label=-1) at the end — same as original with pad=True
        pad_coords = torch.zeros(B, 1, 2, dtype=point_coords.dtype, device=point_coords.device)
        pad_labels = torch.full((B, 1), -1, dtype=point_labels.dtype, device=point_labels.device)
        coords = torch.cat([point_coords, pad_coords], dim=1)  # (B, N+1, 2)
        labels = torch.cat([point_labels, pad_labels], dim=1)  # (B, N+1)

        # Positional encoding of coordinates (shift by 0.5 to pixel center)
        point_pe = self.pe_layer.forward_with_coords(coords + 0.5, self.input_image_size)  # (B, N+1, 256)

        # Add label-specific embeddings using torch.where (ONNX-friendly)
        # For each label value, add the corresponding embedding
        labels_3d = labels.unsqueeze(-1)  # (B, N+1, 1) for broadcasting

        # Start with positional encoding, then add label embeddings
        embed = point_pe
        embed = embed + torch.where(labels_3d == 0, self.embed_bg, torch.zeros_like(self.embed_bg))
        embed = embed + torch.where(labels_3d == 1, self.embed_fg, torch.zeros_like(self.embed_fg))
        # Padding point: zero out PE and add padding embedding
        is_pad = (labels_3d == -1).float()
        embed = embed * (1.0 - is_pad) + is_pad * self.embed_pad

        # Dense embeddings (no mask input — use no_mask_embed)
        dense = self.no_mask_embed.reshape(1, -1, 1, 1).expand(
            B, -1, self.image_embedding_size[0], self.image_embedding_size[1]
        )

        return embed, dense, self.dense_pe


# ---------------------------------------------------------------------------
# ONNX Wrapper: SAM Mask Decoder (for point prompts)
# ---------------------------------------------------------------------------


class SAM3MaskDecoderONNX(nn.Module):
    """ONNX wrapper for SAM mask decoder (embeddings + features → masks + scores).

    Takes prompt embeddings from the prompt encoder and image features from the
    vision encoder, produces segmentation masks and quality scores.

    Inputs:
        image_embeddings:         [B, 256, 72, 72] — fpn_feat_2 from vision encoder
        image_pe:                 [1, 256, 72, 72] — positional encoding from prompt encoder
        sparse_prompt_embeddings: [B, N+1, 256] — from prompt encoder
        dense_prompt_embeddings:  [B, 256, 72, 72] — from prompt encoder
        high_res_feat_0:          [B, 256, 288, 288] — fpn_feat_0 from vision encoder
        high_res_feat_1:          [B, 256, 144, 144] — fpn_feat_1 from vision encoder

    Outputs:
        masks:      [B, num_masks, 288, 288] — predicted masks (1 or 3 depending on multimask)
        iou_scores: [B, num_masks] — quality scores for each mask
    """

    def __init__(self, tracker_model, multimask_output=False):
        super().__init__()
        self.mask_decoder = tracker_model.sam_mask_decoder
        # conv_s0 (256→32) and conv_s1 (256→64) project raw FPN features
        # to the channel dims expected by the mask decoder's upscaling path.
        # These are applied OUTSIDE the mask decoder in the original pipeline
        # (in _prepare_backbone_features), so we fold them in here.
        self.conv_s0 = self.mask_decoder.conv_s0
        self.conv_s1 = self.mask_decoder.conv_s1
        self.multimask_output = multimask_output

        # Bake in no_mem_embed: in PyTorch, this is added to the image
        # embeddings (fpn_feat_2) before the mask decoder on initial frames
        # (when directly_add_no_mem_embed=True, which is the SAM3 default).
        # Shape [1, 1, 256] → [1, 256, 1, 1] for spatial broadcast.
        no_mem = tracker_model.no_mem_embed.data.clone()  # [1, 1, 256]
        self.register_buffer("no_mem_embed", no_mem.squeeze(0).unsqueeze(-1).unsqueeze(-1))  # [1, 256, 1, 1]

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        high_res_feat_0: torch.Tensor,
        high_res_feat_1: torch.Tensor,
    ):
        # Add no_mem_embed bias (matches PyTorch: vision_feats[-1] + no_mem_embed)
        image_embeddings = image_embeddings + self.no_mem_embed

        # Project high-res features to expected channel dims
        feat_s0 = self.conv_s0(high_res_feat_0)  # (B, 256, 288, 288) → (B, 32, 288, 288)
        feat_s1 = self.conv_s1(high_res_feat_1)  # (B, 256, 144, 144) → (B, 64, 144, 144)

        masks, iou_scores, _, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=self.multimask_output,
            repeat_image=False,
            high_res_features=[feat_s0, feat_s1],
        )
        return masks, iou_scores


# ---------------------------------------------------------------------------
# ONNX Wrapper: Decoder (DETR, for text/bbox prompts)
# ---------------------------------------------------------------------------


class SAM3DecoderONNX(nn.Module):
    """ONNX wrapper for SAM3 decoder (geometry encoder + DETR encoder-decoder + mask heads).

    Folds the geometry encoder into the decoder so the engine accepts raw box prompts
    as additional inputs. The geometry encoder produces box embeddings which are
    concatenated with the text prompt features before being fed to the DETR encoder.

    Inputs:
        fpn_feat_0/1/2, fpn_pos_2  : From vision encoder
        prompt_features            : From text encoder, [seq, B, 256]
        prompt_mask                : From text encoder, [B, seq] (True=valid)
        input_boxes                : [B, num_boxes, 4] normalized CxCyWH (use zeros + label=-10 for "no boxes")
        input_boxes_labels         : [B, num_boxes] int32 (1=positive, 0=negative, -10=ignore/padding)

    All nn.MultiheadAttention modules are replaced with TRT-friendly manual attention
    using pre-computed scale constants (no dynamic Sqrt).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        fpn_feat_0: torch.Tensor,
        fpn_feat_1: torch.Tensor,
        fpn_feat_2: torch.Tensor,
        fpn_pos_2: torch.Tensor,
        prompt_features: torch.Tensor,
        prompt_mask: torch.Tensor,
        input_boxes: torch.Tensor,
        input_boxes_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run decoder. Returns (pred_logits, pred_boxes, pred_masks, presence_logits)."""
        from ultralytics.models.sam.modules.sam import SAM2Model
        from ultralytics.models.sam.sam3.geometry_encoders import Prompt

        backbone_out = {
            "vision_pos_enc": [fpn_pos_2, fpn_pos_2, fpn_pos_2],
            "backbone_fpn": [fpn_feat_0, fpn_feat_1, fpn_feat_2],
        }

        backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = SAM2Model._prepare_backbone_features(
            self.model, backbone_out, batch=prompt_mask.shape[0]
        )

        # Build a Prompt object from input_boxes / input_boxes_labels.
        # input_boxes: [B, N, 4] -> reshape to (N, B, 4) for sequence-first convention
        # input_boxes_labels: [B, N] int -> (N, B) ; values: 1=pos, 0=neg, -10=ignore (padding)
        boxes_seq_first = input_boxes.transpose(0, 1)  # (N, B, 4)
        # box_mask: True = padded position to ignore
        box_mask = input_boxes_labels == -10  # (B, N)
        # Replace -10 in labels with 0 (will be masked out anyway)
        labels_clean = (
            torch.where(
                input_boxes_labels == -10,
                torch.zeros_like(input_boxes_labels),
                input_boxes_labels,
            )
            .transpose(0, 1)
            .long()
        )

        geo_prompt = Prompt(
            box_embeddings=boxes_seq_first,
            box_mask=box_mask,
            box_labels=labels_clean,
        )

        # Run geometry encoder + concat with text features (matches model._encode_prompt + forward_grounding)
        prompt_text = prompt_features
        text_mask_pad = ~prompt_mask  # True = padded

        geo_feats, geo_masks = self.model.geometry_encoder(
            geo_prompt=geo_prompt,
            img_feats=img_feats,
            img_sizes=vis_feat_sizes,
            img_pos_embeds=img_pos_embeds,
        )

        # Concatenate text + geometry (text first, then geometry)
        prompt = torch.cat([prompt_text, geo_feats], dim=0)
        pmask = torch.cat([text_mask_pad, geo_masks], dim=1)

        encoder_out = self.model._run_encoder(img_feats, img_pos_embeds, vis_feat_sizes, prompt, pmask)
        out = {"backbone_out": backbone_out}

        out, hs = self.model._run_decoder(
            memory=encoder_out["encoder_hidden_states"],
            pos_embed=encoder_out["pos_embed"],
            src_mask=encoder_out["padding_mask"],
            out=out,
            prompt=prompt,
            prompt_mask=pmask,
            encoder_out=encoder_out,
        )

        self.model._run_segmentation_heads(
            out=out,
            backbone_out=backbone_out,
            encoder_hidden_states=encoder_out["encoder_hidden_states"],
            prompt=prompt,
            prompt_mask=pmask,
            hs=hs,
        )

        return out["pred_logits"], out["pred_boxes"], out["pred_masks"], out["presence_logit_dec"]


# ---------------------------------------------------------------------------
# ONNX preparation
# ---------------------------------------------------------------------------


def _prepare_for_onnx_export(model):
    """Prepare SAM3SemanticModel for TRT-friendly ONNX export.

    Applies FP16 compatibility fixes:
    1. ViT attention: separate Q/K/V, pre-computed scale, rotate_half RoPE
    2. DETR attention: replace nn.MultiheadAttention with pre-computed scale version
    3. GELU: replace with tanh-approximate (traces to native ONNX Gelu op)
    4. Disable activation checkpointing
    """
    from ultralytics.models.sam.sam3.vitdet import Attention

    # Fix ViT attention for TRT
    for module in model.modules():
        if isinstance(module, Attention):
            module.prepare_for_onnx_export()

    # Replace all nn.MultiheadAttention with TRT-friendly version (decoder + encoder)
    n_replaced = _replace_mha_modules(model)
    LOGGER.info(f"SAM3 ONNX: replaced {n_replaced} nn.MultiheadAttention with pre-computed scale")

    # Replace nn.GELU() with tanh-approximate -> native ONNX Gelu op
    for module in model.modules():
        for name, child in module.named_children():
            if isinstance(child, nn.GELU):
                setattr(module, name, nn.GELU(approximate="tanh"))

    # Disable activation checkpointing wherever it appears; the flag name varies across submodules
    # (trunk, DETR encoder/decoder, geometry encoder, segmentation head, CLIP language backbone).
    ckpt_flags = (
        "use_act_checkpoint",
        "use_act_ckpt",
        "act_ckpt",
        "grad_checkpointing",
        "act_ckpt_whole_vision_backbone",
        "act_ckpt_whole_language_backbone",
    )
    for module in model.modules():
        for flag in ckpt_flags:
            if hasattr(module, flag):
                setattr(module, flag, False)

    # Export ROIAlign via grid_sample so the decoder builds without the TensorRT ROIAlign plugin.
    if getattr(model, "geometry_encoder", None) is not None:
        model.geometry_encoder._export_roi_grid_sample = True


# ---------------------------------------------------------------------------
# ONNX post-processing
# ---------------------------------------------------------------------------


def _onnx_postprocess(f, metadata, half=False, device_type="cpu", prefix="SAM3 ONNX:"):
    """Post-process: shape inference + metadata + IR version limit.

    onnxslim is deliberately not run: it corrupts SAM3 subgraphs such as RoI align and the
    complex attention patterns, so the modules are exported without simplification.
    """
    import onnx
    from onnx import shape_inference

    model_onnx = onnx.load(f)

    try:
        model_onnx = shape_inference.infer_shapes(model_onnx)
    except Exception as e:
        LOGGER.warning(f"{prefix} shape inference failed for {Path(f).name}: {e}")

    for k, v in metadata.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)

    if getattr(model_onnx, "ir_version", 0) > 10:
        model_onnx.ir_version = 10

    if half and device_type == "cpu":
        try:
            from onnxruntime.transformers import float16

            LOGGER.info(f"{prefix} converting {Path(f).name} to FP16...")
            model_onnx = float16.convert_float_to_float16(model_onnx, keep_io_types=True)
        except Exception as e:
            LOGGER.warning(f"{prefix} FP16 conversion failure for {Path(f).name}: {e}")

    onnx.save(model_onnx, f)


# ---------------------------------------------------------------------------
# Main ONNX export
# ---------------------------------------------------------------------------


def export_sam3_onnx(
    checkpoint_path: str,
    device: torch.device | str = "cpu",
    opset: int = 20,
    half: bool = False,
    output_dir: str | None = None,
    imgsz: int = 1008,
    prefix: str = "SAM3 ONNX:",
) -> list[str]:
    """Export SAM3SemanticModel as 3 ONNX files from a .pt checkpoint.

    Args:
        checkpoint_path: Path to SAM3 checkpoint (.pt).
        device: Device for export (cpu recommended to avoid OOM).
        opset: ONNX opset version (20 recommended for native Gelu op).
        half: FP16 ONNX export (for ONNX-only deployment, not TRT).
        output_dir: Parent directory for output folder.
        imgsz: Image size (must be divisible by 14).
        prefix: Log prefix.

    Returns:
        List of 3 ONNX file paths.
    """
    from ultralytics.models.sam.build_sam3 import build_sam3_image_model
    from ultralytics.utils.checks import check_requirements

    check_requirements(["onnx>=1.12.0,<2.0.0"])
    import onnx

    device = torch.device(device) if isinstance(device, str) else device
    assert imgsz % 14 == 0, f"imgsz={imgsz} must be divisible by patch_size=14"

    LOGGER.info(f"\n{prefix} building SAM3SemanticModel from {checkpoint_path}...")
    model = build_sam3_image_model(checkpoint_path, enable_segmentation=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    if imgsz != 1008:
        LOGGER.info(f"{prefix} setting image size to {imgsz}x{imgsz}...")
        model.set_imgsz((imgsz, imgsz))

    _prepare_for_onnx_export(model)

    dtype = torch.float32
    if half and device.type != "cpu":
        model = model.half()
        dtype = torch.float16

    if output_dir is None:
        output_dir = str(Path(checkpoint_path).parent)
    output_path = Path(output_dir) / f"{Path(checkpoint_path).stem}_onnx"
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = {"author": "Ultralytics", "task": "segment", "stride": 14, "imgsz": [imgsz, imgsz]}
    exported_files = []

    def _export(module, args, name, input_names, output_names, dynamic_axes=None):
        """Trace one module to ONNX with the shared export options and record the path."""
        f = str(output_path / name)
        torch.onnx.export(
            module,
            args,
            f,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        exported_files.append(f)
        return f

    # === 1. Vision Encoder ===
    # Load SAM2 neck weights from the interactive model (separate learned FPN for point prompts)
    from ultralytics.models.sam.build_sam3 import build_interactive_sam3

    LOGGER.info(f"{prefix} loading interactive model for SAM2 neck weights...")
    tracker_model_for_neck = build_interactive_sam3(checkpoint_path)
    tracker_model_for_neck = tracker_model_for_neck.to(device).eval()
    sam2_convs = tracker_model_for_neck.image_encoder.vision_backbone.sam2_convs
    if sam2_convs is None:
        LOGGER.warning(f"{prefix} interactive model has no sam2_convs — point prompts may not work correctly")

    LOGGER.info(f"{prefix} exporting vision encoder with dual neck (opset {opset})...")
    vis_encoder = SAM3VisionEncoderONNX(model, imgsz=imgsz, sam2_convs=sam2_convs).to(device).eval()
    dummy_image = torch.randn(1, 3, imgsz, imgsz, dtype=dtype, device=device)

    output_names_vis = ["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"]
    if sam2_convs is not None:
        output_names_vis += ["sam2_feat_0", "sam2_feat_1", "sam2_feat_2"]

    _export(vis_encoder, (dummy_image,), "sam3_vision_encoder.onnx", ["images"], output_names_vis)

    with torch.no_grad():
        vis_out = vis_encoder(dummy_image)
    # Only the four FPN tensors are reused as dummy inputs for the decoder and mask decoder exports;
    # the optional sam2_feat_* outputs (when sam2_convs is present) are not needed here.
    fpn0, fpn1, fpn2, fpos2 = vis_out[:4]

    del tracker_model_for_neck

    # === 2. Text Encoder ===
    LOGGER.info(f"{prefix} exporting text encoder (opset {opset})...")
    txt_encoder = SAM3TextEncoderONNX(model).to(device).eval()
    dummy_tokens = torch.zeros(1, 32, dtype=torch.long, device=device)
    dummy_tokens[0, :3] = torch.tensor([49406, 2533, 49407])

    _export(txt_encoder, (dummy_tokens,), "sam3_text_encoder.onnx", ["tokens"], ["text_features", "text_mask"])

    with torch.no_grad():
        txt_feats, txt_mask = txt_encoder(dummy_tokens)

    # === 3. Decoder (with folded geometry encoder) ===
    LOGGER.info(f"{prefix} exporting decoder (opset {opset})...")
    decoder = SAM3DecoderONNX(model).to(device).eval()

    # Dummy box inputs: 1 dummy box with label=-10 (ignored), so the engine works for text-only too.
    # Real bbox inference passes actual boxes with labels=1 (positive) or 0 (negative).
    dummy_boxes = torch.zeros(1, 1, 4, dtype=dtype, device=device)
    dummy_box_labels = torch.full((1, 1), -10, dtype=torch.int32, device=device)

    _export(
        decoder,
        (fpn0, fpn1, fpn2, fpos2, txt_feats, txt_mask, dummy_boxes, dummy_box_labels),
        "sam3_decoder.onnx",
        [
            "fpn_feat_0",
            "fpn_feat_1",
            "fpn_feat_2",
            "fpn_pos_2",
            "prompt_features",
            "prompt_mask",
            "input_boxes",
            "input_boxes_labels",
        ],
        ["pred_logits", "pred_boxes", "pred_masks", "presence_logit_dec"],
        {"input_boxes": {1: "num_boxes"}, "input_boxes_labels": {1: "num_boxes"}},
    )

    # === 4. SAM Prompt Encoder (for point prompts) ===
    LOGGER.info(f"{prefix} exporting SAM prompt encoder (opset {opset})...")
    tracker_model = build_interactive_sam3(checkpoint_path)
    tracker_model = tracker_model.to(device).eval()

    prompt_enc = SAM3PromptEncoderONNX(tracker_model).to(device).eval()
    dummy_pts = torch.tensor([[[500.0, 500.0]]], dtype=dtype, device=device)
    dummy_lbl = torch.tensor([[1]], dtype=torch.int32, device=device)

    _export(
        prompt_enc,
        (dummy_pts, dummy_lbl),
        "sam3_prompt_encoder.onnx",
        ["point_coords", "point_labels"],
        ["sparse_embeddings", "dense_embeddings", "dense_pe"],
        {"point_coords": {1: "num_points"}, "point_labels": {1: "num_points"}, "sparse_embeddings": {1: "num_embeds"}},
    )

    # === 5. SAM Mask Decoder (for point prompts) ===
    # Use multimask_output=True to produce 3 candidate masks + IoU scores.
    # The best mask is selected at runtime by argmax(iou_scores).
    # PyTorch SAM3 uses multimask=True for 1-point prompts, and multimask=True
    # with best-mask selection also improves multi-point quality.
    LOGGER.info(f"{prefix} exporting SAM mask decoder (opset {opset}, multimask=True)...")
    mask_dec = SAM3MaskDecoderONNX(tracker_model, multimask_output=True).to(device).eval()

    with torch.no_grad():
        sparse_dummy, dense_dummy, dpe_dummy = prompt_enc(dummy_pts, dummy_lbl)

    _export(
        mask_dec,
        (fpn2, dpe_dummy, sparse_dummy, dense_dummy, fpn0, fpn1),
        "sam3_mask_decoder.onnx",
        [
            "image_embeddings",
            "image_pe",
            "sparse_prompt_embeddings",
            "dense_prompt_embeddings",
            "high_res_feat_0",
            "high_res_feat_1",
        ],
        ["masks", "iou_scores"],
        {"sparse_prompt_embeddings": {1: "num_embeds"}},
    )

    del tracker_model

    # === Post-processing ===
    for f in exported_files:
        component_metadata = {**metadata, "component": Path(f).stem}
        _onnx_postprocess(
            f,
            metadata=component_metadata,
            half=half,
            device_type=device.type,
            prefix=prefix,
        )
        model_onnx = onnx.load(f)
        onnx.checker.check_model(model_onnx)
        LOGGER.info(f"{prefix} validated {Path(f).name} ({Path(f).stat().st_size / 1e6:.1f} MB)")

    LOGGER.info(f"{prefix} export complete -> {output_path}")
    return exported_files


# ---------------------------------------------------------------------------
# TensorRT engine export
# ---------------------------------------------------------------------------


def export_sam3_engine(
    onnx_dir: str,
    half: bool = True,
    workspace: int | None = None,
    verbose: bool = False,
    prefix: str = "SAM3 TensorRT:",
) -> list[str]:
    """Convert SAM3 ONNX models to TensorRT engines (Python API, no trtexec).

    Args:
        onnx_dir: Path to the ONNX directory.
        half: Enable FP16 precision for TRT.
        workspace: TensorRT workspace size in GB.
        verbose: Enable verbose TRT logging.
        prefix: Log prefix.

    Returns:
        List of 3 engine file paths.
    """
    from ultralytics.utils.checks import check_requirements
    from ultralytics.utils.export.engine import onnx2engine

    check_requirements(["onnx>=1.12.0,<2.0.0"])
    import onnx

    onnx_dir = Path(onnx_dir)
    assert onnx_dir.is_dir(), f"ONNX directory not found: {onnx_dir}"

    engine_dir_name = (
        onnx_dir.name.replace("_onnx", "_engine") if "_onnx" in onnx_dir.name else onnx_dir.name + "_engine"
    )
    engine_dir = onnx_dir.parent / engine_dir_name
    engine_dir.mkdir(parents=True, exist_ok=True)

    onnx_files = [
        onnx_dir / "sam3_vision_encoder.onnx",
        onnx_dir / "sam3_text_encoder.onnx",
        onnx_dir / "sam3_decoder.onnx",
    ]
    # Optional point prompt modules
    if (onnx_dir / "sam3_prompt_encoder.onnx").exists():
        onnx_files.append(onnx_dir / "sam3_prompt_encoder.onnx")
    if (onnx_dir / "sam3_mask_decoder.onnx").exists():
        onnx_files.append(onnx_dir / "sam3_mask_decoder.onnx")
    for f in onnx_files[:3]:  # First 3 are required
        assert f.exists(), f"ONNX file not found: {f}"

    exported_engines = []
    for onnx_file in onnx_files:
        LOGGER.info(f"\n{prefix} converting {onnx_file.name}...")

        model_onnx = onnx.load(str(onnx_file))
        dims = model_onnx.graph.input[0].type.tensor_type.shape.dim
        input_shape = tuple(d.dim_value if d.dim_value > 0 else 1 for d in dims)
        while len(input_shape) < 4:
            input_shape = input_shape + (1,)
        input_shape = input_shape[:4]

        engine_file = str(engine_dir / onnx_file.name.replace(".onnx", ".engine"))

        # Modules with a dynamic axis need a custom build with an optimization profile. They honor
        # FP16 through mixed precision (ModelOpt AutoCast keeps overflow prone nodes in FP32), which
        # keeps the detection decoder accurate and builds identically on TensorRT 10 and 11. The
        # static vision and text encoders go through onnx2engine.
        dynamic_modules = {"sam3_decoder", "sam3_prompt_encoder", "sam3_mask_decoder"}
        if onnx_file.stem in dynamic_modules:
            _build_decoder_engine_dynamic(
                onnx_file=str(onnx_file),
                engine_file=engine_file,
                half=half,
                workspace=workspace,
                metadata={"component": onnx_file.stem, "author": "Ultralytics", "task": "segment"},
                verbose=verbose,
                prefix=prefix,
            )
        else:
            onnx2engine(
                onnx_file=str(onnx_file),
                output_file=engine_file,
                workspace=workspace,
                quantize=16 if half else None,
                dynamic=False,
                shape=input_shape,
                metadata={"component": onnx_file.stem, "author": "Ultralytics", "task": "segment"},
                verbose=verbose,
                prefix=prefix,
            )
        exported_engines.append(engine_file)
        LOGGER.info(f"{prefix} saved {Path(engine_file).name} ({'mixed FP16' if half else 'FP32'})")

    LOGGER.info(f"{prefix} export complete -> {engine_dir}")
    return exported_engines


def _autocast_fp16_onnx(onnx_file: str, opt_dynamic: int, prefix: str) -> str:
    """Convert an ONNX module to mixed FP16/FP32 with ModelOpt AutoCast.

    AutoCast assigns FP16 per node but keeps nodes whose observed range overflows FP16 in FP32,
    which is what lets the detection decoder run mostly in FP16 without the text path overflowing.
    Calibration data is synthesized for every input from its rank and dtype (dynamic dims use
    ``opt_dynamic``). Returns the path to the converted ONNX.
    """
    import numpy as np
    import onnx

    from ultralytics.utils.checks import check_requirements

    check_requirements("nvidia-modelopt[onnx]>=0.44")
    import modelopt.onnx.autocast as autocast

    calib = {}
    for inp in onnx.load(onnx_file, load_external_data=False).graph.input:
        tt = inp.type.tensor_type
        dims = [d.dim_value if d.dim_value > 0 else opt_dynamic for d in tt.shape.dim]
        dtype = onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)
        if dtype == np.bool_:
            calib[inp.name] = np.zeros(dims, dtype=np.bool_)
        elif np.issubdtype(dtype, np.integer):
            calib[inp.name] = np.ones(dims, dtype=dtype)
        else:
            calib[inp.name] = np.random.randn(*dims).astype(dtype)
    LOGGER.info(f"{prefix} converting {Path(onnx_file).name} to mixed FP16/FP32 with ModelOpt AutoCast...")
    out_file = str(Path(onnx_file).with_suffix(".mixed.onnx"))
    onnx.save(
        autocast.convert_to_mixed_precision(
            onnx_file, low_precision_type="fp16", keep_io_types=True, calibration_data=calib
        ),
        out_file,
    )
    return out_file


def _build_decoder_engine_dynamic(
    onnx_file: str,
    engine_file: str,
    half: bool = True,
    workspace: int | None = None,
    metadata: dict | None = None,
    verbose: bool = False,
    prefix: str = "SAM3 TensorRT:",
    min_dynamic: int = 1,
    opt_dynamic: int = 5,
    max_dynamic: int = 32,
) -> None:
    """Build a TensorRT engine for an ONNX module with dynamic dimensions.

    Detects symbolic dims and adds an optimization profile [min_dynamic, opt_dynamic, max_dynamic].
    With ``half`` the module is converted to mixed FP16/FP32 by ModelOpt AutoCast and built as a
    strongly-typed network, so the per-node precision is honored identically on TensorRT 10 and 11
    (the FP16 builder flag was removed in TensorRT 11). Without ``half`` the engine is FP32.
    """
    import tensorrt as trt

    from ultralytics.utils.export.engine import write_engine

    if half:
        onnx_file = _autocast_fp16_onnx(onnx_file, opt_dynamic, prefix)

    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    if workspace:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace * (1 << 30)))

    # A strongly-typed network honors the per-node precision baked in by AutoCast on both TensorRT
    # 10 and 11; an FP32 build uses the default explicit batch (the EXPLICIT_BATCH creation flag was
    # removed in TensorRT 10 and the enum member no longer exists in TensorRT 11).
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)) if half else 0
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_file):
        for i in range(parser.num_errors):
            LOGGER.error(f"{prefix} parser error: {parser.get_error(i)}")
        raise RuntimeError(f"Failed to parse ONNX: {onnx_file}")

    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = list(inp.shape)
        if any(d == -1 for d in shape):
            profile.set_shape(
                inp.name,
                min=tuple(min_dynamic if d == -1 else d for d in shape),
                opt=tuple(opt_dynamic if d == -1 else d for d in shape),
                max=tuple(max_dynamic if d == -1 else d for d in shape),
            )
        else:
            profile.set_shape(inp.name, min=tuple(shape), opt=tuple(shape), max=tuple(shape))
    config.add_optimization_profile(profile)

    LOGGER.info(f"{prefix} building {'mixed FP16' if half else 'FP32'} engine as {Path(engine_file).name}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    write_engine(engine_file, serialized, metadata)
