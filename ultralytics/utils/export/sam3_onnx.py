# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import ast
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.utils import LOGGER


def _compute_sine_pos_enc(shape, device, dtype=torch.float32, num_pos_feats=128, temperature=10000):
    """Compute 2D sine position encoding compatible with TensorRT (no cumsum).

    ``height`` and ``width`` may be traced tensors, so the grid is built by broadcasting two ranges instead of reshaping
    to a literal size, which would bake the size into the graph.
    """
    _, _, height, width = shape
    scale = 2 * math.pi

    rows = torch.arange(1, height + 1, dtype=dtype, device=device)
    cols = torch.arange(1, width + 1, dtype=dtype, device=device)
    y = (rows[:, None] * torch.ones_like(cols)[None, :])[None]
    x = (torch.ones_like(rows)[:, None] * cols[None, :])[None]

    y = y / (rows[-1] + 1e-6) * scale
    x = x / (cols[-1] + 1e-6) * scale

    dim_t = torch.arange(num_pos_feats, dtype=dtype, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x[:, :, :, None] / dim_t
    pos_y = y[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


def _axial_rope(height, width, head_dim, theta=10000.0, pt_size=None, device=None):
    """Build the rotate_half RoPE cos and sin tables for a height by width patch grid.

    Mirrors ``compute_axial_cis`` then ``repeat_interleave(2)``, but as plain cos and sin so no complex tensor reaches
    ONNX, and takes the grid as tensors so a dynamic image size can be traced.

    Args:
        height (int | torch.Tensor): Patch grid height.
        width (int | torch.Tensor): Patch grid width.
        head_dim (int): Attention head dimension.
        theta (float): RoPE frequency base.
        pt_size (int | None): Grid the frequencies were trained on, which rescales the positions, or None to leave them.
        device (torch.device | None): Device to build the tables on.

    Returns:
        (tuple[torch.Tensor, torch.Tensor]): cos and sin, both (height * width, head_dim).
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 4, device=device)[: head_dim // 4].float() / head_dim))
    t = torch.arange(height * width, dtype=torch.float32, device=device)
    scale = 1.0 if pt_size is None else pt_size / width
    t_x = (t % width).float() * scale
    t_y = torch.div(t, width, rounding_mode="floor").float() * scale
    angles = torch.cat([torch.outer(t_x, freqs), torch.outer(t_y, freqs)], dim=-1)

    # Doubling each entry with stack and flatten, the way the rest of this file does, rather than
    # repeat_interleave, whose lowering transposes a tensor the exporter cannot infer a rank for.
    def _pair(v):
        """Repeat every value along the last axis twice, matching repeat_interleave(2, dim=-1)."""
        return torch.stack((v, v), dim=-1).flatten(-2)

    return _pair(angles.cos()), _pair(angles.sin())


class _MHAWithPrecomputedScale(nn.Module):
    """Drop in nn.MultiheadAttention replacement with a constant scale, leaving no dynamic Sqrt in ONNX."""

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

    Only replaces modules under model.transformer.*, model.geometry_encoder.*, and model.segmentation_head.*. Skips
    language backbone (CLIP) which has different conventions.
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


class _ViTBlockONNX(nn.Module):
    """ViT block with inline attention, separate Q/K/V and rotate_half RoPE.

    TRT FP16 reaches cosine 0.9999 per block on this graph against 0.994 for the default forward.
    """

    def __init__(self, block, dynamic=False):
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
        # A windowed block always attends over window_size squared positions, so its table is the
        # same at every image size. Only a global block sees the image grid and needs one built to fit.
        self.dynamic_rope = dynamic and self.window_size == 0 and self.use_rope
        if self.dynamic_rope:
            self.head_dim = block.attn.head_dim
            self.rope_theta = block.attn.rope_theta
            self.rope_pt = block.attn.rope_pt_size[0] if block.attn.rope_interp else None
        elif self.use_rope and hasattr(block.attn, "freqs_cos"):
            self.register_buffer("freqs_cos", block.attn.freqs_cos)
            self.register_buffer("freqs_sin", block.attn.freqs_sin)

    @staticmethod
    def _rotate_half(x):
        x = x.unflatten(-1, (-1, 2))
        a, b = x.unbind(-1)
        return torch.stack((-b, a), dim=-1).flatten(-2)

    def forward(self, x, rope=None):
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
            # A dynamic table is built once by the encoder and handed down. Deriving the grid here
            # instead makes the tracer follow the shape through every preceding block and crash.
            freqs_cos, freqs_sin = rope if self.dynamic_rope else (self.freqs_cos, self.freqs_sin)
            cos = freqs_cos.unsqueeze(0).unsqueeze(0)
            sin = freqs_sin.unsqueeze(0).unsqueeze(0)
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
    """ONNX wrapper for the vision encoder, emitting both necks.

    Wraps every ViT block for TRT FP16 accuracy and precomputes the position encodings as buffers. The backbone has two
    necks with separate weights, ``convs`` for the decoder and ``sam2_convs`` for point prompts, so both sets of
    features are returned.
    """

    def __init__(self, model, imgsz=1008, sam2_convs=None, dynamic=False, max_imgsz=None):
        """Wrap the vision backbone, tracing at ``imgsz`` and optionally emitting the SAM2 neck.

        With ``dynamic`` the graph accepts any square size up to ``max_imgsz``. The absolute position
        embedding is tiled, and tiling repeats with the pretrain period, so one table built for the
        largest grid can simply be cropped for a smaller one. The FPN sine encoding normalizes by the
        grid it covers, so that one has to be rebuilt for every input and cannot be cropped.
        """
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
        self.dynamic = dynamic
        self.h_patches = imgsz // patch_size
        self.w_patches = imgsz // patch_size
        # The tiled table has to cover the largest grid the graph will ever be fed.
        table_patches = (max_imgsz or imgsz) // patch_size if dynamic else self.h_patches
        self.hidden_size = trunk.blocks[0].mlp.fc1.in_features
        self.full_attn_ids = trunk.full_attn_ids
        self.pretrain_use_cls_token = trunk.pretrain_use_cls_token

        # Wrap each block with the TRT-friendly inline attention
        self.blocks = nn.ModuleList([_ViTBlockONNX(blk, dynamic=dynamic) for blk in trunk.blocks])
        global_attn = next((b for b in self.blocks if b.dynamic_rope), None)
        self.rope_head_dim = global_attn.head_dim if global_attn else 0
        self.rope_theta = global_attn.rope_theta if global_attn else 0.0
        self.rope_pt = global_attn.rope_pt if global_attn else None

        # Pre-compute ViT position embeddings
        if trunk.pos_embed is not None:
            pos_embed = trunk.pos_embed.data.clone()
            pos_embed_spatial = pos_embed[:, 1:] if self.pretrain_use_cls_token else pos_embed
            num_positions = pos_embed_spatial.shape[1]
            pretrain_size = int(num_positions**0.5)
            pos_embed_2d = pos_embed_spatial.reshape(1, pretrain_size, pretrain_size, self.hidden_size).permute(
                0, 3, 1, 2
            )
            rh = table_patches // pretrain_size + 1
            rw = table_patches // pretrain_size + 1
            pos_embed_2d = pos_embed_2d.tile([1, 1, rh, rw])[:, :, :table_patches, :table_patches]
            if dynamic:  # keep it square so the forward pass can crop it to the grid it is given
                self.register_buffer("vit_pos_embed", pos_embed_2d)
            else:
                self.register_buffer(
                    "vit_pos_embed",
                    pos_embed_2d.permute(0, 2, 3, 1).reshape(1, self.h_patches * self.w_patches, self.hidden_size),
                )
        else:
            self.vit_pos_embed = None

        # Pre-compute FPN sine position encoding for level 2, unless the grid is only known per call
        self.fpn_hidden_size = 256
        if dynamic:
            self.fpn_pos_2 = None
        else:
            self.register_buffer(
                "fpn_pos_2",
                _compute_sine_pos_enc(
                    shape=(1, self.fpn_hidden_size, self.h_patches, self.w_patches),
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    num_pos_feats=self.fpn_hidden_size // 2,
                ),
            )

    def forward(self, images: torch.Tensor):
        """Forward: patch embed -> pos embed -> ViT blocks -> dual FPN -> outputs.

        Returns SAM3 FPN features (for DETR decoder) and, if available,
        SAM2 FPN features (for point-prompt mask decoder).
        """
        batch_size = images.shape[0]

        x = self.patch_embed(images)  # (B, H, W, C), so the grid is already the shape to follow
        if self.vit_pos_embed is not None:
            if self.dynamic:
                h, w = x.shape[1], x.shape[2]
                x = x + self.vit_pos_embed[:, :, :h, :w].permute(0, 2, 3, 1)
            else:
                x = x.flatten(1, 2).add(self.vit_pos_embed).view(batch_size, self.h_patches, self.w_patches, -1)

        # Every global block shares one grid and one RoPE config, so build the table once here where
        # the shape is still one op from the input, and reuse it.
        rope = None
        if self.dynamic and self.rope_head_dim:
            rope = _axial_rope(x.shape[1], x.shape[2], self.rope_head_dim, self.rope_theta, self.rope_pt, x.device)

        x = self.ln_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x, rope)
            if i == self.full_attn_ids[-1]:
                x = self.ln_post(x)

        feats = x.permute(0, 3, 1, 2)

        # SAM3 FPN (for DETR bbox/text decoder)
        fpn_feat_0 = self.fpn_convs[0](feats)
        fpn_feat_1 = self.fpn_convs[1](feats)
        fpn_feat_2 = self.fpn_convs[2](feats)

        if self.dynamic:  # the encoding normalizes by its own grid, so it cannot be cropped
            fpn_pos = _compute_sine_pos_enc(
                shape=(1, self.fpn_hidden_size, fpn_feat_2.shape[2], fpn_feat_2.shape[3]),
                device=fpn_feat_2.device,
                dtype=torch.float32,
                num_pos_feats=self.fpn_hidden_size // 2,
            ).to(fpn_feat_2.dtype)
        else:
            fpn_pos = self.fpn_pos_2.to(fpn_feat_2.dtype)  # the buffer is fp32, match the features
        fpn_pos = fpn_pos.expand(batch_size, -1, -1, -1)

        if self.has_sam2_neck:
            # SAM2 FPN (for point-prompt mask decoder — separate learned weights)
            sam2_feat_0 = self.sam2_convs[0](feats)
            sam2_feat_1 = self.sam2_convs[1](feats)
            sam2_feat_2 = self.sam2_convs[2](feats)
            return fpn_feat_0, fpn_feat_1, fpn_feat_2, fpn_pos, sam2_feat_0, sam2_feat_1, sam2_feat_2

        return fpn_feat_0, fpn_feat_1, fpn_feat_2, fpn_pos


class SAM3TextEncoderONNX(nn.Module):
    """ONNX wrapper for SAM3 text encoder (CLIP encoder + projection to 256-dim)."""

    def __init__(self, model):
        """Wrap the language backbone of ``model``."""
        super().__init__()
        self.language_backbone = model.backbone.language_backbone

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode pre-tokenized text. Returns (text_features, text_mask)."""
        text_attention_mask = (tokens != 0).bool()
        _, text_memory = self.language_backbone.encoder(tokens)
        text_memory = text_memory.transpose(0, 1)
        text_features = self.language_backbone.resizer(text_memory)
        return text_features, text_attention_mask


def _dense_pos_enc(height, width, gaussian):
    """Build the random frequency positional encoding for a height by width grid.

    Mirrors ``PositionEmbeddingRandom.forward`` but broadcasts two ranges rather than summing a literally sized grid of
    ones, so the grid can come from a traced shape.

    Args:
        height (int | torch.Tensor): Grid height.
        width (int | torch.Tensor): Grid width.
        gaussian (torch.Tensor): The (2, num_pos_feats) random frequency matrix.

    Returns:
        (torch.Tensor): Encoding of shape (1, 2 * num_pos_feats, height, width).
    """
    rows = torch.arange(1, height + 1, dtype=gaussian.dtype, device=gaussian.device)
    cols = torch.arange(1, width + 1, dtype=gaussian.dtype, device=gaussian.device)
    # cumsum of ones minus a half is 0.5, 1.5, ... normalized by the grid it was built on
    y = ((rows - 0.5) / rows[-1])[:, None] * torch.ones_like(cols)[None, :]
    x = torch.ones_like(rows)[:, None] * ((cols - 0.5) / cols[-1])[None, :]
    coords = 2 * math.pi * ((2 * torch.stack([x, y], dim=-1) - 1) @ gaussian)
    return torch.cat([coords.sin(), coords.cos()], dim=-1).permute(2, 0, 1)[None]


class SAM3PromptEncoderONNX(nn.Module):
    """ONNX wrapper turning point prompts into sparse and dense embeddings.

    Embeds points inline rather than with boolean indexing, which traces poorly to ONNX. With ``dynamic`` the graph
    takes the image embedding as a third input purely to read the feature grid off it, since everything this module
    would otherwise bake in follows from that grid.
    """

    def __init__(self, tracker_model, dynamic: bool = False):
        """Wrap the interactive prompt encoder of ``tracker_model``."""
        super().__init__()
        pe = tracker_model.sam_prompt_encoder
        self.pe_layer = pe.pe_layer
        self.input_image_size = pe.input_image_size
        self.image_embedding_size = pe.image_embedding_size
        self.dynamic = dynamic
        # The image is a whole number of patches, so the grid recovers the pixel size it came from
        self.patch_size = pe.input_image_size[0] // pe.image_embedding_size[0]

        # Copy label embedding weights
        self.register_buffer("embed_bg", pe.point_embeddings[0].weight)  # label=0 (background)
        self.register_buffer("embed_fg", pe.point_embeddings[1].weight)  # label=1 (foreground)
        # Labels 2 and 3 are a box's two corners. Without them a box cannot be expressed as a box and
        # the only way to use one is as a visual exemplar, which segments lookalikes instead.
        self.register_buffer("embed_box_tl", pe.point_embeddings[2].weight)  # label=2 (box top left)
        self.register_buffer("embed_box_br", pe.point_embeddings[3].weight)  # label=3 (box bottom right)
        self.register_buffer("embed_pad", pe.not_a_point_embed.weight)  # label=-1 (padding)
        self.register_buffer("no_mask_embed", pe.no_mask_embed.weight)
        if not dynamic:  # a dynamic grid rebuilds this per call, so baking a table would only bloat the graph
            self.register_buffer("dense_pe", pe.get_dense_pe())

    def forward(self, point_coords: torch.Tensor, point_labels: torch.Tensor, image_embeddings: torch.Tensor = None):
        """Embed point prompts into sparse and dense embeddings plus the dense positional encoding."""
        B, _, _ = point_coords.shape
        if self.dynamic:
            grid_h, grid_w = image_embeddings.shape[2], image_embeddings.shape[3]
            image_size = (grid_h * self.patch_size, grid_w * self.patch_size)
        else:
            grid_h, grid_w = self.image_embedding_size
            image_size = self.input_image_size

        # Add padding point (label=-1) at the end — same as original with pad=True
        pad_coords = torch.zeros(B, 1, 2, dtype=point_coords.dtype, device=point_coords.device)
        pad_labels = torch.full((B, 1), -1, dtype=point_labels.dtype, device=point_labels.device)
        coords = torch.cat([point_coords, pad_coords], dim=1)  # (B, N+1, 2)
        labels = torch.cat([point_labels, pad_labels], dim=1)  # (B, N+1)

        # Positional encoding of coordinates (shift by 0.5 to pixel center)
        point_pe = self.pe_layer.forward_with_coords(coords + 0.5, image_size)  # (B, N+1, 256)

        # Add label-specific embeddings using torch.where (ONNX-friendly)
        # For each label value, add the corresponding embedding
        labels_3d = labels.unsqueeze(-1)  # (B, N+1, 1) for broadcasting

        # Start with positional encoding, then add label embeddings
        embed = point_pe
        embed = embed + torch.where(labels_3d == 0, self.embed_bg, torch.zeros_like(self.embed_bg))
        embed = embed + torch.where(labels_3d == 1, self.embed_fg, torch.zeros_like(self.embed_fg))
        embed = embed + torch.where(labels_3d == 2, self.embed_box_tl, torch.zeros_like(self.embed_box_tl))
        embed = embed + torch.where(labels_3d == 3, self.embed_box_br, torch.zeros_like(self.embed_box_br))
        # Padding point: zero out PE and add padding embedding
        is_pad = (labels_3d == -1).to(embed.dtype)
        embed = embed * (1.0 - is_pad) + is_pad * self.embed_pad

        # Dense embeddings (no mask input — use no_mask_embed)
        dense = self.no_mask_embed.reshape(1, -1, 1, 1).expand(B, -1, grid_h, grid_w)

        # The encoding normalizes by its own grid, so a dynamic one has to be rebuilt per call
        dense_pe = (
            _dense_pos_enc(grid_h, grid_w, self.pe_layer.positional_encoding_gaussian_matrix)
            if self.dynamic
            else self.dense_pe
        )
        return embed, dense, dense_pe


class SAM3MaskDecoderONNX(nn.Module):
    """ONNX wrapper turning prompt embeddings and vision features into masks and quality scores."""

    def __init__(self, tracker_model, multimask_output=False):
        """Wrap the interactive mask decoder of ``tracker_model``."""
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
        """Decode point prompts into masks and IoU scores from the SAM2 neck features."""
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


class SAM3DecoderONNX(nn.Module):
    """ONNX wrapper for the DETR decoder and mask heads.

    Folds the geometry encoder in so the graph accepts raw box prompts, whose embeddings are concatenated with the text
    features. Box labels are 1 positive, 0 negative, -10 padding.
    """

    def __init__(self, model, with_geometry: bool = True):
        """Wrap the DETR decoder, tracing the geometry encoder only when ``with_geometry``."""
        super().__init__()
        self.model = model
        self.with_geometry = with_geometry

    def forward(
        self,
        fpn_feat_0: torch.Tensor,
        fpn_feat_1: torch.Tensor,
        fpn_feat_2: torch.Tensor,
        fpn_pos_2: torch.Tensor,
        prompt_features: torch.Tensor,
        prompt_mask: torch.Tensor,
        input_boxes: torch.Tensor = None,
        input_boxes_labels: torch.Tensor = None,
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

        prompt = prompt_features
        pmask = ~prompt_mask  # True = padded

        # A box prompt appends geometry tokens to the text prompt. An ignored box is not neutral, it
        # shifts the presence logit, so the text only graph omits geometry exactly as forward_grounding does.
        if self.with_geometry:
            # Boxes and labels arrive batch first and the encoder wants sequence first. Label -10 marks
            # padding, which the mask already excludes, so those entries are zeroed to keep the embedding valid.
            pad = input_boxes_labels == -10
            geo_feats, geo_masks = self.model.geometry_encoder(
                geo_prompt=Prompt(
                    box_embeddings=input_boxes.transpose(0, 1),
                    box_mask=pad,
                    box_labels=torch.where(pad, torch.zeros_like(input_boxes_labels), input_boxes_labels)
                    .transpose(0, 1)
                    .long(),
                ),
                img_feats=img_feats,
                img_sizes=vis_feat_sizes,
                img_pos_embeds=img_pos_embeds,
            )
            prompt = torch.cat([prompt, geo_feats], dim=0)
            pmask = torch.cat([pmask, geo_masks], dim=1)

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


def _prepare_for_onnx_export(model, dynamic=False):
    """Prepare SAM3SemanticModel for TRT-friendly ONNX export.

    Applies FP16 compatibility fixes:
    1. ViT attention: separate Q/K/V, pre-computed scale, rotate_half RoPE
    2. DETR attention: replace nn.MultiheadAttention with pre-computed scale version
    3. GELU: replace with tanh-approximate (traces to native ONNX Gelu op)
    4. Disable activation checkpointing

    Args:
        model (torch.nn.Module): Model to prepare in place.
        dynamic (bool): Whether the graph will be traced with a dynamic image size, which makes any grid cached at trace
            time a constant that pins the export to that one size.
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

    # Tell the box relative position bias to rebuild its coordinate grid per call instead of reusing
    # the one cached on the first pass, which would otherwise be traced in as a constant.
    if dynamic:
        from ultralytics.models.sam.sam3.decoder import TransformerDecoder
        from ultralytics.models.sam.sam3.encoder import TransformerEncoder

        for module in model.modules():
            if isinstance(module, (TransformerDecoder, TransformerEncoder)):
                module._export_dynamic_shapes = True


def _onnx_postprocess(f, metadata, half=False, device_type="cpu", prefix="SAM3 ONNX:"):
    """Post-process: shape inference + metadata + IR version limit.

    onnxslim is deliberately not run: it corrupts SAM3 subgraphs such as RoI align and the complex attention patterns,
    so the modules are exported without simplification.
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
            from onnxruntime.transformers.onnx_model import OnnxModel

            LOGGER.info(f"{prefix} converting {Path(f).name} to FP16...")
            model_onnx = float16.convert_float_to_float16(model_onnx, keep_io_types=True)
            # keep_io_types wraps every input in a cast appended after the node that consumes it,
            # which is not a sorted graph and fails the checker, so put the nodes back in order.
            sorted_model = OnnxModel(model_onnx)
            sorted_model.topological_sort()
            model_onnx = sorted_model.model
        except Exception as e:
            LOGGER.warning(f"{prefix} FP16 conversion failure for {Path(f).name}: {e}")

    onnx.save(model_onnx, f)


def export_sam3_onnx(
    checkpoint_path: str,
    device: torch.device | str = "cpu",
    opset: int = 20,
    half: bool = False,
    output_dir: str | None = None,
    imgsz: int = 1008,
    dynamic: bool = False,
    min_imgsz: int | None = None,
    prefix: str = "SAM3 ONNX:",
) -> list[str]:
    """Export SAM3SemanticModel as separate ONNX files from a .pt checkpoint.

    Args:
        checkpoint_path: Path to SAM3 checkpoint (.pt).
        device: Device for export (cpu recommended to avoid OOM).
        opset: ONNX opset version (20 recommended for native Gelu op).
        half: FP16 ONNX export (for ONNX-only deployment, not TRT).
        output_dir: Parent directory for output folder.
        imgsz: Image size (must be divisible by 14). With ``dynamic`` this is the largest accepted size.
        dynamic: Accept any image size from ``min_imgsz`` to ``imgsz`` instead of only ``imgsz``.
        min_imgsz: Smallest accepted size when ``dynamic``, recorded so TensorRT can size its profiles. Defaults to half
            of ``imgsz``, rounded down to a whole number of patches.
        prefix: Log prefix.

    Returns:
        (list[str]): Exported ONNX file paths, four without the point prompt modules and six with.
    """
    from ultralytics.models.sam.build_sam3 import build_sam3_image_model
    from ultralytics.utils.checks import check_requirements
    from ultralytics.utils.export.engine import torch2onnx

    check_requirements(["onnx>=1.12.0,<2.0.0"])
    import onnx

    device = torch.device(device) if isinstance(device, str) else device
    assert imgsz % 14 == 0, f"imgsz={imgsz} must be divisible by patch_size=14"
    if dynamic:
        min_imgsz = min_imgsz or (imgsz // 28) * 14
        assert min_imgsz % 14 == 0, f"min_imgsz={min_imgsz} must be divisible by patch_size=14"
        assert min_imgsz <= imgsz, f"min_imgsz={min_imgsz} must not exceed imgsz={imgsz}"
        LOGGER.info(f"{prefix} tracing dynamic graphs accepting {min_imgsz} to {imgsz}...")

    LOGGER.info(f"\n{prefix} building SAM3SemanticModel from {checkpoint_path}...")
    model = build_sam3_image_model(checkpoint_path, enable_segmentation=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    if imgsz != 1008:
        LOGGER.info(f"{prefix} setting image size to {imgsz}x{imgsz}...")
        model.set_imgsz((imgsz, imgsz))

    _prepare_for_onnx_export(model, dynamic=dynamic)

    dtype = torch.float32
    if half and device.type != "cpu":
        model = model.half()
        dtype = torch.float16

    if output_dir is None:
        output_dir = str(Path(checkpoint_path).parent)
    output_path = Path(output_dir) / f"{Path(checkpoint_path).stem}_onnx"
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = {"author": "Ultralytics", "task": "segment", "stride": 14, "imgsz": [imgsz, imgsz]}
    if dynamic:
        metadata["min_imgsz"] = [min_imgsz, min_imgsz]
    exported_files = []

    # Each FPN level has its own resolution, so it needs its own symbolic name. Sharing one name
    # across levels builds and runs in ONNX Runtime but TensorRT reads it as a single dimension and
    # rejects the optimization profile as self contradictory. Tensors on the same level do share.
    def _lvl(*names):
        """Return the dynamic axes entry pairing each named tensor with its FPN level grid."""
        return {n: {2: f"h{i}", 3: f"w{i}"} for n, i in names} if dynamic else {}

    mask_axes = {"pred_masks": {2: "mh", 3: "mw"}} if dynamic else {}

    def _export(module, args, name, input_names, output_names, dynamic_axes=None):
        """Trace one module to ONNX with the shared export options and record the path."""
        f = str(output_path / name)
        torch2onnx(
            module, args, f, opset=opset, input_names=input_names, output_names=output_names, dynamic=dynamic_axes
        )
        exported_files.append(f)
        return f

    # Vision Encoder
    # Load SAM2 neck weights from the interactive model (separate learned FPN for point prompts)
    from ultralytics.models.sam.build_sam3 import build_interactive_sam3

    LOGGER.info(f"{prefix} loading interactive model for SAM2 neck weights...")
    # Keep the interactive model on the CPU and move only the exported pieces to the device: the two
    # multi gigabyte models do not fit on one GPU at the same time. Freeze and half it like the
    # semantic model so the traced dtypes match.
    tracker_model_for_neck = build_interactive_sam3(checkpoint_path).eval()
    for p in tracker_model_for_neck.parameters():
        p.requires_grad = False
    if dtype == torch.float16:
        tracker_model_for_neck = tracker_model_for_neck.half()
    sam2_convs = tracker_model_for_neck.image_encoder.vision_backbone.sam2_convs.to(device)
    # SAM 3.1 restructured the tracker, so its interactive weights do not load and must never be exported untrained
    # Only neck levels 0 to 2 are checked because scalp discards the last level, which SAM 3.1 no longer ships
    point_modules = ("sam_prompt_encoder", "sam_mask_decoder", "sam2_convs.0", "sam2_convs.1", "sam2_convs.2")
    has_point_weights = not any(m in k for k in tracker_model_for_neck.missing_keys for m in point_modules)
    # The interactive model is built at 1008, so resize it too or its prompt encoder would emit a
    # 72x72 dense embedding that cannot pair with an imgsz/14 feature map in the mask decoder.
    if has_point_weights and imgsz != 1008:
        LOGGER.info(f"{prefix} setting interactive model image size to {imgsz}x{imgsz}...")
        tracker_model_for_neck.set_imgsz((imgsz, imgsz))
    if not has_point_weights:
        sam2_convs = None
    if sam2_convs is None:
        LOGGER.warning(f"{prefix} interactive weights are unavailable so point prompt modules are skipped")

    LOGGER.info(f"{prefix} exporting vision encoder with dual neck (opset {opset})...")
    vis_encoder = (
        SAM3VisionEncoderONNX(model, imgsz=imgsz, sam2_convs=sam2_convs, dynamic=dynamic, max_imgsz=imgsz)
        .to(device)
        .eval()
    )
    dummy_image = torch.randn(1, 3, imgsz, imgsz, dtype=dtype, device=device)

    output_names_vis = ["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"]
    if sam2_convs is not None:
        output_names_vis += ["sam2_feat_0", "sam2_feat_1", "sam2_feat_2"]

    # fpn_pos_2 shares level 2's grid with fpn_feat_2, and each sam2 output shares its FPN level's
    vis_axes = _lvl(*_FPN_TENSORS)
    if dynamic:
        vis_axes["images"] = {2: "height", 3: "width"}
        if sam2_convs is not None:
            vis_axes.update(_lvl(("sam2_feat_0", 0), ("sam2_feat_1", 1), ("sam2_feat_2", 2)))

    _export(vis_encoder, (dummy_image,), "sam3_vision_encoder.onnx", ["images"], output_names_vis, vis_axes or None)

    with torch.no_grad():
        vis_out = vis_encoder(dummy_image)
    # Only the four FPN tensors are reused as dummy inputs for the decoder and mask decoder exports;
    # the optional sam2_feat_* outputs (when sam2_convs is present) are not needed here.
    fpn0, fpn1, fpn2, fpos2 = vis_out[:4]

    if not has_point_weights:
        del tracker_model_for_neck  # nothing else needs it, so release it before tracing the rest

    # Text Encoder
    LOGGER.info(f"{prefix} exporting text encoder (opset {opset})...")
    txt_encoder = SAM3TextEncoderONNX(model).to(device).eval()
    dummy_tokens = torch.zeros(1, 32, dtype=torch.long, device=device)
    dummy_tokens[0, :3] = torch.tensor([49406, 2533, 49407])

    _export(txt_encoder, (dummy_tokens,), "sam3_text_encoder.onnx", ["tokens"], ["text_features", "text_mask"])

    with torch.no_grad():
        txt_feats, txt_mask = txt_encoder(dummy_tokens)

    # Decoder (with folded geometry encoder)
    LOGGER.info(f"{prefix} exporting decoder (opset {opset})...")
    decoder = SAM3DecoderONNX(model).to(device).eval()
    dec_axes = {
        **_lvl(*_FPN_TENSORS),
        **mask_axes,
    }

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
        {
            "input_boxes": {1: "num_boxes"},
            "input_boxes_labels": {1: "num_boxes"},
            **dec_axes,
        },
    )

    # Text only decoder. A box prompt appends geometry tokens, and even a box labeled as ignored
    # shifts the presence logit and suppresses detections, so text prompts need a graph without them.
    LOGGER.info(f"{prefix} exporting text only decoder (opset {opset})...")
    _export(
        SAM3DecoderONNX(model, with_geometry=False).to(device).eval(),
        (fpn0, fpn1, fpn2, fpos2, txt_feats, txt_mask),
        "sam3_decoder_text.onnx",
        ["fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2", "prompt_features", "prompt_mask"],
        ["pred_logits", "pred_boxes", "pred_masks", "presence_logit_dec"],
        dec_axes or None,
    )

    if has_point_weights:
        # SAM Prompt Encoder (for point prompts). Reuse the model already built for the neck rather
        # than loading the multi gigabyte checkpoint a second time.
        LOGGER.info(f"{prefix} exporting SAM prompt encoder (opset {opset})...")
        tracker_model = tracker_model_for_neck

        prompt_enc = SAM3PromptEncoderONNX(tracker_model, dynamic=dynamic).to(device).eval()
        dummy_pts = torch.tensor([[[500.0, 500.0]]], dtype=dtype, device=device)
        dummy_lbl = torch.tensor([[1]], dtype=torch.int32, device=device)
        # Everything this module bakes in follows from the feature grid, so when dynamic it reads the
        # grid off the image embedding the mask decoder is given anyway.
        pe_args = (dummy_pts, dummy_lbl, fpn2) if dynamic else (dummy_pts, dummy_lbl)
        pe_names = ["point_coords", "point_labels"] + (["image_embeddings"] if dynamic else [])

        _export(
            prompt_enc,
            pe_args,
            "sam3_prompt_encoder.onnx",
            pe_names,
            ["sparse_embeddings", "dense_embeddings", "dense_pe"],
            {
                "point_coords": {1: "num_points"},
                "point_labels": {1: "num_points"},
                "sparse_embeddings": {1: "num_embeds"},
                **_lvl(("image_embeddings", 2), ("dense_embeddings", 2), ("dense_pe", 2)),
            },
        )

        # SAM Mask Decoder (for point prompts). Export with multimask_output=False, which is what the
        # PyTorch predictor uses. SAM3 enables dynamic_multimask_via_stability, so that path returns
        # the most stable mask instead of the highest scoring one. Exporting the three candidates and
        # picking argmax(iou) at runtime chooses a different, often broken, mask because the score
        # head ranks a shredded candidate above the clean one.
        LOGGER.info(f"{prefix} exporting SAM mask decoder (opset {opset}, stability selected)...")
        mask_dec = SAM3MaskDecoderONNX(tracker_model, multimask_output=False).to(device).eval()

        with torch.no_grad():
            sparse_dummy, dense_dummy, dpe_dummy = prompt_enc(*pe_args)

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
            {
                "sparse_prompt_embeddings": {1: "num_embeds"},
                **_lvl(
                    ("image_embeddings", 2),
                    ("image_pe", 2),
                    ("dense_prompt_embeddings", 2),
                    ("high_res_feat_0", 0),
                    ("high_res_feat_1", 1),
                ),
                **({"masks": {2: "mh", 3: "mw"}} if dynamic else {}),
            },
        )

        del tracker_model

    # Post-processing
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
        (list[str]): Built TensorRT engine file paths, one per exported ONNX module.
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
    if (onnx_dir / "sam3_decoder_text.onnx").exists():
        onnx_files.append(onnx_dir / "sam3_decoder_text.onnx")
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
            input_shape = (*input_shape, 1)
        input_shape = input_shape[:4]

        # Carry the ONNX metadata into the engine so the backend can recover the size the graph was
        # traced at. Without it an engine directory cannot say what imgsz it needs.
        engine_metadata = {p.key: p.value for p in model_onnx.metadata_props}
        engine_metadata.update(component=onnx_file.stem, author="Ultralytics", task="segment")

        # A dynamic image size export records the range it accepts, and every module then needs a
        # profile whose spatial bounds follow from it, including the otherwise static vision encoder.
        spatial = None
        if "min_imgsz" in engine_metadata:
            spatial = _spatial_profile(
                model_onnx,
                min_imgsz=ast.literal_eval(engine_metadata["min_imgsz"])[0],
                max_imgsz=ast.literal_eval(engine_metadata["imgsz"])[0],
                stride=int(engine_metadata.get("stride", 14)),
            )

        engine_file = str(engine_dir / onnx_file.name.replace(".onnx", ".engine"))

        # Modules with a dynamic axis need a custom build with an optimization profile. They honor
        # FP16 through mixed precision (ModelOpt AutoCast keeps overflow prone nodes in FP32), which
        # keeps the detection decoder accurate and builds identically on TensorRT 10 and 11. The
        # static vision and text encoders go through onnx2engine.
        shared = {"workspace": workspace, "metadata": engine_metadata, "verbose": verbose, "prefix": prefix}
        if onnx_file.stem in _DYNAMIC_MODULES or spatial:
            _build_decoder_engine_dynamic(str(onnx_file), engine_file, half=half, spatial=spatial, **shared)
        else:
            onnx2engine(
                str(onnx_file), engine_file, quantize=16 if half else None, dynamic=False, shape=input_shape, **shared
            )
        exported_engines.append(engine_file)
        LOGGER.info(f"{prefix} saved {Path(engine_file).name} ({'mixed FP16' if half else 'FP32'})")

    LOGGER.info(f"{prefix} export complete -> {engine_dir}")
    return exported_engines


# Modules with a symbolic dimension need the custom builder with an optimization profile.
_DYNAMIC_MODULES = frozenset({"sam3_decoder", "sam3_decoder_text", "sam3_prompt_encoder", "sam3_mask_decoder"})

# An FPN level's grid is this multiple of the patch grid, so a level's bounds follow from the image size
_FPN_LEVEL_SCALE = (4, 2, 1)

# Every tensor the vision encoder and decoder exchange, paired with the FPN level whose grid it is on
_FPN_TENSORS = (("fpn_feat_0", 0), ("fpn_feat_1", 1), ("fpn_feat_2", 2), ("fpn_pos_2", 2))


def _spatial_profile(
    model_onnx, min_imgsz: int, max_imgsz: int, stride: int = 14, token_bounds: tuple[int, int] = (1, 32)
) -> dict[str, tuple]:
    """Map each dynamically sized input to the (min, opt, max) shapes its optimization profile needs.

    A profile is per input and the FPN levels sit at different multiples of the patch grid, so the level is read back
    off the symbolic dimension name the export gave it and turned into a grid range.

    Args:
        model_onnx (onnx.ModelProto): The loaded graph to read input shapes from.
        min_imgsz (int): Smallest image size the graph accepts.
        max_imgsz (int): Largest image size the graph accepts.
        stride (int): ViT patch size.
        token_bounds (tuple[int, int]): Bounds for a non spatial symbolic dimension sharing an input with a spatial one.

    Returns:
        (dict[str, tuple]): Input name to (min shape, opt shape, max shape), only for spatial inputs.
    """
    lo, hi = min_imgsz // stride, max_imgsz // stride
    bounds = {"height": (min_imgsz, max_imgsz), "width": (min_imgsz, max_imgsz)}
    for level, scale in enumerate(_FPN_LEVEL_SCALE):
        bounds[f"h{level}"] = bounds[f"w{level}"] = (lo * scale, hi * scale)

    profile = {}
    for inp in model_onnx.graph.input:
        dims = inp.type.tensor_type.shape.dim
        if not any(d.dim_param in bounds for d in dims):
            continue
        shapes = [[], [], []]
        for d in dims:
            if d.dim_param in bounds:
                span = bounds[d.dim_param]
            elif d.dim_param:  # a non spatial symbolic dim sharing this input, such as a token count
                span = token_bounds
            else:
                span = (d.dim_value, d.dim_value)
            for s, v in zip(shapes, (span[0], span[1], span[1])):
                s.append(v)
        profile[inp.name] = tuple(tuple(s) for s in shapes)
    return profile


def _gridsample_mode_for_trt(onnx_file: str, prefix: str) -> bytes | None:
    """Return the ONNX bytes with GridSample renamed from the opset 20 mode to the TensorRT one, or None if untouched.

    TensorRT does not recognize the opset 20 ``linear`` mode name and silently samples nearest neighbor instead, which
    corrupts the ROI features behind box prompts. onnxruntime accepts only ``linear``, so the rename is applied to the
    bytes handed to the TensorRT parser and the exported ONNX file stays spec compliant. Only the detection decoder
    holds a GridSample, so every other module returns None and is parsed straight from disk instead: serializing a multi
    gigabyte encoder to memory for a rename that never fires costs several gigabytes and runs a graph that is already
    1.9 GB at FP32 into the 2 GB protobuf ceiling.
    """
    import onnx

    model = onnx.load(onnx_file)
    renamed = 0
    for node in model.graph.node:
        if node.op_type == "GridSample":
            for attr in node.attribute:
                if attr.name == "mode" and attr.s == b"linear":
                    attr.s = b"bilinear"
                    renamed += 1
    if not renamed:
        return None
    LOGGER.info(f"{prefix} renamed {renamed} GridSample mode to bilinear for the TensorRT parser")
    return model.SerializeToString()


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
    spatial: dict[str, tuple] | None = None,
) -> None:
    """Build a TensorRT engine for an ONNX module with dynamic dimensions.

    Detects symbolic dims and adds an optimization profile [min_dynamic, opt_dynamic, max_dynamic], except for the
    spatial inputs of a dynamic image size export, whose per FPN level bounds are passed in as ``spatial``. With
    ``half`` the module is converted to mixed FP16/FP32 by ModelOpt AutoCast and built as a strongly-typed network, so
    the per-node precision is honored identically on TensorRT 10 and 11 (the FP16 builder flag was removed in TensorRT
    11). Without ``half`` the engine is FP32.
    """
    import onnx

    from ultralytics.utils.export.engine import onnx2engine

    if half:
        from ultralytics.utils.export.engine import modelopt_quantize_onnx

        # AutoCast runs a reference pass on CPU, and the vision encoder at the largest profile shape
        # needs over a hundred gigabytes for it. Only the ranges it collects matter, so use the smallest.
        onnx_file = modelopt_quantize_onnx(
            onnx_file,
            quantize=16,
            dynamic_dim=opt_dynamic,
            calib_shapes={name: lo for name, (lo, _, _) in (spatial or {}).items()},
            prefix=prefix,
        )

    # onnx2engine wants one (min, opt, max) profile per input: the precomputed per FPN level bounds for
    # spatial inputs, the generic dynamic bounds for other symbolic dims, and the declared shape otherwise.
    spatial = spatial or {}
    profile_shapes = {}
    for inp in onnx.load(onnx_file, load_external_data=False).graph.input:
        dims = [d.dim_value if d.dim_value > 0 else -1 for d in inp.type.tensor_type.shape.dim]
        if inp.name in spatial:
            profile_shapes[inp.name] = spatial[inp.name]
        elif -1 in dims:
            profile_shapes[inp.name] = tuple(
                tuple(bound if d == -1 else d for d in dims) for bound in (min_dynamic, opt_dynamic, max_dynamic)
            )
        else:
            profile_shapes[inp.name] = (tuple(dims),) * 3

    onnx2engine(
        onnx_file,
        engine_file,
        workspace=workspace,
        metadata=metadata,
        verbose=verbose,
        prefix=prefix,
        profile_shapes=profile_shapes,
        strongly_typed=half,
        onnx_bytes=_gridsample_mode_for_trt(onnx_file, prefix),
    )
