# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""DEIM decoder modules: D-FINE-style fine-grained distribution refinement with DEIMv2 layer upgrades.

The ``DEIMTransformerDecoder`` here is distinct from ``transformer.DeformableTransformerDecoder`` used by the RT-DETR
head: it refines boxes by iteratively integrating a learned distribution over bin edges (FDR) instead of regressing
deltas directly, and its cross-attention (``MSDeformableAttention``) carries no value/output projections, unlike
``transformer.MSDeformAttn``.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .block import DFL
from .transformer import MLP
from .utils import (
    bias_init_with_prob,
    distance2bbox,
    inverse_sigmoid,
    multi_scale_deformable_attn_pytorch,
    weighting_function,
)


class MSDeformableAttention(nn.Module):
    """Multi-scale deformable attention sampling a small set of points per head and feature level.

    Attributes:
        embed_dim (int): Feature dimension of the query and value tensors.
        num_heads (int): Number of attention heads.
        num_levels (int): Number of feature levels sampled from.
        offset_scale (float): Multiplier applied to the predicted sampling offsets.
        num_points_list (list[int]): Number of sampling points for each feature level.
        num_points_scale (torch.Tensor): Per-point normalizer used when reference points carry width and height.
        total_points (int): Total number of sampling points across all heads and levels.
        head_dim (int): Feature dimension of a single attention head.
        sampling_offsets (nn.Linear): Projection producing the per-point sampling offsets.
        attention_weights (nn.Linear): Projection producing the per-point attention weights.
    """

    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        offset_scale=0.5,
    ):
        """Initialize the deformable attention module.

        Args:
            embed_dim (int): Feature dimension of the query and value tensors.
            num_heads (int): Number of attention heads.
            num_levels (int): Number of feature levels sampled from.
            num_points (int | list[int]): Sampling points per head, shared across levels or one entry per level.
            offset_scale (float): Multiplier applied to the predicted sampling offsets.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ""
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list

        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer("num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize sampling offsets on a ring of head-specific directions and zero the attention weights."""
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

    def forward(
        self, query: torch.Tensor, reference_points: torch.Tensor, value: torch.Tensor, value_spatial_shapes: list
    ):
        """Sample the value tensor at the predicted offsets and combine the samples with the attention weights.

        Args:
            query (torch.Tensor): Queries with shape (B, N, C).
            reference_points (torch.Tensor): Reference points with shape (B, N, n_levels, D), where D is 2 for locations
                normalized to [0, 1] from the top-left corner or 4 for boxes in xywh format.
            value (torch.Tensor): Encoder memory with shape (B, L, C).
            value_spatial_shapes (list): Height and width of each feature level, with shape (n_levels, 2).

        Returns:
            (torch.Tensor): Attended features with shape (B, N, C).
        """
        bs, Len_q = query.shape[:2]

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = (
                reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
            )
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]
            # sampling_offsets [8, 480, 8,    12, 2]
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(reference_points.shape[-1])
            )

        value = value.reshape(value.shape[0], value.shape[1], self.num_heads, self.head_dim)
        output = multi_scale_deformable_attn_pytorch(
            value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list
        )

        return output


class Integral(DFL):
    """DFL over the non-uniform D-FINE bin centers W(n) instead of uniform integer bins.

    Same softmax-then-integrate operation as `DFL`, but takes the channel-last corner logits used by the DEIM decoder.
    """

    def __init__(self, reg_max: int = 32, up: torch.Tensor | None = None, reg_scale: torch.Tensor | float = 4.0):
        """Initialize the integral layer with bin centers from the Weighting Function W(n).

        Args:
            reg_max (int): Max number of the discrete bins.
            up (torch.Tensor, optional): Upper bound controlling the non-uniform bin spacing.
            reg_scale (torch.Tensor | float): Scale controlling the non-uniform bin spacing.
        """
        up = torch.tensor([0.5]) if up is None else up
        bins = weighting_function(reg_max, up, torch.atleast_1d(torch.as_tensor(reg_scale, dtype=torch.float32)))
        super().__init__(reg_max + 1, bins=bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Integrate corner logits with shape (..., 4 * (reg_max + 1)) into distance offsets with shape (..., 4)."""
        shape = x.shape
        return super().forward(x.reshape(-1, 4 * self.c1, 1)).reshape(*shape[:-1], 4)


class LQE(nn.Module):
    """Location quality estimator adding a distribution-derived confidence correction to the class scores.

    Attributes:
        k (int): Number of top distribution bins summarized per box edge.
        reg_max (int): Max number of the discrete bins.
        reg_conf (MLP): Head mapping the distribution statistics to a scalar quality score.
    """

    def __init__(self, k, hidden_dim, num_layers, reg_max, act=nn.ReLU):
        """Initialize the location quality estimator.

        Args:
            k (int): Number of top distribution bins summarized per box edge.
            hidden_dim (int): Hidden width of the quality head.
            num_layers (int): Number of layers in the quality head.
            reg_max (int): Max number of the discrete bins.
            act (nn.Module): Activation used by the quality head.
        """
        super().__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers, act=act)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        """Add a quality correction derived from the corner distribution to the class scores.

        Args:
            scores (torch.Tensor): Class scores with shape (B, N, C).
            pred_corners (torch.Tensor): Corner logits with shape (B, N, 4 * (reg_max + 1)).

        Returns:
            (torch.Tensor): Corrected class scores with shape (B, N, C).
        """
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(-1, self.reg_max + 1), dim=-1)
        topk_ind = prob.topk(self.k, dim=-1).indices
        prob_topk = prob.gather(-1, topk_ind).reshape(B, L, 4, self.k)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score


class DEIMTransformerDecoder(nn.Module):
    """DEIM transformer decoder implementing Fine-grained Distribution Refinement (FDR).

    This decoder refines object detection predictions through iterative updates across multiple layers, utilizing
    attention mechanisms, location quality estimators, and distribution refinement techniques to improve bounding box
    accuracy and robustness. Query-position embeddings are computed once from the initial reference boxes and held
    fixed across layers (DEIMv2 behavior).
    """

    def __init__(
        self, decoder_layer, decoder_layer_wide, num_layers, reg_max, eval_idx=-1, layer_scale=2, act=nn.ReLU()
    ):
        """Initialize the decoder.

        Args:
            decoder_layer (nn.Module): Layer prototype deep-copied for every layer up to eval_idx.
            decoder_layer_wide (nn.Module): Layer prototype deep-copied for the layers after eval_idx.
            num_layers (int): Total number of decoder layers.
            reg_max (int): Max number of the discrete bins.
            eval_idx (int): Layer index used at inference; negative values count back from the last layer.
            layer_scale (int): Width multiplier applied to the layers after eval_idx.
            act (nn.Module): Activation used by the location quality estimators.
        """
        super().__init__()
        self.layer_scale = layer_scale
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)]
        )
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max, act=act)) for _ in range(num_layers)])

    @staticmethod
    def value_op(memory, value_scale, memory_mask):
        """Resize and mask the encoder memory for MSDeformableAttention.

        Args:
            memory (torch.Tensor): Encoder memory with shape (B, L, C).
            value_scale (int, optional): Target width for interpolation of the memory.
            memory_mask (torch.Tensor, optional): Validity mask with shape (B, L).

        Returns:
            (torch.Tensor): Masked value tensor with shape (B, L, C).
        """
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else memory
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        return value

    def forward(
        self,
        target,
        ref_points_unact,
        memory,
        spatial_shapes,
        bbox_head,
        score_head,
        query_pos_head,
        pre_bbox_head,
        integral,
        reg_scale,
        attn_mask=None,
        memory_mask=None,
    ):
        """Refine the queries layer by layer, accumulating corner corrections through the decoder.

        Args:
            target (torch.Tensor): Initial decoder queries with shape (B, N, C).
            ref_points_unact (torch.Tensor): Unactivated reference boxes with shape (B, N, 4).
            memory (torch.Tensor): Encoder memory with shape (B, L, C).
            spatial_shapes (list[list[int]]): Height and width of each feature level.
            bbox_head (nn.ModuleList): Per-layer heads producing the corner distribution corrections.
            score_head (nn.ModuleList): Per-layer heads producing the class scores.
            query_pos_head (nn.Module): Head embedding the reference boxes into query positions.
            pre_bbox_head (nn.Module): Head producing the initial box prediction of the first layer.
            integral (Integral): Layer integrating a corner distribution into distance offsets.
            reg_scale (torch.Tensor): Scale controlling the non-uniform bin spacing.
            attn_mask (torch.Tensor, optional): Self-attention mask isolating the denoising groups.
            memory_mask (torch.Tensor, optional): Validity mask for the encoder memory.

        Returns:
            dec_out_bboxes (torch.Tensor): Boxes per layer in xywh format with shape (L, B, N, 4).
            dec_out_logits (torch.Tensor): Class scores per layer with shape (L, B, N, C).
            dec_out_pred_corners (torch.Tensor): Corner logits per layer with shape (L, B, N, 4 * (reg_max + 1)).
            dec_out_refs (torch.Tensor): Reference boxes per layer with shape (L, B, N, 4).
            pre_bboxes (torch.Tensor): Initial box prediction of the first layer with shape (B, N, 4).
            pre_scores (torch.Tensor): Initial class scores of the first layer with shape (B, N, C).

        Notes:
            At inference only the layers up to eval_idx run, so L is 1 in the returned stacks.
        """
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, memory_mask)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []

        ref_points_detach = F.sigmoid(ref_points_unact)
        query_pos_fixed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_fixed

            # TODO Adjust scale if needed for detachable wider layers
            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                query_pos_fixed = query_pos_embed
                value = self.value_op(memory, query_pos_embed.shape[-1], memory_mask)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)

            if i == 0:
                # Initial bounding box predictions with inverse sigmoid refinement
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            # Refine bounding box corners using FDR, integrating previous layer's corrections
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners), reg_scale)

            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                # Lqe does not affect the performance here.
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)

                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores,
        )


class DEIMRMSNorm(nn.Module):
    """RMSNorm used by DEIMv2 decoder layers."""

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize the normalization layer.

        Args:
            dim (int): Feature dimension normalized over.
            eps (float): Value added to the mean square before the reciprocal square root.
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """Normalize the last dimension by its root mean square and apply the learned scale.

        Args:
            x (torch.Tensor): Input tensor with shape (..., dim).

        Returns:
            (torch.Tensor): Normalized tensor with the same shape and dtype as the input.

        Notes:
            The statistics are computed in float32 and cast back, so the layer stays stable under autocast.
        """
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.scale


class DEIMSwiGLUFFN(nn.Module):
    """SwiGLU FFN used by DEIMv2 decoder layers."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = True) -> None:
        """Initialize the feed-forward network.

        Args:
            in_features (int): Input feature dimension.
            hidden_features (int): Hidden feature dimension of each of the two gate branches.
            out_features (int): Output feature dimension.
            bias (bool): Add a learnable bias to both projections.
        """
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        """Apply Xavier uniform initialization to both projections and zero their biases."""
        init.xavier_uniform_(self.w12.weight)
        init.constant_(self.w12.bias, 0)
        init.xavier_uniform_(self.w3.weight)
        init.constant_(self.w3.bias, 0)

    def forward(self, x):
        """Apply the SwiGLU transform.

        Args:
            x (torch.Tensor): Input tensor with shape (..., in_features).

        Returns:
            (torch.Tensor): Output tensor with shape (..., out_features) and the input dtype.

        Notes:
            The projections run in float32 on CUDA and the result is cast back, keeping the gate product from
            overflowing under autocast.
        """
        with torch.autocast(device_type=x.device.type, dtype=torch.float32, enabled=x.is_cuda):
            x1, x2 = self.w12(x.float()).chunk(2, dim=-1)
            return self.w3(F.silu(x1) * x2).to(x.dtype)


class DEIMGate(nn.Module):
    """DEIM gate with optional RMSNorm."""

    def __init__(self, d_model: int, use_rmsnorm: bool = False):
        """Initialize the gate.

        Args:
            d_model (int): Feature dimension of each of the two gated inputs.
            use_rmsnorm (bool): Normalize the gated sum with RMSNorm instead of LayerNorm.
        """
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = DEIMRMSNorm(d_model) if use_rmsnorm else nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        """Blend two feature streams with learned per-channel gates and normalize the result.

        Args:
            x1 (torch.Tensor): First input with shape (B, N, d_model).
            x2 (torch.Tensor): Second input with shape (B, N, d_model).

        Returns:
            (torch.Tensor): Normalized gated sum with shape (B, N, d_model).
        """
        gate1, gate2 = torch.sigmoid(self.gate(torch.cat([x1, x2], dim=-1))).chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class DEIMTransformerDecoderLayer(nn.Module):
    """DEIMv2 decoder layer (RMSNorm + SwiGLU)."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        n_levels: int = 4,
        n_points: int = 4,
        layer_scale=None,
        use_gateway: bool = False,
        use_rmsnorm: bool = True,
    ):
        """Initialize the decoder layer.

        Args:
            d_model (int): Feature dimension of the queries.
            n_heads (int): Number of attention heads.
            d_ffn (int): Hidden width of the feed-forward network before the SwiGLU halving.
            dropout (float): Dropout probability applied after each sublayer.
            n_levels (int): Number of feature levels sampled by the cross-attention.
            n_points (int): Sampling points per head and level in the cross-attention.
            layer_scale (float, optional): Width multiplier applied to both d_model and d_ffn.
            use_gateway (bool): Merge the cross-attention output with a learned gate instead of a residual add.
            use_rmsnorm (bool): Use RMSNorm instead of LayerNorm.
        """
        super().__init__()
        if layer_scale is not None:
            d_ffn = round(layer_scale * d_ffn)
            d_model = round(layer_scale * d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        norm_layer = DEIMRMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm1 = norm_layer(d_model)

        self.cross_attn = MSDeformableAttention(d_model, n_heads, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.use_gateway = use_gateway
        if use_gateway:
            self.gateway = DEIMGate(d_model, use_rmsnorm=use_rmsnorm)
        else:
            self.norm2 = norm_layer(d_model)

        self.swish_ffn = DEIMSwiGLUFFN(d_model, d_ffn // 2, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = norm_layer(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Add a positional embedding to a tensor when one is supplied.

        Args:
            tensor (torch.Tensor): Input tensor with shape (B, N, C).
            pos (torch.Tensor, optional): Positional embedding with shape (B, N, C).

        Returns:
            (torch.Tensor): Input with the positional embedding added, or the input unchanged when pos is None.
        """
        return tensor if pos is None else tensor + pos

    def forward(self, target, reference_points, value, spatial_shapes, attn_mask=None, query_pos_embed=None):
        """Apply self-attention, deformable cross-attention, and the SwiGLU feed-forward network.

        Args:
            target (torch.Tensor): Decoder queries with shape (B, N, d_model).
            reference_points (torch.Tensor): Reference boxes with shape (B, N, 1, 4).
            value (torch.Tensor): Encoder memory with shape (B, L, d_model).
            spatial_shapes (list[list[int]]): Height and width of each feature level.
            attn_mask (torch.Tensor, optional): Self-attention mask isolating the denoising groups.
            query_pos_embed (torch.Tensor, optional): Query position embedding with shape (B, N, d_model).

        Returns:
            (torch.Tensor): Updated queries with shape (B, N, d_model).

        Notes:
            Self-attention runs in float32 on CUDA and the residual sum is clamped to the fp16 range before the
            final norm, which keeps the layer finite under autocast.
        """
        q = k = self.with_pos_embed(target, query_pos_embed)
        with torch.autocast(device_type=target.device.type, dtype=torch.float32, enabled=target.is_cuda):
            target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target2 = target2.to(target.dtype)
        target = self.norm1(target + self.dropout1(target2))

        target2 = self.cross_attn(self.with_pos_embed(target, query_pos_embed), reference_points, value, spatial_shapes)
        if self.use_gateway:
            target = self.gateway(target, self.dropout2(target2))
        else:
            target = self.norm2(target + self.dropout2(target2))

        target2 = self.swish_ffn(target)
        return self.norm3((target + self.dropout4(target2)).clamp(min=-65504, max=65504))
