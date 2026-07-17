import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .transformer import MLP
import torch.nn.init as init
import math
import copy
from .utils import inverse_sigmoid, multi_scale_deformable_attn_pytorch
from .dfine_utils import weighting_function, distance2bbox


def bias_init_with_prob(prior_prob=0.01):
    """Initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init

class MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        method='default',
        offset_scale=0.5,
    ):
        """Multi-Scale Deformable Attention."""
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list

        num_points_scale = [1/n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)

        self._reset_parameters()

        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
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

    def forward(self,
                query: torch.Tensor,
                reference_points: torch.Tensor,
                value: torch.Tensor,
                value_spatial_shapes: List[int]):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0), bottom-right
                (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]
            # sampling_offsets [8, 480, 8,    12, 2]
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        value = value.reshape(value.shape[0], value.shape[1], self.num_heads, self.head_dim)
        output = multi_scale_deformable_attn_pytorch(
            value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list
        )

        return output


class Integral(nn.Module):
    """A static layer that calculates integral results from a distribution.

    This layer computes the target location using the formula: `sum{Pr(n) * W(n)}`, where Pr(n) is the softmax
    probability vector representing the discrete distribution, and W(n) is the non-uniform Weighting Function.

    Args:
        reg_max (int): Max number of the discrete bins. Default is 32. It can be adjusted based on the dataset or task
            requirements.
    """

    def __init__(self, reg_max=32):
        super(Integral, self).__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        # Match both device AND dtype of activations: `project` is a precomputed (deploy-mode) fp32 plain attribute
        # that model.half() does not convert, so under FP16 export a `.to(x.device)`-only cast leaves it fp32 and
        # F.linear hits "mat1 and mat2 must have the same dtype" (Half != float).
        x = F.linear(x, project.to(device=x.device, dtype=x.dtype).reshape(1, -1)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max, act=nn.ReLU):
        super(LQE, self).__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers, act=act)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max+1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score


class DFineTransformerDecoder(nn.Module):
    """Transformer Decoder implementing Fine-grained Distribution Refinement (FDR).

    This decoder refines object detection predictions through iterative updates across multiple layers, utilizing
    attention mechanisms, location quality estimators, and distribution refinement techniques to improve bounding box
    accuracy and robustness.
    """

    def __init__(
            self,
            hidden_dim,
            decoder_layer,
            decoder_layer_wide,
            num_layers,
            num_head,
            reg_max,
            reg_scale,
            up,
            eval_idx=-1,
            layer_scale=2,
            act=nn.ReLU()):
        super(DFineTransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList([
            copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)] +
            [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)])
        self.lqe_layers = nn.ModuleList([copy.deepcopy(LQE(4, 64, 2, reg_max, act=act)) for _ in range(num_layers)])
        self.fixed_query_pos = False

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        """Mask encoder memory for MSDeformableAttention; returns shape (bs, L, C)."""
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        return value

    def convert_to_deploy(self):
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[:self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList([nn.Identity()] * (self.eval_idx) + [self.lqe_layers[self.eval_idx]])

    def forward(self,
                target,
                ref_points_unact,
                memory,
                spatial_shapes,
                bbox_head,
                score_head,
                query_pos_head,
                pre_bbox_head,
                integral,
                up,
                reg_scale,
                attn_mask=None,
                memory_mask=None,
                dn_meta=None):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        if not hasattr(self, 'project'):
            project = weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project

        ref_points_detach = F.sigmoid(ref_points_unact)
        query_pos_fixed = query_pos_head(ref_points_detach).clamp(min=-10, max=10) if self.fixed_query_pos else None

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_fixed
            if query_pos_embed is None:
                query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

            # TODO Adjust scale if needed for detachable wider layers
            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                if self.fixed_query_pos:
                    query_pos_fixed = query_pos_embed
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
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
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

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

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), \
               torch.stack(dec_out_pred_corners), torch.stack(dec_out_refs), pre_bboxes, pre_scores


class DEIMRMSNorm(nn.Module):
    """RMSNorm used by DEIMv2 decoder layers."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.scale


class DEIMSwiGLUFFN(nn.Module):
    """SwiGLU FFN used by DEIMv2 decoder layers."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.w12.weight)
        init.constant_(self.w12.bias, 0)
        init.xavier_uniform_(self.w3.weight)
        init.constant_(self.w3.bias, 0)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class DeimGate(nn.Module):
    """DEIM gate with optional RMSNorm."""

    def __init__(self, d_model: int, use_rmsnorm: bool = False):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, bias)
        init.constant_(self.gate.weight, 0)
        self.norm = DEIMRMSNorm(d_model) if use_rmsnorm else nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gate1, gate2 = torch.sigmoid(self.gate(torch.cat([x1, x2], dim=-1))).chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class DeimTransformerDecoderLayer(nn.Module):
    """DEIMv2 decoder layer (RMSNorm + SwiGLU)."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        n_levels: int = 4,
        n_points: int = 4,
        cross_attn_method: str = "default",
        layer_scale=None,
        use_gateway: bool = False,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        del act
        if layer_scale is not None:
            d_ffn = round(layer_scale * d_ffn)
            d_model = round(layer_scale * d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        norm_layer = DEIMRMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm1 = norm_layer(d_model)

        self.cross_attn = MSDeformableAttention(d_model, n_heads, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)
        self.use_gateway = use_gateway
        if use_gateway:
            self.gateway = DeimGate(d_model, use_rmsnorm=use_rmsnorm)
        else:
            self.norm2 = norm_layer(d_model)

        self.swish_ffn = DEIMSwiGLUFFN(d_model, d_ffn // 2, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = norm_layer(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, target, reference_points, value, spatial_shapes, attn_mask=None, query_pos_embed=None):
        q = k = self.with_pos_embed(target, query_pos_embed)
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = self.norm1(target + self.dropout1(target2))

        target2 = self.cross_attn(self.with_pos_embed(target, query_pos_embed), reference_points, value, spatial_shapes)
        if self.use_gateway:
            target = self.gateway(target, self.dropout2(target2))
        else:
            target = self.norm2(target + self.dropout2(target2))

        target2 = self.swish_ffn(target)
        return self.norm3((target + self.dropout4(target2)).clamp(min=-65504, max=65504))


class DeimTransformerDecoder(DFineTransformerDecoder):
    """DEIMv2 decoder wrapper using DFine forward path with fixed query-position embeddings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_query_pos = True
