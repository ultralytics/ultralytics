# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""SAM 3.1 Object Multiplex video tracker network.

Inference-only port of Meta's multiplex tracker stack (VideoTrackingMultiplex →
VideoTrackingDynamicMultiplex → VideoTrackingMultiplexDemo → Sam3VideoTrackingMultiplexDemo)
flattened into a single ``SAM3MultiplexModel`` class. The memory encoder, memory attention, and
mask decoder run once per bucket of ``multiplex_count`` objects instead of once per object.

Scope: mask-seeded objects and propagation (the paths used by text-prompted semantic video
tracking). Interactive click refinement (add_new_points and per-object singleton extraction) is
not ported. Training-only code (correction-point sampling, loss bookkeeping, torch.compile,
multi-GPU) is removed.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import MLP
from ultralytics.utils import TQDM

from ..modules.blocks import PositionEmbeddingRandom
from ..modules.encoders import PromptEncoder
from ..modules.transformer import TwoWayTransformer
from ..modules.utils import get_1d_sine_pe, select_closest_cond_frames
from .multiplex import MultiplexController, MultiplexMaskDecoder, MultiplexState

NO_OBJ_SCORE = -1024.0


def _tensors(x):
    """Return the underlying tensor of a NestedTensor-like object, or the tensor itself."""
    return x.tensors if hasattr(x, "tensors") else x


def _append(d1: dict, d2: dict, k1: str, k2: str, dim: int = 0, strict: bool = True):
    """Concatenate d2[k2] onto d1[k1] along ``dim`` (skip when the key is absent and not strict)."""
    if not strict and k1 not in d1:
        return
    assert k1 in d1, f"{k1} not found"
    d1[k1] = torch.cat([d1[k1], d2[k2]], dim=dim)


def _merge(d1: dict, d2: dict, k1: str, k2: str, d2_idx: list[int], strict: bool = True):
    """Write d2[k2] into d1[k1] at rows ``d2_idx`` (skip when the key is absent and not strict)."""
    if not strict and k1 not in d1:
        return
    assert k1 in d1, f"{k1} not found"
    d1[k1][d2_idx] = d2[k2].to(dtype=d1[k1].dtype)


class TrackerTransformerWrapper(nn.Module):
    """Encoder-only transformer holder matching Meta's TransformerWrapper checkpoint layout."""

    def __init__(self, encoder: nn.Module, d_model: int):
        """Store the decoupled memory-attention encoder under the ``encoder`` prefix."""
        super().__init__()
        self.encoder = encoder
        self.decoder = None
        self.d_model = d_model


class SAM3MultiplexModel(nn.Module):
    """SAM 3.1 Object Multiplex tracker with bucketed shared-memory multi-object tracking.

    Objects are grouped into buckets of ``multiplex_count`` slots (managed by a MultiplexState);
    memory encoding, memory attention, and mask decoding run with batch = num_buckets. Spatial
    memory and object pointers are stored per bucket in the mux space, and demuxed to the
    per-object data space on read.
    """

    def __init__(
        self,
        backbone: nn.Module | None,
        transformer: nn.Module,
        maskmem_backbone: nn.Module,
        multiplex_controller: MultiplexController,
        num_maskmem: int = 7,
        image_size: int = 1008,
        backbone_stride: int = 14,
        apply_sigmoid_to_mask_logits_for_mem_enc: bool = False,
        sigmoid_scale_for_mem_enc: float = 1.0,
        sigmoid_bias_for_mem_enc: float = 0.0,
        use_mask_input_as_output_without_sam: bool = False,
        max_cond_frames_in_attn: int = -1,
        add_all_frames_to_correct_as_cond: bool = False,
        directly_add_no_mem_embed: bool = False,
        use_high_res_features_in_sam: bool = False,
        multimask_output_in_sam: bool = False,
        multimask_min_pt_num: int = 1,
        multimask_max_pt_num: int = 1,
        multimask_output_for_tracking: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        forward_backbone_per_frame_for_eval: bool = False,
        memory_temporal_stride_for_eval: int = 1,
        offload_output_to_cpu_for_eval: bool = False,
        trim_past_non_cond_mem_for_eval: bool = False,
        non_overlap_masks_for_mem_enc: bool = False,
        use_obj_ptrs_in_encoder: bool = False,
        max_obj_ptrs_in_encoder: int = 16,
        add_tpos_enc_to_obj_ptrs: bool = True,
        proj_tpos_enc_in_obj_ptrs: bool = False,
        use_signed_tpos_enc_to_obj_ptrs: bool = False,
        only_obj_ptrs_in_the_past_for_eval: bool = False,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        fixed_no_obj_ptr: bool = False,
        use_no_obj_ptr: bool = True,
        use_mlp_for_obj_ptr_proj: bool = False,
        use_linear_no_obj_ptr: bool = False,
        no_obj_embed_spatial: bool = False,
        sincos_tpos_enc: bool = True,
        sam_mask_decoder_extra_args: dict | None = None,
        save_image_features: bool = False,
        num_multimask_outputs: int = 3,
        add_output_suppression_embeddings: bool = False,
        add_object_conditional_embeddings: bool = False,
        condition_as_mask_input: bool = False,
        condition_as_mask_input_fg: float = 1.0,
        condition_as_mask_input_bg: float = 0.0,
        use_maskmem_tpos_v2: bool = False,
        is_dynamic_model: bool = False,
        object_score_logit_threshold: float = 0.0,
        # demo/session options
        clear_non_cond_mem_around_input: bool = False,
        clear_non_cond_mem_for_multi_obj: bool = False,
        fill_hole_area: int = 0,
        always_start_from_first_ann_frame: bool = False,
        max_point_num_in_prompt_enc: int = 16,
        non_overlap_masks_for_output: bool = True,
        # accepted for config parity with Meta's builder (training/eval-harness only)
        **kwargs,
    ):
        """Initialize the multiplex tracker with the configuration used by sam3.1_multiplex.pt."""
        super().__init__()
        from torch.nn.init import trunc_normal_

        assert not kwargs.get("use_memory_selection", False), "memory selection is not ported"
        assert not kwargs.get("stability_score_attentuation", False), "stability attenuation is not ported"
        assert not kwargs.get("decode_mask_with_shared_tokens", False), "shared mask tokens are not ported"
        assert not kwargs.get("decode_mask_attribute_with_shared_tokens", False), "shared tokens are not ported"
        assert not kwargs.get("share_necks", False), "shared necks are not ported"
        assert fill_hole_area == 0, "hole filling is not supported in this port"
        assert save_image_features, "the decoupled memory attention requires image features in memory"
        assert sincos_tpos_enc, "only sincos temporal encodings are supported"

        # Part 1: image backbone (may be None when features are injected via cached_features)
        self.backbone = backbone
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.num_feature_levels = 3 if use_high_res_features_in_sam else 1
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder
        if use_obj_ptrs_in_encoder:
            self.interactive_mask_downsample = nn.Conv2d(1, 1, kernel_size=4, stride=4)
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        if proj_tpos_enc_in_obj_ptrs:
            assert add_tpos_enc_to_obj_ptrs
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval
        self.multiplex_controller = multiplex_controller
        self.save_image_features = save_image_features
        self.multiplex_count = multiplex_controller.multiplex_count

        # Part 2: encoder-only transformer fusing current features with the memory bank
        assert transformer.decoder is None, "transformer should be encoder-only"
        self.transformer = transformer
        self.hidden_dim: int = transformer.d_model

        # Part 3: memory encoder
        self.maskmem_backbone = maskmem_backbone
        self.mem_dim = self.hidden_dim
        if hasattr(maskmem_backbone, "out_proj") and hasattr(maskmem_backbone.out_proj, "weight"):
            assert maskmem_backbone.out_proj.weight.shape[0] == self.hidden_dim, "no memory compression expected"
        self.num_maskmem = num_maskmem
        self.sincos_tpos_enc = sincos_tpos_enc
        self.use_maskmem_tpos_v2 = use_maskmem_tpos_v2
        self.maskmem_tpos_enc = nn.Parameter(torch.zeros(num_maskmem, 1, 1, self.mem_dim))
        trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        self.interactivity_no_mem_embed = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        trunc_normal_(self.interactivity_no_mem_embed, std=0.02)
        self.directly_add_no_mem_embed = directly_add_no_mem_embed

        self.apply_sigmoid_to_mask_logits_for_mem_enc = apply_sigmoid_to_mask_logits_for_mem_enc
        if apply_sigmoid_to_mask_logits_for_mem_enc:
            self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
            self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
            self.binarize_mask_from_pts_for_mem_enc = False  # not trained with binarization
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.multimask_min_pt_num = multimask_min_pt_num
        self.multimask_max_pt_num = multimask_max_pt_num
        self.multimask_output_for_tracking = multimask_output_for_tracking
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid
        self.object_score_logit_threshold = object_score_logit_threshold

        # Part 4: SAM heads
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.low_res_mask_size = self.image_size // self.backbone_stride * 4
        self.input_mask_size = self.low_res_mask_size * 4
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval
        self.offload_output_to_cpu_for_eval = offload_output_to_cpu_for_eval
        self.trim_past_non_cond_mem_for_eval = trim_past_non_cond_mem_for_eval
        # Meta resets dynamic_multimask_via_stability to False for the multiplex decoder while the
        # interactive decoder keeps it.
        self.interactive_sam_mask_decoder_extra_args = deepcopy(sam_mask_decoder_extra_args)
        if sam_mask_decoder_extra_args is not None:
            sam_mask_decoder_extra_args = dict(sam_mask_decoder_extra_args)
            sam_mask_decoder_extra_args["dynamic_multimask_via_stability"] = False
        self.sam_mask_decoder_extra_args = sam_mask_decoder_extra_args
        self.pred_obj_scores = pred_obj_scores
        self.pred_obj_scores_mlp = pred_obj_scores_mlp
        self.fixed_no_obj_ptr = fixed_no_obj_ptr
        self.use_no_obj_ptr = use_no_obj_ptr
        self.use_linear_no_obj_ptr = use_linear_no_obj_ptr
        if self.fixed_no_obj_ptr:
            assert self.pred_obj_scores and self.use_obj_ptrs_in_encoder
        if self.pred_obj_scores and self.use_obj_ptrs_in_encoder and self.use_no_obj_ptr:
            if self.use_linear_no_obj_ptr:
                self.no_obj_ptr_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
            else:
                self.no_obj_ptr = nn.Parameter(torch.zeros(self.multiplex_count, self.hidden_dim))
                trunc_normal_(self.no_obj_ptr, std=0.02)
        self.use_mlp_for_obj_ptr_proj = use_mlp_for_obj_ptr_proj
        self.no_obj_embed_spatial = None
        if no_obj_embed_spatial:
            self.no_obj_embed_spatial = nn.Parameter(torch.zeros(self.multiplex_count, self.hidden_dim))
            trunc_normal_(self.no_obj_embed_spatial, std=0.02)
        self.num_multimask_outputs = num_multimask_outputs

        self.add_output_suppression_embeddings = add_output_suppression_embeddings
        if add_output_suppression_embeddings:
            self.output_valid_embed = nn.Parameter(torch.zeros(self.multiplex_count, self.hidden_dim))
            self.output_invalid_embed = nn.Parameter(torch.zeros(self.multiplex_count, self.hidden_dim))
            trunc_normal_(self.output_valid_embed, std=0.02)
            trunc_normal_(self.output_invalid_embed, std=0.02)
        self.add_object_conditional_embeddings = add_object_conditional_embeddings
        if add_object_conditional_embeddings:
            self.obj_cond_embed = nn.Parameter(torch.zeros(self.multiplex_count, self.hidden_dim))
            trunc_normal_(self.obj_cond_embed, std=0.02)
        self.condition_as_mask_input = condition_as_mask_input
        self.condition_as_mask_input_fg = condition_as_mask_input_fg
        self.condition_as_mask_input_bg = condition_as_mask_input_bg
        self.is_dynamic_model = is_dynamic_model

        self._build_sam_heads(TwoWayTransformer)
        self.max_cond_frames_in_attn = max_cond_frames_in_attn
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond

        # demo/session options
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj
        self.fill_hole_area = fill_hole_area
        self.always_start_from_first_ann_frame = always_start_from_first_ann_frame
        self.max_point_num_in_prompt_enc = max_point_num_in_prompt_enc
        self.non_overlap_masks_for_output = non_overlap_masks_for_output

    def _build_sam_heads(self, two_way_transformer_cls):
        """Build the SAM-style prompt encoder, interactive decoder, and the multiplex decoder."""
        from ..modules.decoders import SAM2MaskDecoder

        self.sam_prompt_embed_dim = self.hidden_dim
        self.sam_image_embedding_size = self.image_size // self.backbone_stride
        self.image_pe_layer = PositionEmbeddingRandom(self.hidden_dim // 2)

        self.interactive_sam_prompt_encoder = PromptEncoder(
            embed_dim=self.sam_prompt_embed_dim,
            image_embedding_size=(self.sam_image_embedding_size, self.sam_image_embedding_size),
            input_image_size=(self.image_size, self.image_size),
            mask_in_chans=16,
        )
        self.interactive_sam_mask_decoder = SAM2MaskDecoder(
            num_multimask_outputs=3,
            transformer=two_way_transformer_cls(
                depth=2, embedding_dim=self.sam_prompt_embed_dim, mlp_dim=2048, num_heads=8
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.interactive_sam_mask_decoder_extra_args or {}),
        )
        self.sam_mask_decoder = MultiplexMaskDecoder(
            multiplex_count=self.multiplex_count,
            num_multimask_outputs=self.num_multimask_outputs,
            transformer=two_way_transformer_cls(depth=2, embedding_dim=self.hidden_dim, mlp_dim=2048, num_heads=8),
            transformer_dim=self.hidden_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )

        if self.use_obj_ptrs_in_encoder:
            if self.use_mlp_for_obj_ptr_proj:
                self.obj_ptr_proj = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
                self.interactive_obj_ptr_proj = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
            else:
                self.obj_ptr_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
                self.interactive_obj_ptr_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        else:
            self.obj_ptr_proj = nn.Identity()
            self.interactive_obj_ptr_proj = nn.Identity()
        if self.proj_tpos_enc_in_obj_ptrs:
            self.obj_ptr_tpos_proj = nn.Linear(self.hidden_dim, self.mem_dim)
        else:
            self.obj_ptr_tpos_proj = nn.Identity()

    def _get_tpos_enc(self, rel_pos_list, device, max_abs_pos=None, dummy=False):
        """Sine temporal position encoding for object pointers, projected to mem_dim."""
        if dummy:
            return torch.zeros(len(rel_pos_list), self.mem_dim, device=device)
        t_diff_max = max_abs_pos - 1 if max_abs_pos is not None else 1
        pos_enc = torch.tensor(rel_pos_list, device=device, dtype=torch.float32) / t_diff_max
        tpos_dim = self.hidden_dim if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
        pos_enc = get_1d_sine_pe(pos_enc, dim=tpos_dim)
        return self.obj_ptr_tpos_proj(pos_enc)

    def _get_interactive_pix_mem(self, features, feat_sizes):
        """Interactive-head pixel features with the no-memory embedding added (BCHW)."""
        assert self.directly_add_no_mem_embed
        pix_feat_with_mem = features[-1] + self.interactivity_no_mem_embed
        B = features[-1].size(1)
        H, W = feat_sizes[-1]
        return pix_feat_with_mem.permute(1, 2, 0).view(B, self.hidden_dim, H, W)

    def _use_multimask(self, is_init_cond_frame, point_inputs):
        """Whether the SAM head should produce multiple masks."""
        num_pts = 0 if point_inputs is None else point_inputs["point_labels"].size(1)
        return (
            self.multimask_output_in_sam
            and (is_init_cond_frame or self.multimask_output_for_tracking)
            and (self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num)
            and self.num_multimask_outputs > 0
        )

    @staticmethod
    def _apply_non_overlapping_constraints(pred_masks):
        """Keep only the highest-scoring object per location."""
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        batch_obj_inds = torch.arange(batch_size, device=pred_masks.device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        return torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))

    def get_propagation_dense_pe(self) -> torch.Tensor:
        """Dense positional encoding for the multiplex mask decoder."""
        return self.image_pe_layer((self.sam_image_embedding_size, self.sam_image_embedding_size)).unsqueeze(0)

    # ------------------------------------------------------------------
    # Backbone features
    # ------------------------------------------------------------------

    def forward_image(self, img_batch, need_sam3_out=False, need_interactive_out=False, need_propagation_out=False):
        """Run the vision backbone and pre-project high-res levels for the SAM decoders."""
        assert self.backbone is not None, "no backbone attached; inject features via cached_features instead"
        backbone_out = self.backbone.forward_image(
            img_batch,
            need_sam3_out=need_sam3_out,
            need_interactive_out=need_interactive_out,
            need_propagation_out=need_propagation_out,
        )
        if self.use_high_res_features_in_sam:
            if need_interactive_out:
                fpn = backbone_out["interactive"]["backbone_fpn"]
                fpn[0] = self.interactive_sam_mask_decoder.conv_s0(_tensors(fpn[0]))
                fpn[1] = self.interactive_sam_mask_decoder.conv_s1(_tensors(fpn[1]))
            if need_propagation_out:
                fpn = backbone_out["sam2_backbone_out"]["backbone_fpn"]
                fpn[0] = self.sam_mask_decoder.conv_s0(_tensors(fpn[0]))
                fpn[1] = self.sam_mask_decoder.conv_s1(_tensors(fpn[1]))
        return backbone_out

    def _prepare_backbone_features(self, backbone_out):
        """Flatten the interactive and propagation pyramids to (HW)BC lists."""
        backbone_features = {}
        for neck_k in ("interactive", "sam2_backbone_out"):
            if neck_k not in backbone_out:
                continue
            neck_out = backbone_out[neck_k]
            assert len(neck_out["backbone_fpn"]) == len(neck_out["vision_pos_enc"])
            assert len(neck_out["backbone_fpn"]) >= self.num_feature_levels
            feature_maps = neck_out["backbone_fpn"][-self.num_feature_levels :]
            vision_pos_embeds = neck_out["vision_pos_enc"][-self.num_feature_levels :]
            feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
            vision_feats = [_tensors(x).flatten(2).permute(2, 0, 1) for x in feature_maps]
            vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]
            backbone_features[neck_k] = {
                "vision_feats": vision_feats,
                "vision_pos_embeds": vision_pos_embeds,
                "vision_masks": [None] * len(vision_feats),
                "feat_sizes": feat_sizes,
            }
        return backbone_features

    # ------------------------------------------------------------------
    # SAM heads
    # ------------------------------------------------------------------

    def _forward_sam_heads(
        self,
        backbone_features: torch.Tensor,
        point_inputs: dict | None = None,
        mask_inputs: torch.Tensor | None = None,
        interactive_high_res_features: list | None = None,
        propagation_high_res_features: list | None = None,
        multimask_output: bool = False,
        multiplex_state: MultiplexState = None,
        objects_to_interact: list[int] | None = None,
    ) -> dict:
        """Run the interactive head (per object) or the multiplex propagation head (per bucket).

        Returns a dict with low/high-res multimasks, best masks, ious, object_score_logits, and
        (when enabled) obj_ptr in the per-object data space.
        """
        device = backbone_features.device
        assert backbone_features.size(1) == self.hidden_dim
        assert backbone_features.size(2) == self.sam_image_embedding_size
        assert backbone_features.size(3) == self.sam_image_embedding_size

        is_interactive = point_inputs is not None or mask_inputs is not None
        if is_interactive:
            # Image-level, per-object interactive path
            assert interactive_high_res_features is not None
            assert objects_to_interact is not None
            if point_inputs is not None:
                sam_point_coords = point_inputs["point_coords"]
                sam_point_labels = point_inputs["point_labels"]
            else:
                sam_point_coords = torch.zeros(mask_inputs.shape[0], 1, 2, device=device)
                sam_point_labels = -torch.ones(mask_inputs.shape[0], 1, dtype=torch.int32, device=device)

            if mask_inputs is not None:
                if mask_inputs.shape[-2:] != self.interactive_sam_prompt_encoder.mask_input_size:
                    sam_mask_prompt = F.interpolate(
                        mask_inputs.float(),
                        size=self.interactive_sam_prompt_encoder.mask_input_size,
                        align_corners=False,
                        mode="bilinear",
                        antialias=True,
                    )
                else:
                    sam_mask_prompt = mask_inputs
            else:
                sam_mask_prompt = None

            sparse_embeddings, dense_embeddings = self.interactive_sam_prompt_encoder(
                points=(sam_point_coords, sam_point_labels), boxes=None, masks=sam_mask_prompt
            )
            low_res_multimasks, ious, sam_output_tokens, object_score_logits = self.interactive_sam_mask_decoder(
                image_embeddings=backbone_features,
                image_pe=self.interactive_sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image=True,
                high_res_features=interactive_high_res_features,
            )
        else:
            # Multiplexed propagation path
            assert propagation_high_res_features is not None
            assert multiplex_state is not None
            if self.add_output_suppression_embeddings:
                valid_object_mask = multiplex_state.get_valid_object_mask().unsqueeze(-1).float()
                output_merged_embed = valid_object_mask * self.output_valid_embed.unsqueeze(0) + (
                    1 - valid_object_mask
                ) * self.output_invalid_embed.unsqueeze(0)
            else:
                output_merged_embed = None

            out = self.sam_mask_decoder(
                image_embeddings=backbone_features,
                image_pe=self.get_propagation_dense_pe(),
                high_res_features=propagation_high_res_features,
                multimask_output=multimask_output,
                extra_per_object_embeddings=output_merged_embed,
            )
            low_res_multimasks = multiplex_state.demux(out["masks"])
            ious = multiplex_state.demux(out["iou_pred"])
            object_score_logits = multiplex_state.demux(out["object_score_logits"])
            sam_output_tokens = multiplex_state.demux(out["sam_tokens_out"])

        if self.pred_obj_scores:
            is_obj_appearing = object_score_logits > self.object_score_logit_threshold
            low_res_multimasks = torch.where(is_obj_appearing[:, None, None], low_res_multimasks, NO_OBJ_SCORE)

        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(ious.shape[0], device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        outputs = {
            "low_res_multimasks": low_res_multimasks,
            "high_res_multimasks": high_res_multimasks,
            "ious": ious,
            "low_res_masks": low_res_masks,
            "high_res_masks": high_res_masks,
            "object_score_logits": object_score_logits,
        }
        if self.use_obj_ptrs_in_encoder:
            if is_interactive:
                obj_ptr = self.interactive_obj_ptr_proj(sam_output_token)
            else:
                obj_ptr = self.obj_ptr_proj(sam_output_token)
            if self.pred_obj_scores and self.use_no_obj_ptr:
                lambda_is_obj_appearing = is_obj_appearing.float()
                if self.use_linear_no_obj_ptr:
                    obj_ptr = lambda_is_obj_appearing * obj_ptr + (
                        1 - lambda_is_obj_appearing
                    ) * self.no_obj_ptr_linear(obj_ptr)
                else:
                    if self.fixed_no_obj_ptr:
                        obj_ptr = lambda_is_obj_appearing * obj_ptr
                    selected_no_obj_ptr = self.no_obj_ptr.unsqueeze(0).repeat(multiplex_state.num_buckets, 1, 1)
                    selected_no_obj_ptr = multiplex_state.demux(selected_no_obj_ptr)
                    if is_interactive:
                        selected_no_obj_ptr = selected_no_obj_ptr[objects_to_interact]
                    obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * selected_no_obj_ptr
            outputs["obj_ptr"] = obj_ptr  # [num_objects, C], data space
        return outputs

    def _use_mask_as_output(
        self,
        backbone_features: torch.Tensor,
        high_res_features: list,
        mask_inputs: torch.Tensor,
        multiplex_state: MultiplexState,
        objects_in_mask: list[int] | None = None,
    ) -> dict:
        """Turn binary mask inputs directly into output logits without running SAM."""
        if objects_in_mask is None:
            objects_in_mask = list(range(multiplex_state.total_valid_entries))
        out_scale, out_bias = 20.0, -10.0  # sigmoid(-10.0)=4.5398e-05
        mask_inputs_float = mask_inputs.to(backbone_features.dtype)
        assert mask_inputs.shape[0] == len(objects_in_mask)
        high_res_masks = mask_inputs_float * out_scale + out_bias
        low_res_masks = F.interpolate(
            high_res_masks,
            size=(high_res_masks.size(-2) // 4, high_res_masks.size(-1) // 4),
            align_corners=False,
            mode="bilinear",
            antialias=True,
        )
        ious = mask_inputs.new_ones(mask_inputs.size(0), 1, dtype=backbone_features.dtype)

        if self.use_obj_ptrs_in_encoder:
            sam_outputs = self._forward_sam_heads(
                backbone_features=backbone_features,
                mask_inputs=self.interactive_mask_downsample(mask_inputs_float),
                interactive_high_res_features=high_res_features,
                objects_to_interact=objects_in_mask,
                multiplex_state=multiplex_state,
            )
            obj_ptr = sam_outputs["obj_ptr"]
            is_obj_appearing = torch.any(mask_inputs.flatten(1).float() > 0.0, dim=1)[..., None]
            lambda_is_obj_appearing = is_obj_appearing.float()
            object_score_logits = out_scale * lambda_is_obj_appearing + out_bias
            if self.pred_obj_scores and self.use_no_obj_ptr:
                if self.use_linear_no_obj_ptr:
                    obj_ptr = lambda_is_obj_appearing * obj_ptr + (
                        1 - lambda_is_obj_appearing
                    ) * self.no_obj_ptr_linear(obj_ptr)
                else:
                    if self.fixed_no_obj_ptr:
                        obj_ptr = lambda_is_obj_appearing * obj_ptr
                    selected_no_obj_ptr = self.no_obj_ptr.unsqueeze(0).repeat(multiplex_state.num_buckets, 1, 1)
                    selected_no_obj_ptr = multiplex_state.demux(selected_no_obj_ptr)[objects_in_mask]
                    obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * selected_no_obj_ptr

        outputs = {
            "low_res_multimasks": low_res_masks,
            "high_res_multimasks": high_res_masks,
            "ious": ious,
            "low_res_masks": low_res_masks,
            "high_res_masks": high_res_masks,
            "object_score_logits": object_score_logits,
        }
        if self.use_obj_ptrs_in_encoder:
            outputs["obj_ptr"] = obj_ptr
        return outputs

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
        multiplex_state: MultiplexState = None,
    ):
        """Fuse the current frame's features (per bucket) with the memory bank."""
        B = multiplex_state.num_buckets
        vision_feat = current_vision_feats[-1].expand(-1, B, -1)
        vision_pos_embed = current_vision_pos_embeds[-1].expand(-1, B, -1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        device = current_vision_feats[-1].device
        if self.num_maskmem == 0:
            return vision_feat.permute(1, 2, 0).view(B, C, H, W)

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        if is_init_cond_frame:
            raise NotImplementedError("Any init cond frame should have gone to _use_mask_as_output instead")

        to_cat_prompt, to_cat_prompt_pos_embed = [], []
        to_cat_image_feat, to_cat_image_pos_embed = [], []
        assert len(output_dict["cond_frame_outputs"]) > 0
        cond_outputs = output_dict["cond_frame_outputs"]
        selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
            frame_idx, cond_outputs, self.max_cond_frames_in_attn
        )
        t_pos_and_prevs = [((frame_idx - t) * tpos_sign_mul, out, True) for t, out in selected_cond_outputs.items()]
        r = self.memory_temporal_stride_for_eval
        for t_pos in range(1, self.num_maskmem):
            t_rel = self.num_maskmem - t_pos
            if t_rel == 1:
                prev_frame_idx = frame_idx + t_rel if track_in_reverse else frame_idx - t_rel
            elif not track_in_reverse:
                prev_frame_idx = ((frame_idx - 2) // r) * r - (t_rel - 2) * r
            else:
                prev_frame_idx = -(-(frame_idx + 2) // r) * r + (t_rel - 2) * r
            out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
            if out is None:
                out = unselected_cond_outputs.get(prev_frame_idx, None)
            t_pos_and_prevs.append((t_pos, out, False))

        for t_pos, prev, _is_selected_cond_frame in t_pos_and_prevs:
            if prev is None:
                continue
            feats = prev.get("maskmem_features")
            if feats is None:
                continue
            feats = feats.to(device, non_blocking=device.type == "cuda")
            if feats.dim() == 5:
                feats = multiplex_state.demux(feats).contiguous()
                prev["maskmem_features"] = feats.cpu() if not feats.is_cuda else feats
            if feats.shape[0] == 0:
                continue
            to_cat_prompt.append(feats.float().flatten(2).permute(2, 0, 1))
            maskmem_pos_list = prev.get("maskmem_pos_enc")
            if not maskmem_pos_list:
                continue
            maskmem_enc = maskmem_pos_list[-1]
            if maskmem_enc is None:
                continue
            maskmem_enc = maskmem_enc.to(device, non_blocking=device.type == "cuda")
            if maskmem_enc.dim() == 5:
                maskmem_enc = multiplex_state.demux(maskmem_enc).contiguous()
                prev["maskmem_pos_enc"][-1] = maskmem_enc.cpu() if not maskmem_enc.is_cuda else maskmem_enc
            maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)

            # use_maskmem_tpos_v2: the last slot is an "out-of-range" embedding
            if self.use_maskmem_tpos_v2:
                if t_pos <= 0 or t_pos >= self.num_maskmem:
                    tpos_enc = self.maskmem_tpos_enc[self.num_maskmem - 1]
                else:
                    tpos_enc = self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
            else:
                t = t_pos if not _is_selected_cond_frame else 0
                tpos_enc = self.maskmem_tpos_enc[self.num_maskmem - t - 1]
            maskmem_enc = maskmem_enc + tpos_enc

            image_feat = prev["image_features"].to(device)
            image_pos_embed = prev["image_pos_enc"].to(device) + tpos_enc
            to_cat_image_feat.append(image_feat.float())
            to_cat_image_pos_embed.append(image_pos_embed.float())
            to_cat_prompt_pos_embed.append(maskmem_enc)

        if self.use_obj_ptrs_in_encoder:
            max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
            if self.only_obj_ptrs_in_the_past_for_eval:
                ptr_cond_outputs = {
                    t: out
                    for t, out in selected_cond_outputs.items()
                    if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                }
            else:
                ptr_cond_outputs = selected_cond_outputs
            pos_and_outs_for_ptr = [
                (
                    (frame_idx - t) * tpos_sign_mul if self.use_signed_tpos_enc_to_obj_ptrs else abs(frame_idx - t),
                    out,
                )
                for t, out in ptr_cond_outputs.items()
            ]
            for t_diff in range(1, max_obj_ptrs_in_encoder):
                t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                if t < 0 or (num_frames is not None and t >= num_frames):
                    break
                out = output_dict["non_cond_frame_outputs"].get(t, unselected_cond_outputs.get(t, None))
                if out is not None:
                    pos_and_outs_for_ptr.append((t_diff, out))
            filtered_data = [(pos, out) for pos, out in pos_and_outs_for_ptr if "obj_ptr" in out]
            if filtered_data:
                pos_list, out_list = zip(*filtered_data)
                # each out["obj_ptr"] is (num_buckets, multiplex_count, C) in the mux space
                obj_ptrs = torch.cat([out["obj_ptr"] for out in out_list], dim=1).transpose(0, 1).float()
                if self.add_tpos_enc_to_obj_ptrs:
                    obj_pos = self._get_tpos_enc(pos_list, max_abs_pos=max_obj_ptrs_in_encoder, device=device)
                else:
                    obj_pos = self._get_tpos_enc(pos_list, device=device, dummy=True)
                obj_pos = obj_pos.unsqueeze(1).expand(-1, B, -1)
                assert self.mem_dim == C
                obj_pos = obj_pos.repeat_interleave(multiplex_state.multiplex_count, dim=0)
                to_cat_prompt.append(obj_ptrs)
                to_cat_prompt_pos_embed.append(obj_pos)
                num_obj_ptr_tokens = obj_ptrs.shape[0]

        if len(to_cat_prompt) == 0 or len(to_cat_image_feat) == 0:
            # No available memory (e.g. cleared); fall back to the current frame features.
            return vision_feat.permute(1, 2, 0).view(B, C, H, W)

        prompt = torch.cat(to_cat_prompt, dim=0)
        prompt_pos_embed = torch.cat(to_cat_prompt_pos_embed, dim=0)
        image_feat = torch.cat(to_cat_image_feat, dim=0)
        image_pos_embed = torch.cat(to_cat_image_pos_embed, dim=0)

        encoder_out = self.transformer.encoder(
            image=current_vision_feats[-1],
            src=vision_feat,
            memory_image=image_feat,
            memory=prompt,
            image_pos=current_vision_pos_embeds[-1],
            src_pos=vision_pos_embed,
            memory_image_pos=image_pos_embed,
            memory_pos=prompt_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        return encoder_out["memory"].permute(1, 2, 0).view(B, C, H, W)

    def _encode_new_memory(
        self,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
        conditioning_objects=None,
        multiplex_state: MultiplexState = None,
    ):
        """Encode the current frame and its (muxed) masks into a bucket-level memory feature."""
        B = current_vision_feats[-1].size(1)
        C = self.hidden_dim
        H, W = feat_sizes[-1]
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.non_overlap_masks_for_mem_enc:
            pred_masks_high_res = self._apply_non_overlapping_constraints(pred_masks_high_res)
        if self.apply_sigmoid_to_mask_logits_for_mem_enc:
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
            if self.sigmoid_scale_for_mem_enc != 1.0:
                mask_for_mem = mask_for_mem * self.sigmoid_scale_for_mem_enc
            if self.sigmoid_bias_for_mem_enc != 0.0:
                mask_for_mem = mask_for_mem + self.sigmoid_bias_for_mem_enc
        else:
            mask_for_mem = pred_masks_high_res

        if self.add_object_conditional_embeddings or self.condition_as_mask_input:
            if conditioning_objects is None:
                conditioning_objects = []
            else:
                conditioning_objects = sorted(conditioning_objects)

        mux_mask_for_mem = multiplex_state.mux(mask_for_mem).squeeze(2)

        if self.condition_as_mask_input:
            num_objects = mask_for_mem.shape[0]
            cond_values = torch.full(
                (num_objects,), self.condition_as_mask_input_bg, device=mask_for_mem.device, dtype=mask_for_mem.dtype
            )
            if len(conditioning_objects) > 0:
                cond_values[list(conditioning_objects)] = self.condition_as_mask_input_fg
            embedded_conditions = cond_values.view(-1, 1, 1, 1).expand_as(mask_for_mem)
            embedded_conditions = multiplex_state.mux(embedded_conditions).squeeze(2)
            mux_mask_for_mem = torch.cat([mux_mask_for_mem, embedded_conditions], dim=1)

        maskmem_out = self.maskmem_backbone(pix_feat, mux_mask_for_mem, skip_mask_sigmoid=True)
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = list(maskmem_out["vision_pos_enc"])

        if self.no_obj_embed_spatial is not None:
            no_obj_embed_spatial = self.no_obj_embed_spatial.unsqueeze(0).repeat(multiplex_state.num_buckets, 1, 1)
            obj_expected = multiplex_state.total_valid_entries
            if object_score_logits.shape[0] != obj_expected:
                if object_score_logits.shape[0] < obj_expected:
                    pad = object_score_logits.new_zeros(
                        (obj_expected - object_score_logits.shape[0],) + tuple(object_score_logits.shape[1:])
                    )
                    object_score_logits = torch.cat([object_score_logits, pad], dim=0)
                else:
                    object_score_logits = object_score_logits[:obj_expected]
            object_score_logits = multiplex_state.mux(object_score_logits)
            is_obj_appearing = (object_score_logits > self.object_score_logit_threshold).float()
            no_obj_embed = ((1 - is_obj_appearing) * no_obj_embed_spatial).sum(dim=1)
            maskmem_features = maskmem_features + no_obj_embed[..., None, None].expand_as(maskmem_features)

        if maskmem_features.dim() == 5:
            maskmem_features = multiplex_state.demux(maskmem_features).contiguous()
        maskmem_pos_enc = [
            multiplex_state.demux(p).contiguous() if p is not None and p.dim() == 5 else p for p in maskmem_pos_enc
        ]
        return maskmem_features, maskmem_pos_enc

    # ------------------------------------------------------------------
    # Track step
    # ------------------------------------------------------------------

    def _track_step_aux(
        self,
        frame_idx,
        is_init_cond_frame,
        backbone_features_interactive,
        backbone_features_propagation,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,
        run_mem_encoder=True,
        prev_sam_mask_logits=None,
        multiplex_state: MultiplexState = None,
        objects_to_interact: list[int] | None = None,
        need_aux_output: bool = False,
    ):
        """Single-frame tracking: mask-as-output, propagation, interaction, or propagation+interaction."""
        current_out = {"conditioning_objects": set(), "point_inputs": point_inputs, "mask_inputs": mask_inputs}

        if mask_inputs is not None:
            mode = "mask_as_output"
        elif point_inputs is None:
            mode = "propagation_only"
        elif prev_sam_mask_logits is not None or is_init_cond_frame:
            assert prev_sam_mask_logits is None or objects_to_interact is not None
            mode = "interaction_only"
        elif objects_to_interact is not None:
            mode = "propagation_and_interaction"
        else:
            raise ValueError("Unable to determine tracking mode")

        interactive_high_res_features = interactive_vision_feats = interactive_feat_sizes = None
        if backbone_features_interactive is not None:
            interactive_vision_feats = backbone_features_interactive["vision_feats"]
            interactive_feat_sizes = backbone_features_interactive["feat_sizes"]
            if len(interactive_vision_feats) > 1:
                interactive_high_res_features = [
                    x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                    for x, s in zip(interactive_vision_feats[:-1], interactive_feat_sizes[:-1])
                ]
        else:
            assert mode not in ("interaction_only", "propagation_and_interaction")

        propagation_high_res_features = propagation_vision_feats = None
        propagation_vision_pos_embeds = propagation_feat_sizes = None
        if backbone_features_propagation is not None:
            propagation_vision_feats = backbone_features_propagation["vision_feats"]
            propagation_vision_pos_embeds = backbone_features_propagation["vision_pos_embeds"]
            propagation_feat_sizes = backbone_features_propagation["feat_sizes"]
            if len(propagation_vision_feats) > 1:
                propagation_high_res_features = [
                    x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                    for x, s in zip(propagation_vision_feats[:-1], propagation_feat_sizes[:-1])
                ]
        else:
            assert mode not in ("propagation_only", "propagation_and_interaction")
            assert not run_mem_encoder

        interactive_pix_feat = None
        if mode == "mask_as_output":
            assert self.use_mask_input_as_output_without_sam
            interactive_pix_feat = self._get_interactive_pix_mem(interactive_vision_feats, interactive_feat_sizes)
            sam_outputs = self._use_mask_as_output(
                backbone_features=interactive_pix_feat,
                high_res_features=interactive_high_res_features,
                mask_inputs=mask_inputs,
                multiplex_state=multiplex_state,
            )
            current_out["conditioning_objects"].update(range(mask_inputs.shape[0]))
        else:
            propagation_out = None
            if mode in ("propagation_only", "propagation_and_interaction"):
                pix_feat_with_mem = self._prepare_memory_conditioned_features(
                    frame_idx=frame_idx,
                    is_init_cond_frame=is_init_cond_frame,
                    current_vision_feats=propagation_vision_feats[-1:],
                    current_vision_pos_embeds=propagation_vision_pos_embeds[-1:],
                    feat_sizes=propagation_feat_sizes[-1:],
                    output_dict=output_dict,
                    num_frames=num_frames,
                    track_in_reverse=track_in_reverse,
                    multiplex_state=multiplex_state,
                )
                propagation_out = self._forward_sam_heads(
                    backbone_features=pix_feat_with_mem,
                    propagation_high_res_features=propagation_high_res_features,
                    multimask_output=self._use_multimask(is_init_cond_frame, point_inputs=None),
                    objects_to_interact=list(range(multiplex_state.total_valid_entries)),
                    multiplex_state=multiplex_state,
                )

            interaction_out = None
            if mode in ("interaction_only", "propagation_and_interaction"):
                interactive_pix_feat = self._get_interactive_pix_mem(interactive_vision_feats, interactive_feat_sizes)
                assert mask_inputs is None and point_inputs is not None
                if prev_sam_mask_logits is not None:
                    mask_inputs = prev_sam_mask_logits[objects_to_interact]
                elif mode == "propagation_and_interaction":
                    mask_inputs = propagation_out["low_res_masks"][objects_to_interact]
                if objects_to_interact is not None:
                    assert point_inputs["point_coords"].shape[0] == len(objects_to_interact)
                interaction_out = self._forward_sam_heads(
                    backbone_features=interactive_pix_feat,
                    point_inputs=point_inputs,
                    mask_inputs=mask_inputs,
                    interactive_high_res_features=interactive_high_res_features,
                    multimask_output=self._use_multimask(is_init_cond_frame, point_inputs=point_inputs),
                    objects_to_interact=(
                        objects_to_interact
                        if objects_to_interact is not None
                        else list(range(multiplex_state.total_valid_entries))
                    ),
                    multiplex_state=multiplex_state,
                )
                if objects_to_interact is None:
                    current_out["conditioning_objects"].update(multiplex_state.get_all_valid_object_idx())
                else:
                    current_out["conditioning_objects"].update(objects_to_interact)

            if propagation_out is None:
                sam_outputs = interaction_out
            elif interaction_out is None:
                sam_outputs = propagation_out
            else:
                for k in (
                    "low_res_multimasks",
                    "high_res_multimasks",
                    "low_res_masks",
                    "high_res_masks",
                    "ious",
                    "object_score_logits",
                    "obj_ptr",
                ):
                    src = interaction_out[k]
                    if torch.is_floating_point(src) and src.dtype != propagation_out[k].dtype:
                        src = src.to(dtype=propagation_out[k].dtype)
                    propagation_out[k][objects_to_interact] = src
                sam_outputs = propagation_out

        low_res_masks = sam_outputs["low_res_masks"]
        high_res_masks = sam_outputs["high_res_masks"]
        object_score_logits = sam_outputs["object_score_logits"]

        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [sam_outputs["low_res_multimasks"]]
        current_out["multistep_pred_multimasks_high_res"] = [sam_outputs["high_res_multimasks"]]
        current_out["multistep_pred_ious"] = [sam_outputs["ious"]]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        if self.use_obj_ptrs_in_encoder:
            # like spatial memory, object pointers are stored in the mux space
            current_out["obj_ptr"] = multiplex_state.mux(sam_outputs["obj_ptr"])
        current_out["object_score_logits"] = object_score_logits

        if run_mem_encoder and self.num_maskmem > 0:
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=propagation_vision_feats,
                feat_sizes=propagation_feat_sizes,
                pred_masks_high_res=high_res_masks,
                object_score_logits=object_score_logits,
                is_mask_from_pts=(point_inputs is not None),
                conditioning_objects=current_out["conditioning_objects"],
                multiplex_state=multiplex_state,
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc

        if self.save_image_features:
            current_out["image_features"] = propagation_vision_feats[-1]
            current_out["image_pos_enc"] = propagation_vision_pos_embeds[-1]

        aux_output = {}
        if need_aux_output:
            if interactive_pix_feat is None:
                interactive_pix_feat = self._get_interactive_pix_mem(interactive_vision_feats, interactive_feat_sizes)
            aux_output["interactive_pix_feat"] = interactive_pix_feat
            aux_output["interactive_high_res_features"] = interactive_high_res_features
            aux_output["propagation_vision_feats"] = propagation_vision_feats
            aux_output["propagation_feat_sizes"] = propagation_feat_sizes
        return current_out, aux_output

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        backbone_features_interactive,
        backbone_features_propagation,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,
        run_mem_encoder=True,
        prev_sam_mask_logits=None,
        multiplex_state: MultiplexState = None,
        objects_to_interact: list[int] | None = None,
        new_object_masks: torch.Tensor | None = None,
        new_object_idxs: list[int] | None = None,
        new_object_ids: list[int] | None = None,
        are_new_masks_from_pts: bool = False,
    ) -> dict:
        """Track one frame, optionally merging dynamically added objects into the state."""
        current_out, aux_out = self._track_step_aux(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            backbone_features_interactive=backbone_features_interactive,
            backbone_features_propagation=backbone_features_propagation,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
            run_mem_encoder=(run_mem_encoder and new_object_masks is None),
            prev_sam_mask_logits=prev_sam_mask_logits,
            multiplex_state=multiplex_state,
            objects_to_interact=objects_to_interact,
            need_aux_output=(new_object_masks is not None),
        )
        if new_object_masks is not None:
            assert new_object_idxs is not None
            self.add_new_masks_to_existing_state(
                interactive_pix_feat=aux_out["interactive_pix_feat"],
                interactive_high_res_features=aux_out["interactive_high_res_features"],
                propagation_vision_feats=aux_out["propagation_vision_feats"],
                propagation_feat_sizes=aux_out["propagation_feat_sizes"],
                new_masks=new_object_masks,
                obj_idxs_in_mask=new_object_idxs,
                obj_ids_in_mask=new_object_ids,
                prev_output=current_out,
                multiplex_state=multiplex_state,
                add_mask_to_memory=run_mem_encoder,
                are_masks_from_pts=are_new_masks_from_pts,
            )
        return current_out

    # ------------------------------------------------------------------
    # Dynamic object management
    # ------------------------------------------------------------------

    def add_new_masks_to_existing_state(
        self,
        interactive_pix_feat: torch.Tensor,
        interactive_high_res_features: list,
        propagation_vision_feats: list | None,
        propagation_feat_sizes: list | None,
        new_masks: torch.Tensor,
        obj_idxs_in_mask: list[int],
        obj_ids_in_mask: list[int] | None,
        prev_output: dict,
        multiplex_state: MultiplexState,
        add_mask_to_memory: bool = True,
        are_masks_from_pts: bool = False,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
    ) -> None:
        """Append new mask-seeded objects to an existing frame output and multiplex state."""
        assert self.use_mask_input_as_output_without_sam
        assert new_masks.shape[0] == len(obj_idxs_in_mask)
        num_new_objects = new_masks.shape[0]
        if obj_ids_in_mask is not None:
            assert len(obj_ids_in_mask) == num_new_objects

        if self.use_obj_ptrs_in_encoder:
            existing_pointers = multiplex_state.demux(prev_output["obj_ptr"])

        new_object_idx = multiplex_state.find_next_batch_of_available_indices(
            num_objects=num_new_objects, allow_new_buckets=allow_new_buckets
        )
        multiplex_state.add_objects(
            object_indices=new_object_idx,
            object_ids=obj_ids_in_mask,
            allow_new_buckets=allow_new_buckets,
            prefer_new_buckets=prefer_new_buckets,
        )

        mask_output = self._use_mask_as_output(
            backbone_features=interactive_pix_feat,
            high_res_features=interactive_high_res_features,
            mask_inputs=new_masks,
            multiplex_state=multiplex_state,
            objects_in_mask=new_object_idx,
        )

        interactive_resolution = mask_output["high_res_masks"].shape[-1]
        if prev_output.get("pred_masks_high_res") is not None:
            if prev_output["pred_masks_high_res"].shape[-1] != interactive_resolution:
                prev_output["pred_masks_high_res"] = F.interpolate(
                    prev_output["pred_masks_high_res"],
                    size=(interactive_resolution, interactive_resolution),
                    mode="bilinear",
                    align_corners=False,
                )
        h, w = prev_output["pred_masks"].shape[-2:]
        mask_output["low_res_masks"] = F.interpolate(
            mask_output["low_res_masks"], size=(h, w), align_corners=False, mode="bilinear", antialias=True
        )

        _append(prev_output, mask_output, "pred_masks", "low_res_masks")
        _append(prev_output, mask_output, "pred_masks_high_res", "high_res_masks", strict=False)
        _append(prev_output, mask_output, "object_score_logits", "object_score_logits")
        if "input_masks" in prev_output:
            prev_output["input_masks"] = torch.cat([prev_output["input_masks"], new_masks], dim=0)
        if self.use_obj_ptrs_in_encoder:
            new_pointers = mask_output["obj_ptr"].to(existing_pointers.dtype)
            prev_output["obj_ptr"] = multiplex_state.mux(torch.cat([existing_pointers, new_pointers], dim=0))
        prev_output["conditioning_objects"].update(new_object_idx)

        if add_mask_to_memory:
            assert prev_output["pred_masks_high_res"].shape[0] == multiplex_state.total_valid_entries
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=propagation_vision_feats,
                feat_sizes=propagation_feat_sizes,
                pred_masks_high_res=prev_output["pred_masks_high_res"],
                object_score_logits=prev_output["object_score_logits"],
                conditioning_objects=prev_output["conditioning_objects"],
                is_mask_from_pts=are_masks_from_pts,
                multiplex_state=multiplex_state,
            )
            prev_output["maskmem_features"] = maskmem_features
            prev_output["maskmem_pos_enc"] = maskmem_pos_enc
            if self.save_image_features:
                assert "image_features" in prev_output and "image_pos_enc" in prev_output

    def recondition_masks_in_existing_state(
        self,
        interactive_pix_feat: torch.Tensor,
        interactive_high_res_features: list,
        propagation_vision_feats: list | None,
        propagation_feat_sizes: list | None,
        new_masks: torch.Tensor,
        obj_idxs_in_mask: list[int],
        obj_ids_in_mask: list[int] | None,
        prev_output: dict,
        multiplex_state: MultiplexState,
        add_mask_to_memory: bool = True,
    ) -> None:
        """Re-anchor existing objects with fresh masks in an existing frame output."""
        assert self.use_mask_input_as_output_without_sam
        assert new_masks.shape[0] == len(obj_idxs_in_mask)
        if obj_ids_in_mask is not None:
            assert len(obj_ids_in_mask) == new_masks.shape[0]

        if self.use_obj_ptrs_in_encoder:
            existing_pointers = multiplex_state.demux(prev_output["obj_ptr"])

        mask_output = self._use_mask_as_output(
            backbone_features=interactive_pix_feat,
            high_res_features=interactive_high_res_features,
            mask_inputs=new_masks,
            multiplex_state=multiplex_state,
            objects_in_mask=obj_idxs_in_mask,
        )
        h, w = prev_output["pred_masks"].shape[-2:]
        mask_output["low_res_masks"] = F.interpolate(
            mask_output["low_res_masks"], size=(h, w), align_corners=False, mode="bilinear", antialias=True
        )
        _merge(prev_output, mask_output, "pred_masks", "low_res_masks", obj_idxs_in_mask)
        _merge(prev_output, mask_output, "pred_masks_high_res", "high_res_masks", obj_idxs_in_mask, strict=False)
        _merge(prev_output, mask_output, "object_score_logits", "object_score_logits", obj_idxs_in_mask)
        if "input_masks" in prev_output:
            prev_output["input_masks"][obj_idxs_in_mask] = new_masks
        if self.use_obj_ptrs_in_encoder:
            existing_pointers[obj_idxs_in_mask] = mask_output["obj_ptr"].to(existing_pointers.dtype)
            prev_output["obj_ptr"] = multiplex_state.mux(existing_pointers)
        prev_output["conditioning_objects"].update(obj_idxs_in_mask)

        if add_mask_to_memory:
            assert prev_output["pred_masks_high_res"].shape[0] == multiplex_state.total_valid_entries
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                current_vision_feats=propagation_vision_feats,
                feat_sizes=propagation_feat_sizes,
                pred_masks_high_res=prev_output["pred_masks_high_res"],
                object_score_logits=prev_output["object_score_logits"],
                conditioning_objects=prev_output["conditioning_objects"],
                is_mask_from_pts=False,
                multiplex_state=multiplex_state,
            )
            prev_output["maskmem_features"] = maskmem_features
            prev_output["maskmem_pos_enc"] = maskmem_pos_enc

    # ------------------------------------------------------------------
    # Session API (demo-style inference state)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def init_state(
        self,
        video_height,
        video_width,
        num_frames,
        cached_features=None,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        device: torch.device | str | None = None,
    ):
        """Initialize an inference state (frames/features are supplied externally via cached_features)."""
        assert self.apply_sigmoid_to_mask_logits_for_mem_enc, (
            "Multi-object tracking requires sigmoid in memory encoder for non-overlapping constraints."
        )
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        inference_state = {
            "num_frames": num_frames,
            "offload_video_to_cpu": offload_video_to_cpu,
            "offload_state_to_cpu": offload_state_to_cpu,
            "video_height": video_height,
            "video_width": video_width,
            "device": torch.device(device),
            "storage_device": torch.device("cpu") if offload_state_to_cpu else torch.device(device),
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {} if cached_features is None else cached_features,
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            "first_ann_frame_idx": None,
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "consolidated_frame_inds": {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()},
            "tracking_has_started": False,
            "frames_already_tracked": {},
            "multiplex_state": None,
        }
        return inference_state

    def _obj_id_to_idx(self, inference_state, obj_id, error_if_new=False):
        """Map client-side object id to model-side object index (allocating when allowed)."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx
        if (self.is_dynamic_model or not inference_state["tracking_has_started"]) and not error_if_new:
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }
            return obj_idx
        raise RuntimeError(f"Cannot add new object id {obj_id}. Existing ids: {inference_state['obj_ids']}.")

    def _obj_idx_to_id(self, inference_state, obj_idx):
        """Map model-side object index to client-side object id."""
        return inference_state["obj_idx_to_id"][obj_idx]

    def _get_obj_num(self, inference_state):
        """Number of live objects in the multiplex state."""
        return inference_state["multiplex_state"].total_valid_entries

    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        """Resize scores to the original video resolution and apply output non-overlap constraints."""
        device = inference_state["device"]
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        any_res_masks = any_res_masks.to(device, non_blocking=True)
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = F.interpolate(
                any_res_masks, size=(video_H, video_W), mode="bilinear", align_corners=False
            )
        if self.non_overlap_masks_for_output:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        return any_res_masks, video_res_masks

    @torch.inference_mode()
    def add_new_masks(self, inference_state, frame_idx, obj_ids, masks, reconditioning=False):
        """Add (or recondition) mask-seeded objects on a frame.

        Args:
            inference_state (dict): Session state from init_state.
            frame_idx (int): Frame receiving the masks.
            obj_ids (list[int]): Client-side ids, one per mask.
            masks (torch.Tensor): (N, H, W) binary or soft masks.
            reconditioning (bool): Re-anchor existing objects instead of adding new ones.
        """
        if hasattr(obj_ids, "tolist"):
            obj_ids = obj_ids.tolist()
        obj_idxs = [self._obj_id_to_idx(inference_state, obj_id, error_if_new=reconditioning) for obj_id in obj_ids]
        point_inputs_per_frame = [inference_state["point_inputs_per_obj"][i] for i in obj_idxs]
        mask_inputs_per_frame = [inference_state["mask_inputs_per_obj"][i] for i in obj_idxs]

        assert masks.dim() == 3
        num_objects, mask_H, mask_W = masks.shape
        assert num_objects == len(obj_ids)
        masks_inputs_orig = masks[:, None].float().to(inference_state["device"])

        if mask_H != self.input_mask_size or mask_W != self.input_mask_size:
            mask_inputs = F.interpolate(
                masks_inputs_orig,
                size=(self.input_mask_size, self.input_mask_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,
            )
        else:
            mask_inputs = masks_inputs_orig

        video_H, video_W = inference_state["video_height"], inference_state["video_width"]
        if mask_H != video_H or mask_W != video_W:
            mask_inputs_video_res = F.interpolate(
                masks_inputs_orig, size=(video_H, video_W), align_corners=False, mode="bilinear", antialias=True
            )
        else:
            mask_inputs_video_res = masks_inputs_orig
        mask_inputs_video_res = mask_inputs_video_res > 0.5

        multiplex_state = inference_state["multiplex_state"]
        is_new_state = multiplex_state is None
        if not reconditioning:
            if is_new_state:
                multiplex_state = self.multiplex_controller.get_state(
                    num_valid_entries=num_objects,
                    device=inference_state["device"],
                    dtype=torch.float32,
                    random=False,
                    object_ids=obj_ids,
                )
                inference_state["multiplex_state"] = multiplex_state
            else:
                assert self.is_dynamic_model, "New objects are not allowed after state creation"

        for i in range(num_objects):
            mask_inputs_per_frame[i][frame_idx] = mask_inputs_video_res[i : i + 1]
            point_inputs_per_frame[i].pop(frame_idx, None)

        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        reverse = False if is_init_cond_frame else inference_state["frames_already_tracked"][frame_idx]["reverse"]
        obj_output_dicts = [inference_state["output_dict_per_obj"][i] for i in obj_idxs]
        obj_temp_output_dicts = [inference_state["temp_output_dict_per_obj"][i] for i in obj_idxs]
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        allow_new_buckets_local = False
        if not is_new_state and not reconditioning and multiplex_state is not None:
            if multiplex_state.available_slots < num_objects:
                allow_new_buckets_local = True

        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=inference_state["output_dict"],
            frame_idx=frame_idx,
            batch_size=num_objects,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            run_mem_encoder=False,  # memory is encoded in propagate_in_video_preflight
            add_to_existing_state=not is_new_state and not reconditioning,
            new_obj_idxs=obj_idxs,
            new_obj_ids=obj_ids,
            allow_new_buckets=allow_new_buckets_local,
            reconditioning=reconditioning,
        )
        _, video_res_masks = self._get_orig_video_res_output(inference_state, current_out["pred_masks"])
        obj_idxs_t = torch.as_tensor(obj_idxs, device=video_res_masks.device)
        video_res_masks[obj_idxs_t] = torch.where(mask_inputs_video_res, -NO_OBJ_SCORE, NO_OBJ_SCORE)
        current_out["pred_masks_video_res"] = video_res_masks
        current_out["local_obj_id_to_idx"] = deepcopy(inference_state["obj_id_to_idx"])

        if is_cond and frame_idx in inference_state["output_dict"]["non_cond_frame_outputs"]:
            del inference_state["output_dict"]["non_cond_frame_outputs"][frame_idx]
            inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].discard(frame_idx)
        inference_state["output_dict"][storage_key][frame_idx] = current_out
        inference_state["consolidated_frame_inds"][storage_key].add(frame_idx)

        # per-object placeholder slices at video resolution
        for i, obj_idx in enumerate(obj_idxs):
            obj_temp_output_dicts[i][storage_key][frame_idx] = {
                "pred_masks_video_res": current_out["pred_masks_video_res"][obj_idx : obj_idx + 1]
            }
            obj_output_dicts[i][storage_key][frame_idx] = obj_temp_output_dicts[i][storage_key][frame_idx]

        # suppress overlaps between new masks and other objects' temp outputs on this frame
        combined_new_mask = mask_inputs_video_res.any(dim=0, keepdim=True)
        num_new = len(obj_idxs)
        exclude_self_masks = {}
        if num_new > 1:
            for i in range(num_new):
                other = torch.cat(
                    [
                        torch.arange(i, device=mask_inputs_video_res.device),
                        torch.arange(i + 1, num_new, device=mask_inputs_video_res.device),
                    ]
                )
                exclude_self_masks[obj_idxs[i]] = mask_inputs_video_res[other].any(dim=0, keepdim=True)
        obj_idxs_set = set(obj_idxs)
        for obj_idx2, obj_temp_output_dict2 in inference_state["temp_output_dict_per_obj"].items():
            current_out2 = obj_temp_output_dict2[storage_key].get(frame_idx, None)
            if current_out2 is None:
                continue
            if obj_idx2 not in obj_idxs_set:
                suppress_mask = combined_new_mask
            elif obj_idx2 in exclude_self_masks:
                suppress_mask = exclude_self_masks[obj_idx2]
            else:
                continue
            current_out2["pred_masks_video_res"] = torch.where(
                suppress_mask, NO_OBJ_SCORE, current_out2["pred_masks_video_res"]
            )

        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=False, consolidate_at_video_res=True
        )
        _, video_res_masks = self._get_orig_video_res_output(inference_state, consolidated_out["pred_masks_video_res"])
        consolidated_out["local_obj_id_to_idx"] = current_out["local_obj_id_to_idx"]
        return frame_idx, obj_ids, None, video_res_masks

    def _consolidate_temp_output_across_obj(
        self, inference_state, frame_idx, is_cond, run_mem_encoder, consolidate_at_video_res=False
    ):
        """Consolidate per-object temporary outputs on a frame into a single all-object output."""
        batch_size = self._get_obj_num(inference_state)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        max_obj_idx = batch_size - 1
        for obj_idx in inference_state["temp_output_dict_per_obj"]:
            max_obj_idx = max(max_obj_idx, obj_idx)
        for obj_idx in inference_state["output_dict_per_obj"]:
            max_obj_idx = max(max_obj_idx, obj_idx)
        consolidated_batch_size = max(max_obj_idx + 1, 0)

        if consolidate_at_video_res:
            assert not run_mem_encoder, "memory encoder cannot run at video resolution"
            consolidated_H = inference_state["video_height"]
            consolidated_W = inference_state["video_width"]
            consolidated_mask_key = "pred_masks_video_res"
        else:
            consolidated_H = consolidated_W = self.low_res_mask_size
            consolidated_mask_key = "pred_masks"

        consolidated_out = {
            "conditioning_objects": None,
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            "image_features": None,
            "image_pos_enc": None,
            "obj_ptr": None,
            consolidated_mask_key: torch.full(
                size=(consolidated_batch_size, 1, consolidated_H, consolidated_W),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["storage_device"],
            ),
        }

        all_out = inference_state["output_dict"]["cond_frame_outputs"].get(frame_idx, None)
        if all_out is None:
            all_out = inference_state["output_dict"]["non_cond_frame_outputs"].get(frame_idx, None)
        need_to_reconstruct_from_per_obj = all_out is None

        if need_to_reconstruct_from_per_obj:
            conditioning_objects = set()
            for obj_idx in range(batch_size):
                point_inputs = inference_state["point_inputs_per_obj"].get(obj_idx, {})
                if point_inputs.get(frame_idx) is not None:
                    conditioning_objects.add(obj_idx)
                    continue
                mask_inputs = inference_state["mask_inputs_per_obj"].get(obj_idx, {})
                if mask_inputs.get(frame_idx) is not None:
                    conditioning_objects.add(obj_idx)
            consolidated_out["conditioning_objects"] = conditioning_objects
        else:
            consolidated_out["conditioning_objects"] = all_out.get("conditioning_objects", set())
            consolidated_out["obj_ptr"] = all_out["obj_ptr"]
            consolidated_out["object_score_logits"] = all_out["object_score_logits"]
            consolidated_out["maskmem_features"] = all_out.get("maskmem_features")
            consolidated_out["maskmem_pos_enc"] = all_out.get("maskmem_pos_enc")
            consolidated_out["image_features"] = all_out.get("image_features")
            consolidated_out["image_pos_enc"] = all_out.get("image_pos_enc")
            consolidated_out["local_obj_id_to_idx"] = all_out.get("local_obj_id_to_idx", {})
            all_mask = all_out.get("pred_masks_video_res", all_out["pred_masks"])
            if all_mask.shape[-2:] == (consolidated_H, consolidated_W):
                consolidated_out[consolidated_mask_key] = all_mask
            else:
                consolidated_out[consolidated_mask_key] = F.interpolate(
                    all_mask,
                    size=(consolidated_H, consolidated_W),
                    mode="bilinear",
                    align_corners=False,
                    antialias=all_mask.shape[-1] > consolidated_W,
                )

        obj_score_logits_list = []
        if need_to_reconstruct_from_per_obj:
            consolidated_out["object_score_logits"] = torch.full(
                (consolidated_batch_size, 1),
                NO_OBJ_SCORE,
                dtype=torch.float32,
                device=inference_state["storage_device"],
            )

        for obj_idx in range(consolidated_batch_size):
            if obj_idx not in inference_state["temp_output_dict_per_obj"]:
                continue
            if obj_idx not in inference_state["output_dict_per_obj"]:
                continue
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                continue
            obj_mask = out.get("pred_masks_video_res")
            if obj_mask is None:
                obj_mask = out.get("pred_masks")
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]
            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                consolidated_pred_masks[obj_idx : obj_idx + 1] = obj_mask.to(consolidated_pred_masks.dtype)
            else:
                resized_obj_mask = F.interpolate(
                    obj_mask,
                    size=consolidated_pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                    antialias="pred_masks_video_res" in out,
                )
                consolidated_pred_masks[obj_idx : obj_idx + 1] = resized_obj_mask.to(consolidated_pred_masks.dtype)
            if need_to_reconstruct_from_per_obj and "object_score_logits" in out:
                obj_score_logits_list.append(out["object_score_logits"])

        if need_to_reconstruct_from_per_obj:
            if not obj_score_logits_list and run_mem_encoder:
                run_mem_encoder = False  # encode during propagation instead
            if obj_score_logits_list:
                consolidated_out["object_score_logits"] = torch.cat(obj_score_logits_list, dim=0)
            else:
                consolidated_out["object_score_logits"] = torch.zeros(
                    (batch_size, 1), dtype=torch.float32, device=inference_state["device"]
                )
            consolidated_out["obj_ptr"] = None

        if run_mem_encoder:
            device = inference_state["device"]
            high_res_masks = F.interpolate(
                consolidated_out["pred_masks"].to(device, non_blocking=True),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            high_res_masks = self._apply_non_overlapping_constraints(high_res_masks)
            maskmem_features, maskmem_pos_enc, image_features, image_pos_enc = self._run_memory_encoder(
                inference_state=inference_state,
                frame_idx=frame_idx,
                batch_size=batch_size,
                high_res_masks=high_res_masks,
                object_score_logits=consolidated_out["object_score_logits"],
                is_mask_from_pts=True,
                conditioning_objects=consolidated_out["conditioning_objects"],
            )
            consolidated_out["maskmem_features"] = maskmem_features
            consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc
            consolidated_out["image_features"] = image_features
            consolidated_out["image_pos_enc"] = image_pos_enc
        return consolidated_out

    @torch.inference_mode()
    def propagate_in_video_preflight(self, inference_state, run_mem_encoder=True):
        """Consolidate temporary outputs into output_dict before propagation starts."""
        inference_state["tracking_has_started"] = True
        batch_size = self._get_obj_num(inference_state)
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        for is_cond in (False, True):
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            for frame_idx in temp_frame_inds:
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=run_mem_encoder
                )
                output_dict[storage_key][frame_idx] = consolidated_out
                self._add_output_per_object(inference_state, frame_idx, consolidated_out, storage_key)
                if self.clear_non_cond_mem_around_input and (self.clear_non_cond_mem_for_multi_obj or batch_size <= 1):
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()

        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

        all_consolidated_frame_inds = (
            consolidated_frame_inds["cond_frame_outputs"] | consolidated_frame_inds["non_cond_frame_outputs"]
        )
        input_frames_inds = set()
        for point_inputs_per_frame in inference_state["point_inputs_per_obj"].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in inference_state["mask_inputs_per_obj"].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())
        assert all_consolidated_frame_inds == input_frames_inds
        if inference_state["first_ann_frame_idx"] is None:
            inference_state["first_ann_frame_idx"] = min(input_frames_inds, default=None)
        if inference_state["first_ann_frame_idx"] not in output_dict["cond_frame_outputs"]:
            inference_state["first_ann_frame_idx"] = min(output_dict["cond_frame_outputs"], default=None)

    def _get_processing_order(self, inference_state, start_frame_idx, max_frame_num_to_track, reverse):
        """Frame indices to process for this propagation call."""
        num_frames = inference_state["num_frames"]
        if self.always_start_from_first_ann_frame:
            start_frame_idx = inference_state["first_ann_frame_idx"]
        if start_frame_idx is None:
            start_frame_idx = min(inference_state["output_dict"]["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            processing_order = range(start_frame_idx, end_frame_idx - 1, -1) if start_frame_idx > 0 else [0]
        else:
            end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        return processing_order

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx,
        max_frame_num_to_track,
        reverse,
        tqdm_disable=False,
        run_mem_encoder=True,
    ):
        """Propagate all objects through the video; yields per-frame results incl. object scores."""
        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        obj_ids = inference_state["obj_ids"]
        batch_size = self._get_obj_num(inference_state)
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No prompts are provided; please add masks first")
        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )
        processing_order = self._get_processing_order(inference_state, start_frame_idx, max_frame_num_to_track, reverse)
        for frame_idx in TQDM(processing_order, desc="propagate in video", disable=tqdm_disable):
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                obj_scores = current_out["object_score_logits"]
                if clear_non_cond_mem:
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                obj_scores = current_out["object_score_logits"]
            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=run_mem_encoder,
                )
                obj_scores = current_out["object_score_logits"]
                current_out["local_obj_id_to_idx"] = deepcopy(inference_state["obj_id_to_idx"])
                output_dict[storage_key][frame_idx] = current_out
            self._add_output_per_object(inference_state, frame_idx, current_out, storage_key)
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}
            low_res_masks, video_res_masks = self._get_orig_video_res_output(inference_state, pred_masks)
            yield frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores

    def _add_output_per_object(self, inference_state, frame_idx, current_out, storage_key):
        """Split a multi-object output into per-object slices sharing the same storage."""
        for obj_idx, obj_output_dict in inference_state["output_dict_per_obj"].items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_output_dict[storage_key][frame_idx] = {
                "pred_masks": current_out["pred_masks"][obj_slice],
                "object_score_logits": current_out["object_score_logits"][obj_slice],
            }

    @torch.inference_mode()
    def clear_all_points_in_frame(self, inference_state, frame_idx, obj_id, need_output=True):
        """Remove all inputs on a frame for one object, downgrading the frame when it has none left."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

        batch_size = self._get_obj_num(inference_state)
        frame_has_input = False
        for obj_idx2 in range(batch_size):
            if frame_idx in inference_state["point_inputs_per_obj"].get(obj_idx2, {}):
                frame_has_input = True
                break
            if frame_idx in inference_state["mask_inputs_per_obj"].get(obj_idx2, {}):
                frame_has_input = True
                break

        if not frame_has_input:
            output_dict = inference_state["output_dict"]
            consolidated_frame_inds = inference_state["consolidated_frame_inds"]
            consolidated_frame_inds["cond_frame_outputs"].discard(frame_idx)
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)
            out = output_dict["cond_frame_outputs"].pop(frame_idx, None)
            if out is not None:
                output_dict["non_cond_frame_outputs"][frame_idx] = out
                inference_state["frames_already_tracked"].pop(frame_idx, None)
            for obj_idx2 in range(batch_size):
                if obj_idx2 not in inference_state["output_dict_per_obj"]:
                    continue
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx2]
                obj_out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
                if obj_out is not None:
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = obj_out
            if len(output_dict["cond_frame_outputs"]) == 0:
                self._reset_tracking_results(inference_state)

        if not need_output:
            return
        obj_ids = inference_state["obj_ids"]
        is_cond = any(
            frame_idx in obj_temp_output_dict["cond_frame_outputs"]
            for obj_temp_output_dict in temp_output_dict_per_obj.values()
        )
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=False, consolidate_at_video_res=True
        )
        _, video_res_masks = self._get_orig_video_res_output(inference_state, consolidated_out["pred_masks_video_res"])
        return frame_idx, obj_ids, None, video_res_masks

    @torch.inference_mode()
    def clear_all_points_in_video(self, inference_state):
        """Remove all inputs and objects across the whole video."""
        self._reset_tracking_results(inference_state)
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()
        inference_state["multiplex_state"] = None

    def _reset_tracking_results(self, inference_state):
        """Reset all tracking inputs and results."""
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"].clear()
        inference_state["first_ann_frame_idx"] = None

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Look up (or compute) the backbone features for a frame."""
        image, backbone_out = inference_state["cached_features"].get(frame_idx, (None, None))
        if backbone_out is None:
            assert "images" in inference_state, (
                "no cached features for this frame and no frames stored; populate cached_features"
            )
            image = inference_state["images"][frame_idx].to(inference_state["device"]).float().unsqueeze(0)
            backbone_out = self.forward_image(
                image, need_sam3_out=True, need_interactive_out=True, need_propagation_out=True
            )
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}
        features = self._prepare_backbone_features(backbone_out)
        return image, features

    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
        add_to_existing_state: bool = False,
        new_obj_idxs: list[int] | None = None,
        new_obj_ids: list[int] | None = None,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
        reconditioning: bool = False,
        objects_to_interact: list[int] | None = None,
    ):
        """Run tracking on a single frame and return a compact output for the memory bank."""
        image, backbone_features = self._get_image_feature(inference_state, frame_idx, batch_size)
        if add_to_existing_state or reconditioning:
            assert new_obj_idxs is not None and new_obj_ids is not None
        backbone_features_interactive = backbone_features["interactive"]
        backbone_features_propagation = backbone_features["sam2_backbone_out"]

        if add_to_existing_state or reconditioning:
            existing_out = output_dict["cond_frame_outputs"].get(frame_idx)
            if existing_out is None:
                existing_out = output_dict["non_cond_frame_outputs"].get(frame_idx)
            if existing_out is None:
                raise RuntimeError(f"No existing output found for frame {frame_idx}")
            interactive_pix_feat = self._get_interactive_pix_mem(
                backbone_features_interactive["vision_feats"], backbone_features_interactive["feat_sizes"]
            )
            interactive_high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(
                    backbone_features_interactive["vision_feats"][:-1],
                    backbone_features_interactive["feat_sizes"][:-1],
                )
            ]
            propagation_vision_feats = backbone_features_propagation["vision_feats"] if run_mem_encoder else None
            propagation_feat_sizes = backbone_features_propagation["feat_sizes"] if run_mem_encoder else None
            if reconditioning:
                self.recondition_masks_in_existing_state(
                    interactive_pix_feat=interactive_pix_feat,
                    interactive_high_res_features=interactive_high_res_features,
                    propagation_vision_feats=propagation_vision_feats,
                    propagation_feat_sizes=propagation_feat_sizes,
                    new_masks=mask_inputs,
                    obj_idxs_in_mask=new_obj_idxs,
                    obj_ids_in_mask=new_obj_ids,
                    prev_output=existing_out,
                    multiplex_state=inference_state["multiplex_state"],
                    add_mask_to_memory=run_mem_encoder,
                )
            else:
                new_masks_from_points = None
                if mask_inputs is None and point_inputs is not None:
                    interaction_out = self._forward_sam_heads(
                        backbone_features=interactive_pix_feat,
                        point_inputs=point_inputs,
                        mask_inputs=None,
                        interactive_high_res_features=interactive_high_res_features,
                        multimask_output=self._use_multimask(is_init_cond_frame, point_inputs=point_inputs),
                        objects_to_interact=new_obj_idxs,
                        multiplex_state=inference_state["multiplex_state"],
                    )
                    new_masks_from_points = interaction_out["low_res_masks"]
                self.add_new_masks_to_existing_state(
                    interactive_pix_feat=interactive_pix_feat,
                    interactive_high_res_features=interactive_high_res_features,
                    propagation_vision_feats=propagation_vision_feats,
                    propagation_feat_sizes=propagation_feat_sizes,
                    new_masks=mask_inputs if mask_inputs is not None else new_masks_from_points,
                    obj_idxs_in_mask=new_obj_idxs,
                    obj_ids_in_mask=new_obj_ids,
                    prev_output=existing_out,
                    multiplex_state=inference_state["multiplex_state"],
                    add_mask_to_memory=run_mem_encoder,
                    are_masks_from_pts=(mask_inputs is None),
                    allow_new_buckets=allow_new_buckets,
                    prefer_new_buckets=prefer_new_buckets,
                )
            current_out = existing_out
        else:
            assert point_inputs is None or mask_inputs is None
            current_out = self.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                backbone_features_interactive=backbone_features_interactive,
                backbone_features_propagation=backbone_features_propagation,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                output_dict=output_dict,
                num_frames=inference_state["num_frames"],
                track_in_reverse=reverse,
                run_mem_encoder=run_mem_encoder,
                prev_sam_mask_logits=prev_sam_mask_logits,
                multiplex_state=inference_state["multiplex_state"],
                objects_to_interact=objects_to_interact,
            )

        storage_device = inference_state["storage_device"]
        maskmem_features = current_out.get("maskmem_features")
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(device=storage_device, dtype=torch.bfloat16, non_blocking=True)
        if current_out.get("image_features") is not None:
            image_features = current_out["image_features"].to(storage_device, non_blocking=True)
            image_pos_enc = current_out["image_pos_enc"].to(storage_device, non_blocking=True)
        else:
            image_features = image_pos_enc = None
        pred_masks_gpu = current_out["pred_masks"]
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "image_features": image_features,
            "image_pos_enc": image_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": current_out["obj_ptr"],
            "object_score_logits": current_out["object_score_logits"],
            "conditioning_objects": current_out["conditioning_objects"],
        }
        return compact_current_out, pred_masks_gpu

    def _run_memory_encoder(
        self,
        inference_state,
        frame_idx,
        batch_size,
        high_res_masks,
        object_score_logits,
        is_mask_from_pts,
        conditioning_objects=None,
    ):
        """Re-run the memory encoder on updated masks (e.g. after non-overlap constraints)."""
        image, backbone_features = self._get_image_feature(inference_state, frame_idx, batch_size)
        backbone_features_propagation = backbone_features["sam2_backbone_out"]
        propagation_vision_feats = backbone_features_propagation["vision_feats"]
        propagation_vision_pos_embeds = backbone_features_propagation["vision_pos_embeds"]
        propagation_feat_sizes = backbone_features_propagation["feat_sizes"]

        if conditioning_objects is None:
            output_dict = inference_state["output_dict"]
            for storage_key in ("cond_frame_outputs", "non_cond_frame_outputs"):
                if frame_idx in output_dict[storage_key]:
                    conditioning_objects = output_dict[storage_key][frame_idx]["conditioning_objects"]
                    break
            else:
                raise ValueError(f"conditioning objects not found at {frame_idx=}")

        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=propagation_vision_feats,
            feat_sizes=propagation_feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=object_score_logits,
            is_mask_from_pts=is_mask_from_pts,
            conditioning_objects=conditioning_objects,
            multiplex_state=inference_state["multiplex_state"],
        )
        storage_device = inference_state["storage_device"]
        maskmem_features = maskmem_features.to(torch.bfloat16).to(storage_device, non_blocking=True)
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, {"maskmem_pos_enc": maskmem_pos_enc})
        image_features = propagation_vision_feats[-1].to(storage_device, non_blocking=True)
        image_pos_enc = propagation_vision_pos_embeds[-1].to(storage_device, non_blocking=True)
        return maskmem_features, maskmem_pos_enc, image_features, image_pos_enc

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        """Cache the (frame-invariant) memory position encoding once per session."""
        model_constants = inference_state["constants"]
        out_maskmem_pos_enc = current_out.get("maskmem_pos_enc")
        if out_maskmem_pos_enc is None:
            return None
        if "maskmem_pos_enc" not in model_constants:
            assert isinstance(out_maskmem_pos_enc, list)
            model_constants["maskmem_pos_enc"] = [x[0:1].clone() for x in out_maskmem_pos_enc]
        maskmem_pos_enc = model_constants["maskmem_pos_enc"]
        batch_size = out_maskmem_pos_enc[0].size(0)
        return [x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc]

    @torch.inference_mode()
    def remove_object(self, inference_state, obj_id: int, strict=False, need_output=True):
        """Remove a single object from the tracking state."""
        return self.remove_objects(inference_state, obj_ids=[obj_id], strict=strict, need_output=need_output)

    @torch.inference_mode()
    def remove_objects(self, inference_state, obj_ids, strict=False, need_output=True):
        """Remove objects from the tracking state, dropping empty buckets and remapping indices."""
        obj_ids = list(obj_ids)
        old_obj_idxs_to_rm = [inference_state["obj_id_to_idx"].get(obj_id, None) for obj_id in obj_ids]
        updated_frames = []
        actually_used_obj_ids = []
        for old_obj_idx_to_rm, obj_id in zip(old_obj_idxs_to_rm, obj_ids):
            if old_obj_idx_to_rm is None:
                if strict:
                    raise ValueError(f"Object id {obj_id} does not exist in the tracking state.")
            else:
                actually_used_obj_ids.append(obj_id)
        if not actually_used_obj_ids:
            return inference_state["obj_ids"], updated_frames
        old_obj_idxs_to_rm = [x for x in old_obj_idxs_to_rm if x is not None]
        obj_ids = actually_used_obj_ids

        # Step 0: clear inputs of the removed objects (may downgrade cond frames)
        all_obj_input_frames_inds = set()
        for old_obj_idx_to_rm, obj_id in zip(old_obj_idxs_to_rm, obj_ids):
            obj_input_frames_inds = set()
            obj_input_frames_inds.update(inference_state["point_inputs_per_obj"][old_obj_idx_to_rm])
            obj_input_frames_inds.update(inference_state["mask_inputs_per_obj"][old_obj_idx_to_rm])
            for frame_idx in obj_input_frames_inds:
                self.clear_all_points_in_frame(inference_state, frame_idx, obj_id, need_output=False)
            all_obj_input_frames_inds.update(obj_input_frames_inds)

        # Step 1: update id mappings
        old_obj_ids = inference_state["obj_ids"]
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = old_obj_inds.copy()
        for old_obj_idx_to_rm in old_obj_idxs_to_rm:
            remain_old_obj_inds.remove(old_obj_idx_to_rm)
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        inference_state["obj_id_to_idx"] = dict(zip(new_obj_ids, new_obj_inds))
        inference_state["obj_idx_to_id"] = dict(zip(new_obj_inds, new_obj_ids))
        inference_state["obj_ids"] = new_obj_ids
        if len(new_obj_ids) == 0:
            return new_obj_ids, updated_frames

        # Step 2: shift per-object dict keys
        def _map_keys(container):
            new_kvs = []
            for k in old_obj_inds:
                v = container.pop(k)
                if k in old_idx_to_new_idx:
                    new_kvs.append((old_idx_to_new_idx[k], v))
            container.update(new_kvs)

        _map_keys(inference_state["point_inputs_per_obj"])
        _map_keys(inference_state["mask_inputs_per_obj"])
        _map_keys(inference_state["output_dict_per_obj"])
        _map_keys(inference_state["temp_output_dict_per_obj"])

        multiplex_state: MultiplexState = inference_state["multiplex_state"]
        buckets_to_keep = multiplex_state.remove_objects(old_obj_idxs_to_rm, strict=True)
        obj_ids = set(obj_ids)

        # Step 3: slice packed per-bucket/per-object storage
        def _slice_state(output_dict, storage_key):
            for frame_idx, out in output_dict[storage_key].items():
                out["maskmem_features"] = out["maskmem_features"][buckets_to_keep]
                out["maskmem_pos_enc"] = [x[buckets_to_keep] for x in out["maskmem_pos_enc"]]
                out["maskmem_pos_enc"] = self._get_maskmem_pos_enc(inference_state, out)
                out["obj_ptr"] = out["obj_ptr"][buckets_to_keep]
                local_obj_id_to_idx = out["local_obj_id_to_idx"]
                local_remain_old_obj_inds = [
                    obj_idx for obj_id, obj_idx in local_obj_id_to_idx.items() if obj_id not in obj_ids
                ]
                max_rows = min(out["pred_masks"].shape[0], out["object_score_logits"].shape[0])
                keep_indices = [idx for idx in local_remain_old_obj_inds if 0 <= idx < max_rows]
                out["pred_masks"] = out["pred_masks"][keep_indices]
                out["object_score_logits"] = out["object_score_logits"][keep_indices]
                sliced_conditioning_objects = set()
                new_local_obj_id_to_idx = {}
                old_to_new = {old_idx: new_i for new_i, old_idx in enumerate(keep_indices)}
                for obj_id, old_idx in local_obj_id_to_idx.items():
                    if obj_id not in obj_ids and old_idx in old_to_new:
                        new_idx = old_to_new[old_idx]
                        new_local_obj_id_to_idx[obj_id] = new_idx
                        if old_idx in out["conditioning_objects"]:
                            sliced_conditioning_objects.add(new_idx)
                out["local_obj_id_to_idx"] = new_local_obj_id_to_idx
                out["conditioning_objects"] = sliced_conditioning_objects
                self._add_output_per_object(inference_state, frame_idx, out, storage_key)

        _slice_state(inference_state["output_dict"], "cond_frame_outputs")
        _slice_state(inference_state["output_dict"], "non_cond_frame_outputs")

        # Step 4: refresh outputs on frames that had inputs from the removed objects
        if need_output:
            temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
            for frame_idx in all_obj_input_frames_inds:
                is_cond = any(
                    frame_idx in obj_temp_output_dict["cond_frame_outputs"]
                    for obj_temp_output_dict in temp_output_dict_per_obj.values()
                )
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=False, consolidate_at_video_res=True
                )
                _, video_res_masks = self._get_orig_video_res_output(
                    inference_state, consolidated_out["pred_masks_video_res"]
                )
                updated_frames.append((frame_idx, video_res_masks))
        return inference_state["obj_ids"], updated_frames

    def _clear_non_cond_mem_around_input(self, inference_state, frame_idx):
        """Clear non-conditioning memory around an interacted frame."""
        r = self.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.num_maskmem
        frame_idx_end = frame_idx + r * self.num_maskmem
        non_cond_frame_outputs = inference_state["output_dict"]["non_cond_frame_outputs"]
        for t in range(frame_idx_begin, frame_idx_end + 1):
            non_cond_frame_outputs.pop(t, None)
            for obj_output_dict in inference_state["output_dict_per_obj"].values():
                obj_output_dict["non_cond_frame_outputs"].pop(t, None)

    # ------------------------------------------------------------------
    # Output suppression heuristics (used by the semantic video orchestrator)
    # ------------------------------------------------------------------

    @staticmethod
    def _suppress_shrinked_masks(pred_masks, new_pred_masks, shrink_threshold=0.3):
        """Suppress masks whose area shrinks heavily under pixel-wise non-overlap."""
        area_before = torch.clamp((pred_masks > 0).sum(dim=(-1, -2)), min=1.0)
        area_after = (new_pred_masks > 0).sum(dim=(-1, -2))
        keep = (area_after / area_before) >= shrink_threshold
        keep_mask = keep[..., None, None].expand_as(pred_masks)
        return torch.where(keep_mask, pred_masks, torch.clamp(pred_masks, max=-10.0))

    @staticmethod
    def _suppress_object_pw_area_shrinkage(pred_masks):
        """Fully suppress masks that shrink heavily under pixel-wise non-overlap constraints."""
        if pred_masks.size(0) == 1:
            return pred_masks
        pixel_level = SAM3MultiplexModel._apply_non_overlapping_constraints(pred_masks)
        return SAM3MultiplexModel._suppress_shrinked_masks(pred_masks, pixel_level)

    def _apply_object_wise_non_overlapping_constraints(self, pred_masks, obj_scores, background_value=-10.0):
        """Let only one object (by score) claim each overlapping region."""
        pred_masks_single_score = torch.where(pred_masks > 0, obj_scores[..., None, None], background_value)
        pixel_level = self._apply_non_overlapping_constraints(pred_masks_single_score)
        return torch.where(pixel_level > 0, pred_masks, torch.clamp(pred_masks, max=background_value))
