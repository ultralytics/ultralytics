from __future__ import annotations

from .modules.sam import SAM2Model
from .modules.utils import get_1d_sine_pe, select_closest_cond_frames
from .modules.blocks import TwoWayTransformer
from .modules.decoders import SAM2MaskDecoder
import torch


class SAM3Model(SAM2Model):
    def __init__(
        self,
        image_encoder,
        memory_attention,
        memory_encoder,
        num_maskmem=7,
        image_size=1008,
        backbone_stride=14,
        sigmoid_scale_for_mem_enc=1,
        sigmoid_bias_for_mem_enc=0,
        binarize_mask_from_pts_for_mem_enc=False,
        use_mask_input_as_output_without_sam=False,
        max_cond_frames_in_attn=-1,
        directly_add_no_mem_embed=False,
        use_high_res_features_in_sam=False,
        multimask_output_in_sam=False,
        multimask_min_pt_num=1,
        multimask_max_pt_num=1,
        multimask_output_for_tracking=False,
        use_multimask_token_for_obj_ptr: bool = False,
        iou_prediction_use_sigmoid=False,
        memory_temporal_stride_for_eval=1,
        non_overlap_masks_for_mem_enc=False,
        use_obj_ptrs_in_encoder=False,
        max_obj_ptrs_in_encoder=16,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=False,
        use_signed_tpos_enc_to_obj_ptrs=False,
        only_obj_ptrs_in_the_past_for_eval=False,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        fixed_no_obj_ptr: bool = False,
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        no_obj_embed_spatial: bool = False,
        sam_mask_decoder_extra_args=None,
        compile_image_encoder: bool = False,
    ):
        super().__init__(
            image_encoder,
            memory_attention,
            memory_encoder,
            num_maskmem,
            image_size,
            backbone_stride,
            sigmoid_scale_for_mem_enc,
            sigmoid_bias_for_mem_enc,
            binarize_mask_from_pts_for_mem_enc,
            use_mask_input_as_output_without_sam,
            max_cond_frames_in_attn,
            directly_add_no_mem_embed,
            use_high_res_features_in_sam,
            multimask_output_in_sam,
            multimask_min_pt_num,
            multimask_max_pt_num,
            multimask_output_for_tracking,
            use_multimask_token_for_obj_ptr,
            iou_prediction_use_sigmoid,
            memory_temporal_stride_for_eval,
            non_overlap_masks_for_mem_enc,
            use_obj_ptrs_in_encoder,
            max_obj_ptrs_in_encoder,
            add_tpos_enc_to_obj_ptrs,
            proj_tpos_enc_in_obj_ptrs,
            use_signed_tpos_enc_to_obj_ptrs,
            only_obj_ptrs_in_the_past_for_eval,
            pred_obj_scores,
            pred_obj_scores_mlp,
            fixed_no_obj_ptr,
            soft_no_obj_ptr,
            use_mlp_for_obj_ptr_proj,
            no_obj_embed_spatial,
            sam_mask_decoder_extra_args,
            compile_image_encoder,
        )
        self.sam_mask_decoder = SAM2MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=self.sam_prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.sam_prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            use_high_res_features=self.use_high_res_features_in_sam,
            iou_prediction_use_sigmoid=self.iou_prediction_use_sigmoid,
            pred_obj_scores=self.pred_obj_scores,
            pred_obj_scores_mlp=self.pred_obj_scores_mlp,
            use_multimask_token_for_obj_ptr=self.use_multimask_token_for_obj_ptr,
            **(self.sam_mask_decoder_extra_args or {}),
        )

    def forward_image(self, img_batch: torch.Tensor):
        """Process image batch through encoder to extract multi-level features for SAM model."""
        backbone_out = self.image_encoder.forward_image(img_batch)["sam2_backbone_out"]
        if self.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(backbone_out["backbone_fpn"][0])
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(backbone_out["backbone_fpn"][1])
        return backbone_out

    def set_imgsz(self, imgsz: tuple[int, int]):
        """Set the image size for the model and mask downsampler."""
        super().set_imgsz(imgsz)
        self.memory_encoder.mask_downsampler.interpol_size = [size // 14 * 16 for size in imgsz]
