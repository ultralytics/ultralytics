# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.utils.downloads import attempt_download_asset

from .modules.encoders import FpnNeck, Hiera, ImageEncoder, MemoryEncoder
from .modules.memory_attention import MemoryAttention, MemoryAttentionLayer
from .modules.sam2 import SAM2Model


def build_sam2_t(checkpoint=None):
    """Build and return a Segment Anything Model (SAM2) tiny-size model with specified architecture parameters."""
    return _build_sam2(
        encoder_embed_dim=96,
        encoder_stages=[1, 2, 7, 2],
        encoder_num_heads=1,
        encoder_global_att_blocks=[5, 7, 9],
        encoder_window_spec=[8, 4, 14, 7],
        encoder_backbone_channel_list=[768, 384, 192, 96],
        checkpoint=checkpoint,
    )


def build_sam2_s(checkpoint=None):
    """Builds and returns a small-size Segment Anything Model (SAM2) with specified architecture parameters."""
    return _build_sam2(
        encoder_embed_dim=96,
        encoder_stages=[1, 2, 11, 2],
        encoder_num_heads=1,
        encoder_global_att_blocks=[7, 10, 13],
        encoder_window_spec=[8, 4, 14, 7],
        encoder_backbone_channel_list=[768, 384, 192, 96],
        checkpoint=checkpoint,
    )


def build_sam2_b(checkpoint=None):
    """Builds and returns a Segment Anything Model (SAM2) base-size model with specified architecture parameters."""
    return _build_sam2(
        encoder_embed_dim=112,
        encoder_stages=[2, 3, 16, 3],
        encoder_num_heads=2,
        encoder_global_att_blocks=[12, 16, 20],
        encoder_window_spec=[8, 4, 14, 7],
        encoder_window_spatial_size=[14, 14],
        encoder_backbone_channel_list=[896, 448, 224, 112],
        checkpoint=checkpoint,
    )


def build_sam2_l(checkpoint=None):
    """Build and return a Segment Anything Model (SAM2) large-size model with specified architecture parameters."""
    return _build_sam2(
        encoder_embed_dim=144,
        encoder_stages=[2, 6, 36, 4],
        encoder_num_heads=2,
        encoder_global_att_blocks=[23, 33, 43],
        encoder_window_spec=[8, 4, 16, 8],
        encoder_backbone_channel_list=[1152, 576, 288, 144],
        checkpoint=checkpoint,
    )


def _build_sam2(
    encoder_embed_dim=1280,
    encoder_stages=[2, 6, 36, 4],
    encoder_num_heads=2,
    encoder_global_att_blocks=[7, 15, 23, 31],
    encoder_backbone_channel_list=[1152, 576, 288, 144],
    encoder_window_spatial_size=[7, 7],
    encoder_window_spec=[8, 4, 16, 8],
    checkpoint=None,
):
    """Builds a SAM2 model with specified architecture parameters and optional checkpoint loading."""
    image_encoder = ImageEncoder(
        trunk=Hiera(
            embed_dim=encoder_embed_dim,
            num_heads=encoder_num_heads,
            stages=encoder_stages,
            global_att_blocks=encoder_global_att_blocks,
            window_pos_embed_bkg_spatial_size=encoder_window_spatial_size,
            window_spec=encoder_window_spec,
        ),
        neck=FpnNeck(
            d_model=256,
            backbone_channel_list=encoder_backbone_channel_list,
            fpn_top_down_levels=[2, 3],
            fpn_interp_model="nearest",
        ),
        scalp=1,
    )
    memory_attention = MemoryAttention(d_model=256, pos_enc_at_input=True, num_layers=4, layer=MemoryAttentionLayer())
    memory_encoder = MemoryEncoder(out_dim=64)

    sam2 = SAM2Model(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        num_maskmem=7,
        image_size=1024,
        sigmoid_scale_for_mem_enc=20.0,
        sigmoid_bias_for_mem_enc=-10.0,
        use_mask_input_as_output_without_sam=True,
        directly_add_no_mem_embed=True,
        use_high_res_features_in_sam=True,
        multimask_output_in_sam=True,
        iou_prediction_use_sigmoid=True,
        use_obj_ptrs_in_encoder=True,
        add_tpos_enc_to_obj_ptrs=True,
        only_obj_ptrs_in_the_past_for_eval=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,
        multimask_output_for_tracking=True,
        use_multimask_token_for_obj_ptr=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        use_mlp_for_obj_ptr_proj=True,
        compile_image_encoder=False,
        sam_mask_decoder_extra_args=dict(
            dynamic_multimask_via_stability=True,
            dynamic_multimask_stability_delta=0.05,
            dynamic_multimask_stability_thresh=0.98,
        ),
    )

    if checkpoint is not None:
        checkpoint = attempt_download_asset(checkpoint)
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)["model"]
        sam2.load_state_dict(state_dict)
    sam2.eval()
    return sam2


sam_model_map = {
    "sam2_t.pt": build_sam2_t,
    "sam2_s.pt": build_sam2_s,
    "sam2_b.pt": build_sam2_b,
    "sam2_l.pt": build_sam2_l,
}


def build_sam2(ckpt="sam_b.pt"):
    """Constructs a Segment Anything Model (SAM2) based on the specified checkpoint, with various size options."""
    model_builder = None
    ckpt = str(ckpt)  # to allow Path ckpt types
    for k in sam_model_map.keys():
        if ckpt.endswith(k):
            model_builder = sam_model_map.get(k)

    if not model_builder:
        raise FileNotFoundError(f"{ckpt} is not a supported SAM model. Available models are: \n {sam_model_map.keys()}")

    return model_builder(ckpt)
