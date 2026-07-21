# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import torch.nn as nn

from ultralytics.nn.modules.transformer import MLP
from ultralytics.utils.patches import torch_load

from .modules.blocks import PositionEmbeddingSine, RoPEAttention
from .modules.encoders import MemoryEncoder
from .modules.memory_attention import MemoryAttention, MemoryAttentionLayer
from .modules.sam import SAM3Model
from .sam3.decoder import TransformerDecoder, TransformerDecoderLayer
from .sam3.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from .sam3.geometry_encoders import SequenceGeometryEncoder
from .sam3.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from .sam3.model_misc import DotProductScoring, TransformerWrapper
from .sam3.necks import Sam3DualViTDetNeck
from .sam3.sam3_image import SAM3SemanticModel
from .sam3.text_encoder_ve import VETextEncoder
from .sam3.vitdet import ViT
from .sam3.vl_combiner import SAM3VLBackbone


def _create_vision_backbone(compile_mode=None, enable_inst_interactivity=True) -> Sam3DualViTDetNeck:
    """Create SAM3 visual backbone with ViT and neck."""
    # Position encoding
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
    )

    # ViT backbone
    vit_backbone = ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
    )
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )


def _create_sam3_transformer() -> TransformerWrapper:
    """Create SAM3 detector encoder and decoder."""
    encoder: TransformerEncoderFusion = TransformerEncoderFusion(
        layer=TransformerEncoderLayer(
            d_model=256,
            dim_feedforward=2048,
            dropout=0.1,
            pos_enc_at_attn=True,
            pos_enc_at_cross_attn_keys=False,
            pos_enc_at_cross_attn_queries=False,
            pre_norm=True,
            self_attention=nn.MultiheadAttention(
                num_heads=8,
                dropout=0.1,
                embed_dim=256,
                batch_first=True,
            ),
            cross_attention=nn.MultiheadAttention(
                num_heads=8,
                dropout=0.1,
                embed_dim=256,
                batch_first=True,
            ),
        ),
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    decoder: TransformerDecoder = TransformerDecoder(
        layer=TransformerDecoderLayer(
            d_model=256,
            dim_feedforward=2048,
            dropout=0.1,
            cross_attention=nn.MultiheadAttention(
                num_heads=8,
                dropout=0.1,
                embed_dim=256,
            ),
            n_heads=8,
            use_text_cross_attention=True,
        ),
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        use_act_checkpoint=True,
        presence_token=True,
    )

    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)


def _create_text_encoder(clip) -> VETextEncoder:
    """Create the CLIP-based text encoder shared by the SAM3 detector builders."""
    return VETextEncoder(
        tokenizer=clip.simple_tokenizer.SimpleTokenizer(),
        d_model=256,
        width=1024,
        heads=16,
        layers=24,
    )


def _create_dot_prod_scoring() -> DotProductScoring:
    """Create the dot-product scoring head shared by the SAM3 detector builders."""
    return DotProductScoring(
        d_model=256,
        d_proj=256,
        prompt_mlp=MLP(
            input_dim=256,
            hidden_dim=2048,
            output_dim=256,
            num_layers=2,
            residual=True,
            out_norm=nn.LayerNorm(256),
        ),
    )


def _create_segmentation_head(compile_mode=None) -> UniversalSegmentationHead:
    """Create the universal segmentation head shared by the SAM3 detector builders."""
    return UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=True,
        cross_attend_prompt=nn.MultiheadAttention(
            num_heads=8,
            dropout=0,
            embed_dim=256,
        ),
        pixel_decoder=PixelDecoder(
            num_upsampling_stages=3,
            interpolation_mode="nearest",
            hidden_dim=256,
            compile_mode=compile_mode,
        ),
    )


def _create_geometry_encoder() -> SequenceGeometryEncoder:
    """Create the geometry (box/point prompt) encoder shared by the SAM3 detector builders."""
    return SequenceGeometryEncoder(
        pos_enc=PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            scale=None,
            temperature=10000,
        ),
        encode_boxes_as_points=False,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=TransformerEncoderLayer(
            d_model=256,
            dim_feedforward=2048,
            dropout=0.1,
            pos_enc_at_attn=False,
            pre_norm=True,
            pos_enc_at_cross_attn_queries=False,
            pos_enc_at_cross_attn_keys=True,
        ),
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )


def build_sam3_image_model(checkpoint_path: str, enable_segmentation: bool = True, compile: bool = False):
    """Build SAM3 image model.

    Args:
        checkpoint_path: Optional path to model checkpoint
        enable_segmentation: Whether to enable segmentation head
        compile: Whether to enable compilation of the model

    Returns:
        A SAM3 image model
    """
    try:
        import clip
    except ImportError:
        from ultralytics.utils.checks import check_requirements

        check_requirements("git+https://github.com/ultralytics/CLIP.git")
        import clip
    # Create visual components
    compile_mode = "default" if compile else None
    vision_encoder = _create_vision_backbone(compile_mode=compile_mode, enable_inst_interactivity=True)

    # Create visual-language backbone
    backbone = SAM3VLBackbone(visual=vision_encoder, text=_create_text_encoder(clip), scalp=1)

    # Create the SAM3SemanticModel model
    model = SAM3SemanticModel(
        backbone=backbone,
        transformer=_create_sam3_transformer(),
        input_geometry_encoder=_create_geometry_encoder(),
        segmentation_head=_create_segmentation_head(compile_mode) if enable_segmentation else None,
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=_create_dot_prod_scoring(),
        use_instance_query=False,
        multimask_output=True,
    )

    # Load checkpoint
    model = _load_checkpoint(model, checkpoint_path)
    model.eval()
    return model


def build_interactive_sam3(checkpoint_path: str, compile=None, with_backbone=True) -> SAM3Model:
    """Build the SAM3 Tracker module for video tracking.

    Args:
        checkpoint_path (str): Path to model checkpoint.
        compile (str | None): Compilation mode for the vision backbone.
        with_backbone (bool): Whether to include the vision backbone in the model.

    Returns:
        (SAM3Model): A configured and initialized SAM3 model.
    """
    # Create model components
    memory_encoder = MemoryEncoder(out_dim=64, interpol_size=[1152, 1152])
    memory_attention = MemoryAttention(
        batch_first=True,
        d_model=256,
        pos_enc_at_input=True,
        layer=MemoryAttentionLayer(
            dim_feedforward=2048,
            dropout=0.1,
            pos_enc_at_attn=False,
            pos_enc_at_cross_attn_keys=True,
            pos_enc_at_cross_attn_queries=False,
            self_attn=RoPEAttention(
                embedding_dim=256,
                num_heads=1,
                downsample_rate=1,
                rope_theta=10000.0,
                feat_sizes=[72, 72],
            ),
            d_model=256,
            cross_attn=RoPEAttention(
                embedding_dim=256,
                num_heads=1,
                downsample_rate=1,
                kv_in_dim=64,
                rope_theta=10000.0,
                feat_sizes=[72, 72],
                rope_k_repeat=True,
            ),
        ),
        num_layers=4,
    )

    backbone = (
        SAM3VLBackbone(scalp=1, visual=_create_vision_backbone(compile_mode=compile), text=None)
        if with_backbone
        else None
    )
    model = SAM3Model(
        image_size=1008,
        image_encoder=backbone,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        backbone_stride=14,
        num_maskmem=7,
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
        no_obj_embed_spatial=True,
        proj_tpos_enc_in_obj_ptrs=True,
        use_signed_tpos_enc_to_obj_ptrs=True,
        sam_mask_decoder_extra_args=dict(
            dynamic_multimask_via_stability=True,
            dynamic_multimask_stability_delta=0.05,
            dynamic_multimask_stability_thresh=0.98,
        ),
    )

    # Load checkpoint if provided
    model = _load_checkpoint(model, checkpoint_path, interactive=True)

    # Setup device and mode
    model.eval()
    return model


def build_sam3_multiplex(checkpoint_path: str, multiplex_count: int = 16):
    """Build the SAM 3.1 Object Multiplex detector (tri-neck SAM3SemanticModel).

    The detector runs per-frame text/box grounding and its tri-head neck also produces the
    ``interactive`` and ``sam2_backbone_out`` feature pyramids consumed by the multiplex tracker
    (which is built without its own backbone). Checkpoint keys use the HF layout with
    ``detector.*`` / ``tracker.*`` prefixes.

    Args:
        checkpoint_path (str): Path to sam3.1_multiplex.pt.
        multiplex_count (int): Bucket capacity of the multiplex tracker.

    Returns:
        (SAM3SemanticModel): The multiplex detector with tri-head neck.
    """
    try:
        import clip
    except ImportError:
        from ultralytics.utils.checks import check_requirements

        check_requirements("git+https://github.com/ultralytics/CLIP.git")
        import clip

    from .sam3.necks import Sam3TriViTDetNeck
    from .sam3.vitdet import ViT
    from .sam3.vl_combiner import SAM3VLBackboneTri

    position_encoding = PositionEmbeddingSine(num_pos_feats=256, normalize=True, scale=None, temperature=10000)
    vit_backbone = ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
    )
    tri_neck = Sam3TriViTDetNeck(
        trunk=vit_backbone,
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0],
    )
    backbone = SAM3VLBackboneTri(visual=tri_neck, text=_create_text_encoder(clip), scalp=0)

    model = SAM3SemanticModel(
        backbone=backbone,
        transformer=_create_sam3_transformer(),
        input_geometry_encoder=_create_geometry_encoder(),
        segmentation_head=_create_segmentation_head(),
        num_feature_levels=1,
        o2m_mask_predict=True,
        dot_prod_scoring=_create_dot_prod_scoring(),
        use_instance_query=False,
        multimask_output=True,
        supervise_joint_box_scores=True,
    )

    with open(checkpoint_path, "rb") as f:
        ckpt = torch_load(f)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    detector_ckpt = {k[len("detector.") :]: v for k, v in ckpt.items() if k.startswith("detector.")}
    missing, unexpected = model.load_state_dict(detector_ckpt, strict=False)
    from ultralytics.utils import LOGGER

    LOGGER.info(
        f"SAM3 multiplex detector: loaded {len(detector_ckpt) - len(unexpected)} keys "
        f"({len(missing)} missing, {len(unexpected)} unexpected)"
    )
    model.eval()
    return model


def build_sam3_multiplex_tracker(checkpoint_path: str, multiplex_count: int = 16):
    """Build the SAM 3.1 Object Multiplex tracker net (backbone-less, fed by the detector).

    Mirrors Meta's build_sam3_multiplex_video_model configuration exactly; loads the
    ``tracker.model.*`` checkpoint keys.

    Args:
        checkpoint_path (str): Path to sam3.1_multiplex.pt.
        multiplex_count (int): Bucket capacity (16 for the released checkpoint).

    Returns:
        (SAM3MultiplexModel): The multiplex tracker in eval mode.
    """
    from .sam3.multiplex import (
        DecoupledTransformerDecoderLayerv2,
        MultiplexController,
        SimpleRoPEAttention,
        TransformerEncoderDecoupledCrossAttention,
    )
    from .sam3.multiplex_video import SAM3MultiplexModel, TrackerTransformerWrapper

    layer = DecoupledTransformerDecoderLayerv2(
        activation="gelu",
        d_model=256,
        num_heads=8,
        dropout=0.1,
        dim_feedforward=2048,
        pos_enc_at_attn=False,
        pre_norm=True,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        self_attention_rope=SimpleRoPEAttention(
            d_model=256, num_heads=8, dropout_p=0.1, rope_theta=10000.0, feat_sizes=[72, 72]
        ),
        cross_attention_rope=SimpleRoPEAttention(
            d_model=256, num_heads=8, dropout_p=0.1, rope_theta=10000.0, feat_sizes=[72, 72], rope_k_repeat=True
        ),
    )
    transformer = TrackerTransformerWrapper(
        encoder=TransformerEncoderDecoupledCrossAttention(
            d_model=256,
            pos_enc_at_input=True,
            use_image_in_output=False,
            layer=layer,
            num_layers=4,
            batch_first=True,
        ),
        d_model=256,
    )
    maskmem_backbone = MemoryEncoder(
        out_dim=256,
        in_dim=256,
        interpol_size=[1152, 1152],
        num_pos_feats=256,
        multiplex_count=multiplex_count,
        starting_out_chan=4,
        input_channel_multiplier=2,
    )
    model = SAM3MultiplexModel(
        backbone=None,  # detector's tri-neck features are shared via the frame feature cache
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        multiplex_controller=MultiplexController(multiplex_count, eval_multiplex_count=multiplex_count),
        image_size=1008,
        backbone_stride=14,
        num_maskmem=7,
        use_high_res_features_in_sam=True,
        use_obj_ptrs_in_encoder=True,
        max_obj_ptrs_in_encoder=16,
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=True,
        use_mlp_for_obj_ptr_proj=True,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,
        use_no_obj_ptr=True,
        use_linear_no_obj_ptr=True,
        no_obj_embed_spatial=True,
        sincos_tpos_enc=True,
        multimask_output_in_sam=True,
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        use_multimask_token_for_obj_ptr=True,
        num_multimask_outputs=3,
        apply_sigmoid_to_mask_logits_for_mem_enc=True,
        sigmoid_scale_for_mem_enc=2.0,
        sigmoid_bias_for_mem_enc=-1.0,
        non_overlap_masks_for_mem_enc=False,
        add_output_suppression_embeddings=True,
        add_object_conditional_embeddings=False,
        condition_as_mask_input=True,
        condition_as_mask_input_fg=1.0,
        condition_as_mask_input_bg=0.0,
        use_maskmem_tpos_v2=True,
        save_image_features=True,
        use_mask_input_as_output_without_sam=True,
        directly_add_no_mem_embed=True,
        iou_prediction_use_sigmoid=False,
        forward_backbone_per_frame_for_eval=True,
        max_cond_frames_in_attn=4,
        is_dynamic_model=True,
        sam_mask_decoder_extra_args=dict(
            dynamic_multimask_via_stability=True,
            dynamic_multimask_stability_delta=0.05,
            dynamic_multimask_stability_thresh=0.98,
        ),
    )

    with open(checkpoint_path, "rb") as f:
        ckpt = torch_load(f)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    tracker_ckpt = {k[len("tracker.model.") :]: v for k, v in ckpt.items() if k.startswith("tracker.model.")}
    missing, unexpected = model.load_state_dict(tracker_ckpt, strict=False)
    from ultralytics.utils import LOGGER

    LOGGER.info(
        f"SAM3 multiplex tracker: loaded {len(tracker_ckpt) - len(unexpected)} keys "
        f"({len(missing)} missing, {len(unexpected)} unexpected)"
    )
    if missing:
        LOGGER.warning(f"SAM3 multiplex tracker missing keys (first 5): {missing[:5]}")
    if unexpected:
        LOGGER.warning(f"SAM3 multiplex tracker unexpected keys (first 5): {unexpected[:5]}")
    model.eval()
    return model


def _load_checkpoint(model, checkpoint, interactive=False):
    """Load SAM3 model checkpoint from file."""
    with open(checkpoint, "rb") as f:
        ckpt = torch_load(f)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    sam3_image_ckpt = {k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k}
    if interactive:
        sam3_image_ckpt.update(
            {
                k.replace("backbone.vision_backbone", "image_encoder.vision_backbone"): v
                for k, v in sam3_image_ckpt.items()
                if "backbone.vision_backbone" in k
            }
        )
        sam3_image_ckpt.update(
            {
                k.replace("tracker.transformer.encoder", "memory_attention"): v
                for k, v in ckpt.items()
                if "tracker.transformer" in k
            }
        )
        sam3_image_ckpt.update(
            {
                k.replace("tracker.maskmem_backbone", "memory_encoder"): v
                for k, v in ckpt.items()
                if "tracker.maskmem_backbone" in k
            }
        )
        sam3_image_ckpt.update({k.replace("tracker.", ""): v for k, v in ckpt.items() if "tracker." in k})
    model.load_state_dict(sam3_image_ckpt, strict=False)
    return model
