# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import torch
import torch.nn as nn
from iopath.common.file_io import g_pathmgr
from ultralytics.models.sam.sam3.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
)
from ultralytics.models.sam.sam3.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from ultralytics.models.sam.sam3.geometry_encoders import SequenceGeometryEncoder
from ultralytics.models.sam.sam3.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from ultralytics.models.sam.sam3.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from ultralytics.models.sam.sam3.necks import Sam3DualViTDetNeck
from ultralytics.models.sam.sam3.sam3_image import Sam3Image
from ultralytics.models.sam.sam3.text_encoder_ve import VETextEncoder
from ultralytics.models.sam.sam3.tokenizer_ve import SimpleTokenizer
from ultralytics.models.sam.sam3.vitdet import ViT
from ultralytics.models.sam.sam3.vl_combiner import SAM3VLBackbone
from .modules.blocks import PositionEmbeddingSine
from ultralytics.utils.patches import torch_load


def _create_transformer_encoder() -> TransformerEncoderFusion:
    """Create transformer encoder with its layer."""
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
    )

    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    return encoder


def _create_transformer_decoder() -> TransformerDecoder:
    """Create transformer decoder with its layer."""
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )

    decoder = TransformerDecoder(
        layer=decoder_layer,
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
        resolution=1008,
        stride=14,
        use_act_checkpoint=True,
        presence_token=True,
    )
    return decoder


def _create_dot_product_scoring():
    """Create dot product scoring module."""
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)


def _create_segmentation_head(compile_mode=None):
    """Create segmentation head with pixel decoder."""
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
        compile_mode=compile_mode,
    )

    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dropout=0,
        embed_dim=256,
    )

    segmentation_head = UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=True,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )
    return segmentation_head


def _create_geometry_encoder():
    """Create geometry encoder with all its components."""
    # Create position encoding for geometry encoder
    geo_pos_enc = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
    )
    # Create geometry encoder layer
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )

    # Create geometry encoder
    input_geometry_encoder = SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )
    return input_geometry_encoder


def _create_sam3_model(
    backbone,
    transformer,
    input_geometry_encoder,
    segmentation_head,
    dot_prod_scoring,
    inst_interactive_predictor,
):
    """Create the SAM3 image model."""
    common_params = {
        "backbone": backbone,
        "transformer": transformer,
        "input_geometry_encoder": input_geometry_encoder,
        "segmentation_head": segmentation_head,
        "num_feature_levels": 1,
        "o2m_mask_predict": True,
        "dot_prod_scoring": dot_prod_scoring,
        "use_instance_query": False,
        "multimask_output": True,
        "inst_interactive_predictor": inst_interactive_predictor,
        "matcher": None,
    }

    model = Sam3Image(**common_params)
    return model


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
    vit_neck = Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )
    # Visual neck
    return vit_neck


def _create_sam3_transformer() -> TransformerWrapper:
    """Create SAM3 transformer encoder and decoder."""
    encoder: TransformerEncoderFusion = _create_transformer_encoder()
    decoder: TransformerDecoder = _create_transformer_decoder()

    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)


def _load_checkpoint(model, checkpoint):
    """Load model checkpoint from file."""
    with open(checkpoint, "rb") as f:
        ckpt = torch_load(f)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    sam3_image_ckpt = {k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k}
    if model.inst_interactive_predictor is not None:
        sam3_image_ckpt.update(
            {k.replace("tracker.", "inst_interactive_predictor.model."): v for k, v in ckpt.items() if "tracker" in k}
        )
    model.load_state_dict(sam3_image_ckpt, strict=False)
    return model


def build_sam3_image_model(
    checkpoint_path: str,
    bpe_path: str,
    enable_segmentation: bool = True,
    enable_inst_interactivity: bool = False,
    compile: bool = False,
):
    """
    Build SAM3 image model

    Args:
        checkpoint_path: Optional path to model checkpoint
        bpe_path: Path to the BPE tokenizer vocabulary
        enable_segmentation: Whether to enable segmentation head
        enable_inst_interactivity: Whether to enable instance interactivity (SAM 1 task)
        compile_mode: To enable compilation, set to "default"

    Returns:
        A SAM3 image model
    """
    # Create visual components
    compile_mode = "default" if compile else None
    vision_encoder = _create_vision_backbone(
        compile_mode=compile_mode, enable_inst_interactivity=enable_inst_interactivity
    )

    # Create text components
    text_encoder = VETextEncoder(
        tokenizer=SimpleTokenizer(bpe_path=bpe_path),
        d_model=256,
        width=1024,
        heads=16,
        layers=24,
    )

    # Create visual-language backbone
    backbone = SAM3VLBackbone(visual=vision_encoder, text=text_encoder, scalp=1)

    # Create transformer components
    transformer = _create_sam3_transformer()

    # Create dot product scoring
    dot_prod_scoring = _create_dot_product_scoring()

    # Create segmentation head if enabled
    segmentation_head = _create_segmentation_head(compile_mode=compile_mode) if enable_segmentation else None

    # Create geometry encoder
    input_geometry_encoder = _create_geometry_encoder()
    if enable_inst_interactivity:
        # sam3_pvs_base = build_tracker(apply_temporal_disambiguation=False)
        # inst_predictor = SAM3InteractiveImagePredictor(sam3_pvs_base)
        pass
    else:
        inst_predictor = None
    # Create the SAM3 model
    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
        inst_predictor,
    )

    # Load checkpoint if provided
    model = _load_checkpoint(model, checkpoint_path)

    # Setup device and mode
    model.eval()

    return model
