# Ultralytics YOLO 🚀, AGPL-3.0 license
from .modules.sam2 import SAM2Base
from .modules.encoders import ImageEncoder, MemoryEncoder, Hiera, FpnNeck
from .modules.memory_attention import MemoryAttention, MemoryAttentionLayer


def _build_sam2(
    encoder_embed_dim=1280,
    encoder_stages=[2, 6, 36, 4],
    encoder_num_heads=16,
    encoder_global_att_blocks=[7, 15, 23, 31],
    endcoder_backbone_channel_list=[1152, 576, 288, 144],
    encoder_window_spatial_size=[7, 7],
    encoder_window_spec=[8, 4, 16, 8],
):
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
            backbone_channel_list=endcoder_backbone_channel_list,
            fpn_top_down_levels=[2, 3],
            fpn_interp_model="nearest",
        ),
        scalp=1,
    )
    memory_attention = MemoryAttention(
        d_model=256,
        pos_enc_at_input=True,
        num_layers=4,
        layer=MemoryAttentionLayer()
    )
    memory_encoder = MemoryEncoder(out_dim=64)
