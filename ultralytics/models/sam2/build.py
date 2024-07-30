# Ultralytics YOLO 🚀, AGPL-3.0 license
from .modules.sam2 import SAM2Base
from .modules.encoders import ImageEncoder, MemoryEncoder

def _build_sam2(
    encoder_embed_dim=1280,
    encoder_depth=32,
    encoder_num_heads=16,
    encoder_global_attn_indexes=[7, 15, 23, 31],
    endcoder_backbone_channel_list=[1152, 576, 288, 144],
    window_spec=[8, 4, 16, 8],
):
    pass
