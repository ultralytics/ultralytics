import torch

from .modules.decoders import MaskDecoder
from .modules.encoders import PromptEncoder
from .modules.sam import Sam
from .modules.transformer import TwoWayTransformer
from .tiny_vit_sam import TinyViT


def setup_model(model_path):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
        image_encoder=TinyViT(img_size=1024,
                              in_chans=3,
                              num_classes=1000,
                              embed_dims=[64, 128, 160, 320],
                              depths=[2, 2, 6, 2],
                              num_heads=[2, 4, 5, 10],
                              window_sizes=[7, 7, 14, 7],
                              mlp_ratio=4.,
                              drop_rate=0.,
                              drop_path_rate=0.0,
                              use_checkpoint=False,
                              mbconv_expand_ratio=4.0,
                              local_conv_size=3,
                              layer_lr_decay=0.8),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    mobile_sam.load_state_dict(torch.load(model_path), strict=True)
    mobile_sam.eval()
    return mobile_sam
