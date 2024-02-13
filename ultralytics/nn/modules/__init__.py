# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .block import (C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3, FusedMBConv, MBConv, SABottleneck, sa_layer, C3SA, LightC3x, C3xTR, C2HG, C3xHG, C2fx, C2TR, C3CTR, C2fDA, C3TR2, 
                    HarDBlock, MBC2f, C2fTA, C3xTA, LightC2f, LightBottleneck, BLightC2f, MSDAC3x, QC2f, 
                    LightDSConvC2f, AsymmetricLightBottleneck, AsymmetricBottleneck, AsymmetricLightC2f, 
                    AsymmetricLightBottleneckC2f, C3xAsymmetricLightBottleneck, adderBottleneck, adderC2f, ConvSelfAttention)
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention, CombConv, QConv, 
                   AsymmetricConv, AsymmetricDWConvLightConv, AsymmetricDWConv, adderConv)
from .head import Classify, Detect, Pose, RTDETRDecoder, Segment
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer, DualTransformerBlock)

__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP', 
           'SABottleneck', 'sa_layer', 'C3SA', 'LightC3x', 'C3xTR', 'C2HG', 'FusedMBConv', 'MBConv', 
           'C3xHG', 'C2fx', 'C2TR', 'C3CTR', 'C2fDA', 'C3TR2', 'HarDBlock', 'CombConv', 'MBC2f', 'C2fTA', 'C3xTA', 'LightC2f', 'LightBottleneck', 'BLightC2f', 'MSDAC3x', 'QConv','QC2f', 'LightDSConvC2f', 'AsymmetricLightBottleneck', 'AsymmetricBottleneck', 'AsymmetricLightC2f',
           'AsymmetricConv', 'AsymmetricDWConvLightConv', 'AsymmetricDWConv', 'AsymmetricLightBottleneckC2f', 'C3xAsymmetricLightBottleneck', 'adderBottleneck', 'adderC2f', 'adderConv')