from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torch import nn, Tensor
from torch.nn import functional as F

#The style of importing Considers compatibility for the diversity of torchvision versions
try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    try:
        from torch.hub import load_state_dict_from_url
    except ImportError:
        from torch.utils.model_zoo import load_url as load_state_dict_from_url

from supervision.tracker.utils.fast_reid.fastreid.layers import get_norm
from .build import BACKBONE_REGISTRY
from .mobilenet import _make_divisible

# https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv3.py

model_urls = {
    "Large": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
    "Small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
}


def conv_1x1_bn(inp, oup, bn_norm):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        get_norm(bn_norm, oup),
        nn.ReLU6(inplace=True)
    )


class ConvBNActivation(nn.Sequential):
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            kernel_size: int = 3,
            stride: int = 1,
            groups: int = 1,
            bn_norm=None,
            activation_layer: Optional[Callable[..., nn.Module]] = None,
            dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            get_norm(bn_norm, out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes


class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input: Tensor, inplace: bool) -> Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return F.hardsigmoid(scale, inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input, True)
        return scale * input


class InvertedResidualConfig:
    def __init__(self, input_channels: int, kernel: int, expanded_channels: int, out_channels: int, use_se: bool,
                 activation: str, stride: int, dilation: int, width_mult: float):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    def __init__(self, cnf: InvertedResidualConfig, bn_norm,
                 se_layer: Callable[..., nn.Module] = SqueezeExcitation):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(ConvBNActivation(cnf.input_channels, cnf.expanded_channels, kernel_size=1,
                                           bn_norm=bn_norm, activation_layer=activation_layer))

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.expanded_channels, kernel_size=cnf.kernel,
                                       stride=stride, dilation=cnf.dilation, groups=cnf.expanded_channels,
                                       bn_norm=bn_norm, activation_layer=activation_layer))
        if cnf.use_se:
            layers.append(se_layer(cnf.expanded_channels))

        # project
        layers.append(ConvBNActivation(cnf.expanded_channels, cnf.out_channels, kernel_size=1, bn_norm=bn_norm,
                                       activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3(nn.Module):
    def __init__(
            self,
            bn_norm,
            inverted_residual_setting: List[InvertedResidualConfig],
            last_channel: int,
            block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        MobileNet V3 main class
        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, Sequence) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvBNActivation(3, firstconv_output_channels, kernel_size=3, stride=2, bn_norm=bn_norm,
                                       activation_layer=nn.Hardswish))

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, bn_norm))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(ConvBNActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                       bn_norm=bn_norm, activation_layer=nn.Hardswish))

        self.features = nn.Sequential(*layers)
        self.conv = conv_1x1_bn(lastconv_output_channels, last_channel, bn_norm)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.conv(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _mobilenet_v3_conf(arch: str, params: Dict[str, Any]):
    # non-public config parameters
    reduce_divider = 2 if params.pop('_reduced_tail', False) else 1
    dilation = 2 if params.pop('_dilated', False) else 1
    width_mult = params.pop('_width_mult', 1.0)

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "Large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif arch == "Small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError("Unsupported model type {}".format(arch))

    return inverted_residual_setting, last_channel


def _mobilenet_v3_model(
        bn_norm,
        depth: str,
        pretrained: bool,
        pretrain_path: str,
        **kwargs: Any
):
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(depth, kwargs)
    model = MobileNetV3(bn_norm, inverted_residual_setting, last_channel, **kwargs)
    if pretrained:
        if pretrain_path:
            state_dict = torch.load(pretrain_path)
        else:
            if model_urls.get(depth, None) is None:
                raise ValueError("No checkpoint is available for model type {}".format(depth))
            state_dict = load_state_dict_from_url(model_urls[depth], progress=True)
        model.load_state_dict(state_dict, strict=False)
    return model


@BACKBONE_REGISTRY.register()
def build_mobilenetv3_backbone(cfg):
    pretrain = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    bn_norm = cfg.MODEL.BACKBONE.NORM
    depth = cfg.MODEL.BACKBONE.DEPTH

    model = _mobilenet_v3_model(bn_norm, depth, pretrain, pretrain_path)

    return model
