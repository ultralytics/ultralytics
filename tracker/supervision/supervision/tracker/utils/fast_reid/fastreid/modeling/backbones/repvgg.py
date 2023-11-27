# encoding: utf-8
# ref: https://github.com/CaoWGG/RepVGG/blob/develop/repvgg.py


import logging

import numpy as np
import torch
import torch.nn as nn

from supervision.tracker.utils.fast_reid.fastreid.layers import *
from supervision.tracker.utils.fast_reid.fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import BACKBONE_REGISTRY

logger = logging.getLogger(__name__)


def deploy(self, mode=False):
    self.deploying = mode
    for module in self.children():
        if hasattr(module, 'deploying'):
            module.deploy(mode)


nn.Sequential.deploying = False
nn.Sequential.deploy = deploy


def conv_bn(norm_type, in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', get_norm(norm_type, out_channels))
    return result


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, norm_type, kernel_size,
                 stride=1, padding=0, groups=1):
        super(RepVGGBlock, self).__init__()
        self.deploying = False

        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        self.in_channels = in_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.register_parameter('fused_weight', None)
        self.register_parameter('fused_bias', None)

        self.rbr_identity = get_norm(norm_type, in_channels) if out_channels == in_channels and stride == 1 else None
        self.rbr_dense = conv_bn(norm_type, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, groups=groups)
        self.rbr_1x1 = conv_bn(norm_type, in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                               stride=stride, padding=padding_11, groups=groups)

    def forward(self, inputs):
        if self.deploying:
            assert self.fused_weight is not None and self.fused_bias is not None, \
                "Make deploy mode=True to generate fused weight and fused bias first"
            fused_out = self.nonlinearity(torch.nn.functional.conv2d(
                inputs, self.fused_weight, self.fused_bias, self.stride, self.padding, 1, self.groups))
            return fused_out

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        out = self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

        return out

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert branch.__class__.__name__.find('BatchNorm') != -1
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def deploy(self, mode=False):
        self.deploying = mode
        if mode:
            fused_weight, fused_bias = self.get_equivalent_kernel_bias()
            self.register_parameter('fused_weight', nn.Parameter(fused_weight))
            self.register_parameter('fused_bias', nn.Parameter(fused_bias))
            del self.rbr_identity, self.rbr_1x1, self.rbr_dense


class RepVGG(nn.Module):

    def __init__(self, last_stride, norm_type, num_blocks, width_multiplier=None, override_groups_map=None):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploying = False
        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, norm_type=norm_type,
                                  kernel_size=3, stride=2, padding=1)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), norm_type, num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), norm_type, num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), norm_type, num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), norm_type, num_blocks[3], stride=last_stride)

    def _make_stage(self, planes, norm_type, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, norm_type=norm_type,
                                      kernel_size=3, stride=stride, padding=1, groups=cur_groups))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def deploy(self, mode=False):
        self.deploying = mode
        for module in self.children():
            if hasattr(module, 'deploying'):
                module.deploy(mode)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


def create_RepVGG_A0(last_stride, norm_type):
    return RepVGG(last_stride, norm_type, num_blocks=[2, 4, 14, 1],
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None)


def create_RepVGG_A1(last_stride, norm_type):
    return RepVGG(last_stride, norm_type, num_blocks=[2, 4, 14, 1],
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None)


def create_RepVGG_A2(last_stride, norm_type):
    return RepVGG(last_stride, norm_type, num_blocks=[2, 4, 14, 1],
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None)


def create_RepVGG_B0(last_stride, norm_type):
    return RepVGG(last_stride, norm_type, num_blocks=[4, 6, 16, 1],
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None)


def create_RepVGG_B1(last_stride, norm_type):
    return RepVGG(last_stride, norm_type, num_blocks=[4, 6, 16, 1],
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None)


def create_RepVGG_B1g2(last_stride, norm_type):
    return RepVGG(last_stride, norm_type, num_blocks=[4, 6, 16, 1],
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map)


def create_RepVGG_B1g4(last_stride, norm_type):
    return RepVGG(last_stride, norm_type, num_blocks=[4, 6, 16, 1],
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map)


def create_RepVGG_B2(last_stride, norm_type):
    return RepVGG(last_stride, norm_type, num_blocks=[4, 6, 16, 1],
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None)


def create_RepVGG_B2g2(last_stride, norm_type):
    return RepVGG(last_stride, norm_type, num_blocks=[4, 6, 16, 1],
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map)


def create_RepVGG_B2g4(last_stride, norm_type):
    return RepVGG(last_stride, norm_type, num_blocks=[4, 6, 16, 1],
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map)


def create_RepVGG_B3(last_stride, norm_type):
    return RepVGG(last_stride, norm_type, num_blocks=[4, 6, 16, 1],
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None)


def create_RepVGG_B3g2(last_stride, norm_type):
    return RepVGG(last_stride, norm_type, num_blocks=[4, 6, 16, 1],
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map)


def create_RepVGG_B3g4(last_stride, norm_type):
    return RepVGG(last_stride, norm_type, num_blocks=[4, 6, 16, 1],
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map)


@BACKBONE_REGISTRY.register()
def build_repvgg_backbone(cfg):
    """
    Create a RepVGG instance from config.
    Returns:
        RepVGG: a :class: `RepVGG` instance.
    """

    # fmt: off
    pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    last_stride   = cfg.MODEL.BACKBONE.LAST_STRIDE
    bn_norm       = cfg.MODEL.BACKBONE.NORM
    depth         = cfg.MODEL.BACKBONE.DEPTH
    # fmt: on

    func_dict = {
        'A0': create_RepVGG_A0,
        'A1': create_RepVGG_A1,
        'A2': create_RepVGG_A2,
        'B0': create_RepVGG_B0,
        'B1': create_RepVGG_B1,
        'B1g2': create_RepVGG_B1g2,
        'B1g4': create_RepVGG_B1g4,
        'B2': create_RepVGG_B2,
        'B2g2': create_RepVGG_B2g2,
        'B2g4': create_RepVGG_B2g4,
        'B3': create_RepVGG_B3,
        'B3g2': create_RepVGG_B3g2,
        'B3g4': create_RepVGG_B3g4,
    }

    model = func_dict[depth](last_stride, bn_norm)

    if pretrain:
        try:
            state_dict = torch.load(pretrain_path, map_location=torch.device("cpu"))
            logger.info(f"Loading pretrained model from {pretrain_path}")
        except FileNotFoundError as e:
            logger.info(f'{pretrain_path} is not found! Please check this path.')
            raise e
        except KeyError as e:
            logger.info("State dict keys error! Please check the state dict.")
            raise e

        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

    return model
