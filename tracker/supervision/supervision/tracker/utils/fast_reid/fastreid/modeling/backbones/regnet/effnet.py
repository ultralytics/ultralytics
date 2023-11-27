# !/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""EfficientNet models."""

import logging

import torch
import torch.nn as nn

from supervision.tracker.utils.fast_reid.fastreid.layers import *
from supervision.tracker.utils.fast_reid.fastreid.modeling.backbones.build import BACKBONE_REGISTRY
from supervision.tracker.utils.fast_reid.fastreid.utils import comm
from supervision.tracker.utils.fast_reid.fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .config import cfg as effnet_cfg
from .regnet import drop_connect, init_weights

logger = logging.getLogger(__name__)
model_urls = {
    'b0': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/161305613/EN-B0_dds_8gpu.pyth',
    'b1': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/161304979/EN-B1_dds_8gpu.pyth',
    'b2': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/161305015/EN-B2_dds_8gpu.pyth',
    'b3': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/161304979/EN-B3_dds_8gpu.pyth',
    'b4': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/161305098/EN-B4_dds_8gpu.pyth',
    'b5': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/161304979/EN-B5_dds_8gpu.pyth',
    'b6': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/161304979/EN-B6_dds_8gpu.pyth',
    'b7': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/161304979/EN-B7_dds_8gpu.pyth',
}


class EffHead(nn.Module):
    """EfficientNet head: 1x1, BN, Swish, AvgPool, Dropout, FC."""

    def __init__(self, w_in, w_out, bn_norm):
        super(EffHead, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 1, stride=1, padding=0, bias=False)
        self.conv_bn = get_norm(bn_norm, w_out)
        self.conv_swish = Swish()

    def forward(self, x):
        x = self.conv_swish(self.conv_bn(self.conv(x)))
        return x


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)."""

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block w/ Swish: AvgPool, FC, Swish, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(
            nn.Conv2d(w_in, w_se, 1, bias=True),
            Swish(),
            nn.Conv2d(w_se, w_in, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


class MBConv(nn.Module):
    """Mobile inverted bottleneck block w/ SE (MBConv)."""

    def __init__(self, w_in, exp_r, kernel, stride, se_r, w_out, bn_norm):
        # expansion, 3x3 dwise, BN, Swish, SE, 1x1, BN, skip_connection
        super(MBConv, self).__init__()
        self.exp = None
        w_exp = int(w_in * exp_r)
        if w_exp != w_in:
            self.exp = nn.Conv2d(w_in, w_exp, 1, stride=1, padding=0, bias=False)
            self.exp_bn = get_norm(bn_norm, w_exp)
            self.exp_swish = Swish()
        dwise_args = {"groups": w_exp, "padding": (kernel - 1) // 2, "bias": False}
        self.dwise = nn.Conv2d(w_exp, w_exp, kernel, stride=stride, **dwise_args)
        self.dwise_bn = get_norm(bn_norm, w_exp)
        self.dwise_swish = Swish()
        self.se = SE(w_exp, int(w_in * se_r))
        self.lin_proj = nn.Conv2d(w_exp, w_out, 1, stride=1, padding=0, bias=False)
        self.lin_proj_bn = get_norm(bn_norm, w_out)
        # Skip connection if in and out shapes are the same (MN-V2 style)
        self.has_skip = stride == 1 and w_in == w_out

    def forward(self, x):
        f_x = x
        if self.exp:
            f_x = self.exp_swish(self.exp_bn(self.exp(f_x)))
        f_x = self.dwise_swish(self.dwise_bn(self.dwise(f_x)))
        f_x = self.se(f_x)
        f_x = self.lin_proj_bn(self.lin_proj(f_x))
        if self.has_skip:
            if self.training and effnet_cfg.EN.DC_RATIO > 0.0:
                f_x = drop_connect(f_x, effnet_cfg.EN.DC_RATIO)
            f_x = x + f_x
        return f_x


class EffStage(nn.Module):
    """EfficientNet stage."""

    def __init__(self, w_in, exp_r, kernel, stride, se_r, w_out, d, bn_norm):
        super(EffStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            name = "b{}".format(i + 1)
            self.add_module(name, MBConv(b_w_in, exp_r, kernel, b_stride, se_r, w_out, bn_norm))

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class StemIN(nn.Module):
    """EfficientNet stem for ImageNet: 3x3, BN, Swish."""

    def __init__(self, w_in, w_out, bn_norm):
        super(StemIN, self).__init__()
        self.conv = nn.Conv2d(w_in, w_out, 3, stride=2, padding=1, bias=False)
        self.bn = get_norm(bn_norm, w_out)
        self.swish = Swish()

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class EffNet(nn.Module):
    """EfficientNet model."""

    @staticmethod
    def get_args():
        return {
            "stem_w": effnet_cfg.EN.STEM_W,
            "ds": effnet_cfg.EN.DEPTHS,
            "ws": effnet_cfg.EN.WIDTHS,
            "exp_rs": effnet_cfg.EN.EXP_RATIOS,
            "se_r": effnet_cfg.EN.SE_R,
            "ss": effnet_cfg.EN.STRIDES,
            "ks": effnet_cfg.EN.KERNELS,
            "head_w": effnet_cfg.EN.HEAD_W,
        }

    def __init__(self, last_stride, bn_norm, **kwargs):
        super(EffNet, self).__init__()
        kwargs = self.get_args() if not kwargs else kwargs
        self._construct(**kwargs, last_stride=last_stride, bn_norm=bn_norm)
        self.apply(init_weights)

    def _construct(self, stem_w, ds, ws, exp_rs, se_r, ss, ks, head_w, last_stride, bn_norm):
        stage_params = list(zip(ds, ws, exp_rs, ss, ks))
        self.stem = StemIN(3, stem_w, bn_norm)
        prev_w = stem_w
        for i, (d, w, exp_r, stride, kernel) in enumerate(stage_params):
            name = "s{}".format(i + 1)
            if i == 5: stride = last_stride
            self.add_module(name, EffStage(prev_w, exp_r, kernel, stride, se_r, w, d, bn_norm))
            prev_w = w
        self.head = EffHead(prev_w, head_w, bn_norm)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


def init_pretrained_weights(key):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    filename = model_urls[key].split('/')[-1]

    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        if comm.is_main_process():
            gdown.download(model_urls[key], cached_file, quiet=False)

    comm.synchronize()

    logger.info(f"Loading pretrained model from {cached_file}")
    state_dict = torch.load(cached_file, map_location=torch.device("cpu"))["model_state"]

    return state_dict


@BACKBONE_REGISTRY.register()
def build_effnet_backbone(cfg):
    # fmt: off
    pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    last_stride   = cfg.MODEL.BACKBONE.LAST_STRIDE
    bn_norm       = cfg.MODEL.BACKBONE.NORM
    depth         = cfg.MODEL.BACKBONE.DEPTH
    # fmt: on

    cfg_files = {
        'b0': 'fastreid/modeling/backbones/regnet/effnet/EN-B0_dds_8gpu.yaml',
        'b1': 'fastreid/modeling/backbones/regnet/effnet/EN-B1_dds_8gpu.yaml',
        'b2': 'fastreid/modeling/backbones/regnet/effnet/EN-B2_dds_8gpu.yaml',
        'b3': 'fastreid/modeling/backbones/regnet/effnet/EN-B3_dds_8gpu.yaml',
        'b4': 'fastreid/modeling/backbones/regnet/effnet/EN-B4_dds_8gpu.yaml',
        'b5': 'fastreid/modeling/backbones/regnet/effnet/EN-B5_dds_8gpu.yaml',
    }[depth]

    effnet_cfg.merge_from_file(cfg_files)
    model = EffNet(last_stride, bn_norm)

    if pretrain:
        # Load pretrain path if specifically
        if pretrain_path:
            try:
                state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))["model_state"]
                logger.info(f"Loading pretrained model from {pretrain_path}")
            except FileNotFoundError as e:
                logger.info(f'{pretrain_path} is not found! Please check this path.')
                raise e
            except KeyError as e:
                logger.info("State dict keys error! Please check the state dict.")
                raise e
        else:
            key = depth
            state_dict = init_pretrained_weights(key)

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
