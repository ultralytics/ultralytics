import logging
import math

import numpy as np
import torch
import torch.nn as nn

from supervision.tracker.utils.fast_reid.fastreid.layers import get_norm
from supervision.tracker.utils.fast_reid.fastreid.utils import comm
from supervision.tracker.utils.fast_reid.fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .config import cfg as regnet_cfg
from ..build import BACKBONE_REGISTRY

logger = logging.getLogger(__name__)
model_urls = {
    '800x': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160905981/RegNetX-200MF_dds_8gpu.pyth',
    '800y': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906567/RegNetY-800MF_dds_8gpu.pyth',
    '1600x': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth',
    '1600y': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906681/RegNetY-1.6GF_dds_8gpu.pyth',
    '3200x': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906139/RegNetX-3.2GF_dds_8gpu.pyth',
    '3200y': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906834/RegNetY-3.2GF_dds_8gpu.pyth',
    '4000x': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906383/RegNetX-4.0GF_dds_8gpu.pyth',
    '4000y': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906838/RegNetY-4.0GF_dds_8gpu.pyth',
    '6400x': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/161116590/RegNetX-6.4GF_dds_8gpu.pyth',
    '6400y': 'https://dl.fbaipublicfiles.com/pycls/dds_baselines/160907112/RegNetY-6.4GF_dds_8gpu.pyth',
}


def init_weights(m):
    """Performs ResNet-style weight initialization."""
    if isinstance(m, nn.Conv2d):
        # Note that there is no bias due to BN
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = (
                hasattr(m, "final_bn") and m.final_bn and regnet_cfg.BN.ZERO_INIT_FINAL_GAMMA
        )
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()


def get_stem_fun(stem_type):
    """Retrives the stem function by name."""
    stem_funs = {
        "res_stem_cifar": ResStemCifar,
        "res_stem_in": ResStemIN,
        "simple_stem_in": SimpleStemIN,
    }
    assert stem_type in stem_funs.keys(), "Stem type '{}' not supported".format(
        stem_type
    )
    return stem_funs[stem_type]


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {
        "vanilla_block": VanillaBlock,
        "res_basic_block": ResBasicBlock,
        "res_bottleneck_block": ResBottleneckBlock,
    }
    assert block_type in block_funs.keys(), "Block type '{}' not supported".format(
        block_type
    )
    return block_funs[block_type]


def drop_connect(x, drop_ratio):
    """Drop connect (adapted from DARTS)."""
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x

class AnyHead(nn.Module):
    """AnyNet head."""

    def __init__(self, w_in, nc):
        super(AnyHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(w_in, nc, bias=True)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class VanillaBlock(nn.Module):
    """Vanilla block: [3x3 conv, BN, Relu] x2"""

    def __init__(self, w_in, w_out, stride, bn_norm, bm=None, gw=None, se_r=None):
        assert (
                bm is None and gw is None and se_r is None
        ), "Vanilla block does not support bm, gw, and se_r options"
        super(VanillaBlock, self).__init__()
        self.construct(w_in, w_out, stride, bn_norm)

    def construct(self, w_in, w_out, stride, bn_norm):
        # 3x3, BN, ReLU
        self.a = nn.Conv2d(
            w_in, w_out, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.a_bn = get_norm(bn_norm, w_out)
        self.a_relu = nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE)
        # 3x3, BN, ReLU
        self.b = nn.Conv2d(w_out, w_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.b_bn = get_norm(bn_norm, w_out)
        self.b_relu = nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class BasicTransform(nn.Module):
    """Basic transformation: [3x3 conv, BN, Relu] x2"""

    def __init__(self, w_in, w_out, stride, bn_norm):
        super(BasicTransform, self).__init__()
        self.construct(w_in, w_out, stride, bn_norm)

    def construct(self, w_in, w_out, stride, bn_norm):
        # 3x3, BN, ReLU
        self.a = nn.Conv2d(
            w_in, w_out, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.a_bn = get_norm(bn_norm, w_out)
        self.a_relu = nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE)
        # 3x3, BN
        self.b = nn.Conv2d(w_out, w_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.b_bn = get_norm(bn_norm, w_out)
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBasicBlock(nn.Module):
    """Residual basic block: x + F(x), F = basic transform"""

    def __init__(self, w_in, w_out, stride, bn_norm, bm=None, gw=None, se_r=None):
        assert (
                bm is None and gw is None and se_r is None
        ), "Basic transform does not support bm, gw, and se_r options"
        super(ResBasicBlock, self).__init__()
        self.construct(w_in, w_out, stride, bn_norm)

    def _add_skip_proj(self, w_in, w_out, stride, bn_norm):
        self.proj = nn.Conv2d(
            w_in, w_out, kernel_size=1, stride=stride, padding=0, bias=False
        )
        self.bn = get_norm(bn_norm, w_out)

    def construct(self, w_in, w_out, stride, bn_norm):
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self._add_skip_proj(w_in, w_out, stride, bn_norm)
        self.f = BasicTransform(w_in, w_out, stride, bn_norm)
        self.relu = nn.ReLU(regnet_cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block"""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.construct(w_in, w_se)

    def construct(self, w_in, w_se):
        # AvgPool
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # FC, Activation, FC, Sigmoid
        self.f_ex = nn.Sequential(
            nn.Conv2d(w_in, w_se, kernel_size=1, bias=True),
            nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE),
            nn.Conv2d(w_se, w_in, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


class BottleneckTransform(nn.Module):
    """Bottlenect transformation: 1x1, 3x3, 1x1"""

    def __init__(self, w_in, w_out, stride, bn_norm, bm, gw, se_r):
        super(BottleneckTransform, self).__init__()
        self.construct(w_in, w_out, stride, bn_norm, bm, gw, se_r)

    def construct(self, w_in, w_out, stride, bn_norm, bm, gw, se_r):
        # Compute the bottleneck width
        w_b = int(round(w_out * bm))
        # Compute the number of groups
        num_gs = w_b // gw
        # 1x1, BN, ReLU
        self.a = nn.Conv2d(w_in, w_b, kernel_size=1, stride=1, padding=0, bias=False)
        self.a_bn = get_norm(bn_norm, w_b)
        self.a_relu = nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE)
        # 3x3, BN, ReLU
        self.b = nn.Conv2d(
            w_b, w_b, kernel_size=3, stride=stride, padding=1, groups=num_gs, bias=False
        )
        self.b_bn = get_norm(bn_norm, w_b)
        self.b_relu = nn.ReLU(inplace=regnet_cfg.MEM.RELU_INPLACE)
        # Squeeze-and-Excitation (SE)
        if se_r:
            w_se = int(round(w_in * se_r))
            self.se = SE(w_b, w_se)
        # 1x1, BN
        self.c = nn.Conv2d(w_b, w_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.c_bn = get_norm(bn_norm, w_out)
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform"""

    def __init__(self, w_in, w_out, stride, bn_norm, bm=1.0, gw=1, se_r=None):
        super(ResBottleneckBlock, self).__init__()
        self.construct(w_in, w_out, stride, bn_norm, bm, gw, se_r)

    def _add_skip_proj(self, w_in, w_out, stride, bn_norm):
        self.proj = nn.Conv2d(
            w_in, w_out, kernel_size=1, stride=stride, padding=0, bias=False
        )
        self.bn = get_norm(bn_norm, w_out)

    def construct(self, w_in, w_out, stride, bn_norm, bm, gw, se_r):
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self._add_skip_proj(w_in, w_out, stride, bn_norm)
        self.f = BottleneckTransform(w_in, w_out, stride, bn_norm, bm, gw, se_r)
        self.relu = nn.ReLU(regnet_cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class ResStemCifar(nn.Module):
    """ResNet stem for CIFAR."""

    def __init__(self, w_in, w_out, bn_norm):
        super(ResStemCifar, self).__init__()
        self.construct(w_in, w_out, bn_norm)

    def construct(self, w_in, w_out, bn_norm):
        # 3x3, BN, ReLU
        self.conv = nn.Conv2d(
            w_in, w_out, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = get_norm(bn_norm, w_out)
        self.relu = nn.ReLU(regnet_cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class ResStemIN(nn.Module):
    """ResNet stem for ImageNet."""

    def __init__(self, w_in, w_out, bn_norm):
        super(ResStemIN, self).__init__()
        self.construct(w_in, w_out, bn_norm)

    def construct(self, w_in, w_out, bn_norm):
        # 7x7, BN, ReLU, maxpool
        self.conv = nn.Conv2d(
            w_in, w_out, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn = get_norm(bn_norm, w_out)
        self.relu = nn.ReLU(regnet_cfg.MEM.RELU_INPLACE)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class SimpleStemIN(nn.Module):
    """Simple stem for ImageNet."""

    def __init__(self, in_w, out_w, bn_norm):
        super(SimpleStemIN, self).__init__()
        self.construct(in_w, out_w, bn_norm)

    def construct(self, in_w, out_w, bn_norm):
        # 3x3, BN, ReLU
        self.conv = nn.Conv2d(
            in_w, out_w, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn = get_norm(bn_norm, out_w)
        self.relu = nn.ReLU(regnet_cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x


class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, bn_norm, d, block_fun, bm, gw, se_r):
        super(AnyStage, self).__init__()
        self.construct(w_in, w_out, stride, bn_norm, d, block_fun, bm, gw, se_r)

    def construct(self, w_in, w_out, stride, bn_norm, d, block_fun, bm, gw, se_r):
        # Construct the blocks
        for i in range(d):
            # Stride and w_in apply to the first block of the stage
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            # Construct the block
            self.add_module(
                "b{}".format(i + 1), block_fun(b_w_in, w_out, b_stride, bn_norm, bm, gw, se_r)
            )

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class AnyNet(nn.Module):
    """AnyNet model."""

    def __init__(self, **kwargs):
        super(AnyNet, self).__init__()
        if kwargs:
            self.construct(
                stem_type=kwargs["stem_type"],
                stem_w=kwargs["stem_w"],
                block_type=kwargs["block_type"],
                ds=kwargs["ds"],
                ws=kwargs["ws"],
                ss=kwargs["ss"],
                bn_norm=kwargs["bn_norm"],
                bms=kwargs["bms"],
                gws=kwargs["gws"],
                se_r=kwargs["se_r"],
            )
        else:
            self.construct(
                stem_type=regnet_cfg.ANYNET.STEM_TYPE,
                stem_w=regnet_cfg.ANYNET.STEM_W,
                block_type=regnet_cfg.ANYNET.BLOCK_TYPE,
                ds=regnet_cfg.ANYNET.DEPTHS,
                ws=regnet_cfg.ANYNET.WIDTHS,
                ss=regnet_cfg.ANYNET.STRIDES,
                bn_norm=regnet_cfg.ANYNET.BN_NORM,
                bms=regnet_cfg.ANYNET.BOT_MULS,
                gws=regnet_cfg.ANYNET.GROUP_WS,
                se_r=regnet_cfg.ANYNET.SE_R if regnet_cfg.ANYNET.SE_ON else None,
            )
        self.apply(init_weights)

    def construct(self, stem_type, stem_w, block_type, ds, ws, ss, bn_norm, bms, gws, se_r):
        # Generate dummy bot muls and gs for models that do not use them
        bms = bms if bms else [1.0 for _d in ds]
        gws = gws if gws else [1 for _d in ds]
        # Group params by stage
        stage_params = list(zip(ds, ws, ss, bms, gws))
        # Construct the stem
        stem_fun = get_stem_fun(stem_type)
        self.stem = stem_fun(3, stem_w, bn_norm)
        # Construct the stages
        block_fun = get_block_fun(block_type)
        prev_w = stem_w
        for i, (d, w, s, bm, gw) in enumerate(stage_params):
            self.add_module(
                "s{}".format(i + 1), AnyStage(prev_w, w, s, bn_norm, d, block_fun, bm, gw, se_r)
            )
            prev_w = w
        # Construct the head
        self.in_planes = prev_w
        # self.head = AnyHead(w_in=prev_w, nc=nc)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws = w_0 * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont


class RegNet(AnyNet):
    """RegNet model."""

    def __init__(self, last_stride, bn_norm):
        # Generate RegNet ws per block
        b_ws, num_s, _, _ = generate_regnet(
            regnet_cfg.REGNET.WA, regnet_cfg.REGNET.W0, regnet_cfg.REGNET.WM, regnet_cfg.REGNET.DEPTH
        )
        # Convert to per stage format
        ws, ds = get_stages_from_blocks(b_ws, b_ws)
        # Generate group widths and bot muls
        gws = [regnet_cfg.REGNET.GROUP_W for _ in range(num_s)]
        bms = [regnet_cfg.REGNET.BOT_MUL for _ in range(num_s)]
        # Adjust the compatibility of ws and gws
        ws, gws = adjust_ws_gs_comp(ws, bms, gws)
        # Use the same stride for each stage
        ss = [regnet_cfg.REGNET.STRIDE for _ in range(num_s)]
        ss[-1] = last_stride
        # Use SE for RegNetY
        se_r = regnet_cfg.REGNET.SE_R if regnet_cfg.REGNET.SE_ON else None
        # Construct the model
        kwargs = {
            "stem_type": regnet_cfg.REGNET.STEM_TYPE,
            "stem_w": regnet_cfg.REGNET.STEM_W,
            "block_type": regnet_cfg.REGNET.BLOCK_TYPE,
            "ss": ss,
            "ds": ds,
            "ws": ws,
            "bn_norm": bn_norm,
            "bms": bms,
            "gws": gws,
            "se_r": se_r,
        }
        super(RegNet, self).__init__(**kwargs)


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
    state_dict = torch.load(cached_file, map_location=torch.device('cpu'))['model_state']

    return state_dict


@BACKBONE_REGISTRY.register()
def build_regnet_backbone(cfg):
    # fmt: off
    pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    last_stride   = cfg.MODEL.BACKBONE.LAST_STRIDE
    bn_norm       = cfg.MODEL.BACKBONE.NORM
    depth         = cfg.MODEL.BACKBONE.DEPTH
    # fmt: on

    cfg_files = {
        '200x': 'fastreid/modeling/backbones/regnet/regnetx/RegNetX-200MF_dds_8gpu.yaml',
        '200y': 'fastreid/modeling/backbones/regnet/regnety/RegNetY-200MF_dds_8gpu.yaml',
        '400x': 'fastreid/modeling/backbones/regnet/regnetx/RegNetX-400MF_dds_8gpu.yaml',
        '400y': 'fastreid/modeling/backbones/regnet/regnety/RegNetY-400MF_dds_8gpu.yaml',
        '800x': 'fastreid/modeling/backbones/regnet/regnetx/RegNetX-800MF_dds_8gpu.yaml',
        '800y': 'fastreid/modeling/backbones/regnet/regnety/RegNetY-800MF_dds_8gpu.yaml',
        '1600x': 'fastreid/modeling/backbones/regnet/regnetx/RegNetX-1.6GF_dds_8gpu.yaml',
        '1600y': 'fastreid/modeling/backbones/regnet/regnety/RegNetY-1.6GF_dds_8gpu.yaml',
        '3200x': 'fastreid/modeling/backbones/regnet/regnetx/RegNetX-3.2GF_dds_8gpu.yaml',
        '3200y': 'fastreid/modeling/backbones/regnet/regnety/RegNetY-3.2GF_dds_8gpu.yaml',
        '4000x': 'fastreid/modeling/backbones/regnet/regnety/RegNetX-4.0GF_dds_8gpu.yaml',
        '4000y': 'fastreid/modeling/backbones/regnet/regnety/RegNetY-4.0GF_dds_8gpu.yaml',
        '6400x': 'fastreid/modeling/backbones/regnet/regnetx/RegNetX-6.4GF_dds_8gpu.yaml',
        '6400y': 'fastreid/modeling/backbones/regnet/regnety/RegNetY-6.4GF_dds_8gpu.yaml',
    }[depth]

    regnet_cfg.merge_from_file(cfg_files)
    model = RegNet(last_stride, bn_norm)

    if pretrain:
        # Load pretrain path if specifically
        if pretrain_path:
            try:
                state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
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
