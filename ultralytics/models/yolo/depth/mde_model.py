# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
MDE (Monocular Depth Estimation) model for YOLO.

This module contains the complete MDE model that combines object detection with depth estimation.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules import C2f, Conv, Detect, SPPF
from ultralytics.nn.tasks import BaseModel
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, colorstr
from ultralytics.utils.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, scale_img, time_sync

from .mde_head import Detect_MDE


class MDE(BaseModel):
    """
    MDE (Monocular Depth Estimation) model for YOLO.
    
    This model extends YOLO to perform both object detection and depth estimation
    simultaneously. It uses the Detect_MDE head to predict bounding boxes, class
    probabilities, and depth values for each detected object.
    
    Attributes:
        model (nn.Module): The complete MDE model.
        save (list): List of layer indices to save during forward pass.
        names (list): List of class names.
        nc (int): Number of classes.
        
    Examples:
        Create an MDE model for KITTI dataset
        >>> model = MDE('yolov8n.yaml', nc=5)  # 5 classes for KITTI
        >>> model.train(data='kitti.yaml', epochs=100)
    """
    
    def __init__(self, cfg: str = "yolov8n.yaml", ch: int = 3, nc: int = None, verbose: bool = True):
        """
        Initialize MDE model.
        
        Args:
            cfg (str): Path to model configuration file.
            ch (int): Number of input channels.
            nc (int): Number of classes.
            verbose (bool): Whether to print model information.
        """
        self.yaml_file = Path(cfg).name
        self.yaml_cfg = self._load_cfg(cfg)
        
        # Override nc if provided
        if nc is not None:
            self.yaml_cfg['nc'] = nc
            
        # Initialize base model
        super().__init__(cfg, ch, nc, verbose)
        
        # Replace the detection head with MDE head
        self._replace_detect_head()
    
    def _load_cfg(self, cfg: str) -> dict:
        """Load model configuration from YAML file."""
        if isinstance(cfg, dict):
            return cfg
        elif isinstance(cfg, str):
            import yaml
            with open(cfg, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Invalid cfg type: {type(cfg)}")
    
    def _replace_detect_head(self):
        """Replace the standard Detect head with Detect_MDE head."""
        # Find and replace Detect layers with Detect_MDE
        for i, layer in enumerate(self.model):
            if isinstance(layer, Detect):
                # Get the number of classes and channels
                nc = getattr(layer, 'nc', 80)
                ch = [self.model[j].c2 for j in range(i-3, i)]  # Get channels from previous layers
                
                # Create MDE head
                mde_head = Detect_MDE(nc=nc, ch=ch)
                
                # Replace the layer
                self.model[i] = mde_head
                LOGGER.info(f"Replaced Detect layer {i} with Detect_MDE head")
    
    def forward(self, x: torch.Tensor, augment: bool = False, profile: bool = False) -> torch.Tensor:
        """
        Forward pass through the MDE model.
        
        Args:
            x (torch.Tensor): Input tensor.
            augment (bool): Whether to use test-time augmentation.
            profile (bool): Whether to profile the forward pass.
            
        Returns:
            torch.Tensor: Model predictions with depth.
        """
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile)  # single-scale inference, train
    
    def _forward_once(self, x: torch.Tensor, profile: bool = False, visualize: bool = False) -> torch.Tensor:
        """
        Single-scale forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.
            profile (bool): Whether to profile the forward pass.
            visualize (bool): Whether to visualize features.
            
        Returns:
            torch.Tensor: Model predictions.
        """
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x
    
    def _forward_augment(self, x: torch.Tensor) -> torch.Tensor:
        """Augmented inference."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train
    
    def _descale_pred(self, p: torch.Tensor, flips: int, scale: float, img_size: tuple) -> torch.Tensor:
        """De-scale predictions following augmented inference."""
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p = p.flip(3)  # de-flip ud
            elif flips == 3:
                p = p.flip(2)  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p
    
    def _clip_augmented(self, y: list) -> torch.Tensor:
        """Clip augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y
    
    def _profile_one_layer(self, m: nn.Module, x: torch.Tensor, dt: list):
        """Profile a single layer."""
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")
    
    def fuse(self):
        """Fuse Conv2d + BatchNorm2d layers throughout the model."""
        LOGGER.info("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self
    
    def info(self, detailed: bool = False, verbose: bool = True, imgsz: int = 640):
        """Print model information."""
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)
    
    def _apply(self, fn):
        """Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers."""
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect_MDE):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self


def parse_model(d: dict, ch: list, verbose: bool = True) -> nn.Sequential:
    """
    Parse a YOLO model.yaml dictionary into a PyTorch model.
    
    Args:
        d (dict): Model configuration dictionary.
        ch (list): List of input channels.
        verbose (bool): Whether to print model information.
        
    Returns:
        nn.Sequential: PyTorch model.
    """
    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act = d["anchors"], d["nc"], d["depth_multiple"], d["width_multiple"], d.get("activation")
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr(act+':')} activation function")

    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
            Conv,
            SPPF,
            C2f,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {C2f}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is Detect_MDE:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in {Detect, Segment, Pose, OBB, ImagePoolingAttn}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
        elif m is RTDETRDecoder:  # special case, channels are supplied in index 1
            args[1] = [ch[x] for x in f if x < len(ch)]
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, m.np  # attach index, 'from' index, type, number params
        if verbose:
            LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{m.np:10.0f}  {t:<40}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def make_divisible(x: int, divisor: int) -> int:
    """Make x divisible by divisor."""
    return math.ceil(x / divisor) * divisor
