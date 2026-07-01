# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import pickle
import re
import types
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.modules import (
    AIFI,
    C1,
    C2,
    C2PSA,
    C3,
    C3TR,
    ELAN1,
    OBB,
    OBB26,
    PSA,
    SPP,
    SPPELAN,
    SPPF,
    A2C2f,
    AConv,
    ADown,
    BackboneMemoryBank,
    BboxMaskRenderer,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C2fCIB,
    C2fPSA,
    C3Ghost,
    C3k2,
    C3x,
    CBFuse,
    CBLinear,
    AnomalyMCDetect,
    Classify,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    Detect,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostBottleneck,
    GhostConv,
    HGBlock,
    HGStem,
    HeatmapBiasFusion,
    MaskPriorAugmenter,
    ImagePoolingAttn,
    Index,
    LRPCHead,
    Pose,
    Pose26,
    RepC3,
    RepConv,
    RepNCSPELAN4,
    RepVGGDW,
    ResNetLayer,
    RTDETRDecoder,
    SCDown,
    Segment,
    Segment26,
    TorchVision,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    YOLOESegment26,
    v10Detect,
)
from ultralytics.utils import DEFAULT_CFG_DICT, LOGGER, WINDOWS, YAML, colorstr, emojis
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import (
    AnomalyMCLoss,
    E2ELoss,
    PoseLoss26,
    v8ClassificationLoss,
    v8DetectionLoss,
    v8OBBLoss,
    v8PoseLoss,
    v8SegmentationLoss,
)
from ultralytics.utils.ops import make_divisible
from ultralytics.utils.patches import torch_load
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn,
    fuse_deconv_and_bn,
    initialize_weights,
    intersect_dicts,
    model_info,
    scale_img,
    smart_inference_mode,
    time_sync,
)


class BaseModel(torch.nn.Module):
    """Base class for all YOLO models in the Ultralytics family.

    This class provides common functionality for YOLO models including forward pass handling, model fusion, information
    display, and weight loading capabilities.

    Attributes:
        model (torch.nn.Sequential): The neural network model.
        save (list): List of layer indices to save outputs from.
        stride (torch.Tensor): Model stride values.

    Methods:
        forward: Perform forward pass for training or inference.
        predict: Perform inference on input tensor.
        fuse: Fuse Conv/BatchNorm layers and reparameterize for optimization.
        info: Print model information.
        load: Load weights into the model.
        loss: Compute loss for training.

    Examples:
        Create a BaseModel instance
        >>> model = BaseModel()
        >>> model.info()  # Display model information
    """

    def forward(self, x, *args, **kwargs):
        """Perform forward pass of the model for either training or inference.

        If x is a dict, calculates and returns the loss for training. Otherwise, returns predictions for inference.

        Args:
            x (torch.Tensor | dict): Input tensor for inference, or dict with image tensor and labels for training.
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.

        Returns:
            (torch.Tensor): Loss if x is a dict (training), or network predictions (inference).
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False, embed=None):
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            augment (bool): Augment image during prediction.
            embed (list, optional): A list of layer indices to return embeddings from.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize, embed)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool): Print the computation time of each layer if True.
            visualize (bool): Save the feature maps of the model if True.
            embed (list, optional): A list of layer indices to return embeddings from.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f"{self.__class__.__name__} does not support 'augment=True' prediction. "
            f"Reverting to single-scale prediction."
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """Profile the computation time and FLOPs of a single layer of the model on a given input.

        Args:
            m (torch.nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.
        """
        try:
            import thop
        except ImportError:
            thop = None  # conda support without 'ultralytics-thop' installed

        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1e9 * 2 if thop else 0  # GFLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """Fuse Conv/ConvTranspose and BatchNorm layers, and reparameterize RepConv/RepVGGDW for improved efficiency.

        Args:
            verbose (bool): Whether to print model information after fusion.

        Returns:
            (torch.nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, "bn"):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, "bn"):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, "bn")  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepVGGDW):
                    m.fuse()
                    m.forward = m.forward_fuse
                if isinstance(m, Detect) and getattr(m, "end2end", False):
                    m.fuse()  # remove one2many head
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """Check if the model has less than a certain threshold of normalization layers.

        Args:
            thresh (int, optional): The threshold number of normalization layers.

        Returns:
            (bool): True if the number of normalization layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """Print model information.

        Args:
            detailed (bool): If True, prints out detailed information about the model.
            verbose (bool): If True, prints out the model information.
            imgsz (int): The size of the image used for computing model information.
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """Apply a function to all tensors in the model, including Detect head attributes like stride and anchors.

        Args:
            fn (function): The function to apply to the model.

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(
            m, Detect
        ):  # includes all Detect subclasses like Segment, Pose, OBB, WorldDetect, YOLOEDetect, YOLOESegment
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """Load weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress.
        """
        model = weights["model"] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        updated_csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(updated_csd, strict=False)  # load
        len_updated_csd = len(updated_csd)
        first_conv = "model.0.conv.weight"  # hard-coded to yolo models for now
        # mostly used to boost multi-channel training
        state_dict = self.state_dict()
        if first_conv not in updated_csd and first_conv in state_dict:
            c1, c2, h, w = state_dict[first_conv].shape
            cc1, cc2, ch, cw = csd[first_conv].shape
            if ch == h and cw == w:
                c1, c2 = min(c1, cc1), min(c2, cc2)
                state_dict[first_conv][:c1, :c2] = csd[first_conv][:c1, :c2]
                len_updated_csd += 1
        if verbose:
            LOGGER.info(f"Transferred {len_updated_csd}/{len(self.model.state_dict())} items from pretrained weights")

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"])
        return self.criterion(preds, batch)

    def init_criterion(self):
        """Initialize the loss criterion for the BaseModel."""
        raise NotImplementedError("compute_loss() needs to be implemented by task heads")


class DetectionModel(BaseModel):
    """YOLO detection model.

    This class implements the YOLO detection architecture, handling model initialization, forward pass, augmented
    inference, and loss computation for object detection tasks.

    Attributes:
        yaml (dict): Model configuration dictionary.
        model (torch.nn.Sequential): The neural network model.
        save (list): List of layer indices to save outputs from.
        names (dict): Class names dictionary.
        inplace (bool): Whether to use inplace operations.
        end2end (bool): Whether the model uses end-to-end detection.
        stride (torch.Tensor): Model stride values.

    Methods:
        __init__: Initialize the YOLO detection model.
        _predict_augment: Perform augmented inference.
        _descale_pred: De-scale predictions following augmented inference.
        _clip_augmented: Clip YOLO augmented inference tails.
        init_criterion: Initialize the loss criterion.

    Examples:
        Initialize a detection model
        >>> model = DetectionModel("yolo26n.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n.yaml", ch=3, nc=None, verbose=True):
        """Initialize the YOLO detection model with the given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict
        if self.yaml["backbone"][0][2] == "Silence":
            LOGGER.warning(
                "YOLOv9 `Silence` module is deprecated in favor of torch.nn.Identity. "
                "Please delete local *.pt file and re-download the latest model checkpoint."
            )
            self.yaml["backbone"][0][2] = "nn.Identity"

        # Define model
        self.yaml["channels"] = ch  # save channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.inplace = self.yaml.get("inplace", True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, YOLOEDetect, YOLOESegment
            s = 256  # 2x min stride
            m.inplace = self.inplace

            def _forward(x):
                """Perform a forward pass through the model, handling different Detect subclass types accordingly."""
                output = self.forward(x)
                if self.end2end:
                    output = output["one2many"]
                return output["feats"]

            self.model.eval()  # Avoid changing batch statistics until training begins
            m.training = True  # Setting it to True to properly return strides
            m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
            self.stride = m.stride
            self.model.train()  # Set model back to training(default) mode
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride, e.g., RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info("")

    @property
    def end2end(self):
        """Return whether the model uses end-to-end NMS-free detection."""
        return getattr(self.model[-1], "end2end", False)

    @end2end.setter
    def end2end(self, value):
        """Override the end-to-end detection mode."""
        self.set_head_attr(end2end=value)

    def set_head_attr(self, **kwargs):
        """Set attributes of the model head (last layer).

        Args:
            **kwargs (Any): Arbitrary keyword arguments representing attributes to set.
        """
        head = self.model[-1]
        for k, v in kwargs.items():
            if not hasattr(head, k):
                LOGGER.warning(f"Head has no attribute '{k}'.")
                continue
            setattr(head, k, v)

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            (tuple[torch.Tensor, None]): Augmented inference output and None for train output.
        """
        if getattr(self, "end2end", False) or self.__class__.__name__ != "DetectionModel":
            LOGGER.warning("Model does not support 'augment=True', reverting to single-scale prediction.")
            return self._predict_once(x)
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation).

        Args:
            p (torch.Tensor): Predictions tensor.
            flips (int | None): Flip type (None=none, 2=ud, 3=lr).
            scale (float): Scale factor.
            img_size (tuple): Original image size (height, width).
            dim (int): Dimension to split at.

        Returns:
            (torch.Tensor): De-scaled predictions.
        """
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails.

        Args:
            y (list[torch.Tensor]): List of detection tensors.

        Returns:
            (list[torch.Tensor]): Clipped detection tensors.
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return E2ELoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)


class YOLOAnomalyV2Model(DetectionModel):
    """YOLO Anomaly v2 — detection + soft-hint heatmap fusion (yoloa_v2_softhint branch).

    Extends DetectionModel with two anomaly-side modules attached OUTSIDE the parsed
    Sequential:
      - ``mask_renderer``: rasterizes GT bboxes into a 1-channel mask.
      - ``heatmap_bias_fusion``: HeatmapBiasFusion. Produces a bounded per-pixel bias
        added (broadcast over channels) to each PAN P3/P4/P5 feature before the Detect
        head. Both reg and cls branches see the bias; this is acceptable because the
        bias is bounded and additive, unlike the previous multiplicative amplifier.

    Mask dropout (anti-shortcut, see design.md §3.4):
      During training, with probability ``p_drop`` per sample, the bias for that
      sample is zeroed -> exact passthrough -> model is forced to also perform
      without a mask.

    Mask source:
      - Training: rendered from ``batch["bboxes"]`` (set by ``loss()``).
      - Validation B-on: caller sets bboxes via ``set_mask_input()``.
      - Validation B-off / pure inference: no bboxes -> bias is None -> PAN features
        flow through unchanged (vanilla YOLO).
      - External (e.g. user prompt): ``set_external_mask_once``.

    Spec: docs_yoloa_v2/specs/2026-06-02-softhint-fusion-design.md.
    """

    def __init__(self, cfg="yolo26-anomaly-v2.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        # Read v2-specific config from YAML. Constructor kwargs override.
        v2_cfg = self.yaml.get("anomaly_v2", {}) if isinstance(self.yaml, dict) else {}
        mask_size = int(v2_cfg.get("mask_size", 80))
        mask_mode = str(v2_cfg.get("mask_mode", "rect"))
        _sf = v2_cfg.get("sigma_factor", 0.25)
        sigma_factor = [float(_sf[0]), float(_sf[1])] if isinstance(_sf, (list, tuple)) else float(_sf)
        self.p_drop = float(v2_cfg.get("p_drop", 0.5))
        # Polygon-mask prior: disabled; bbox-rendered gauss prior is used.
        self.seg_target_polygon = False
        # Fusion mechanism: 'bias' (HeatmapBiasFusion, 1-channel additive), 'soft'
        # (HeatmapSoftFusion, temperature-softmax + BN → bias), 'film'
        # (HeatmapFiLMFusion, global residual grouped-FiLM), or 'queryfilm'
        # (QueryFiLMFusion, K learned queries). Exactly one is instantiated.
        fusion_mode = str(v2_cfg.get("fusion_mode", "bias")).lower()
        if fusion_mode != "bias":
            raise ValueError(f"fusion_mode must be bias, got {fusion_mode!r}")

        # AnomalyMCDetect (decoupled binary detection + multi-class type head):
        #   type_gain -- weight of the type cross-entropy in the loss (read by AnomalyMCLoss).
        #   type_tau  -- softmax temperature for the inference-time type distribution (set on head).
        self.type_gain = 0.5
        type_tau = 1.0

        detect = self.model[-1]
        if isinstance(detect, AnomalyMCDetect):
            detect.type_tau = type_tau
        if not isinstance(detect, Detect):
            raise TypeError(f"YOLOAnomalyV2Model expects last layer to be Detect, got {type(detect).__name__}")

        # Anomaly-side modules (live outside self.model so they are not in the Sequential)
        self.mask_renderer = BboxMaskRenderer(mask_size=mask_size, mode=mask_mode, sigma_factor=sigma_factor)
        self.heatmap_bias_fusion = HeatmapBiasFusion(num_scales=detect.nl)

        # Spatial resolution of the rendered/mask prior.
        self.mask_size = mask_size

        self.mask_augmenter = MaskPriorAugmenter(v2_cfg)
        # Inference-time prior processing: minmax stretch, gaussian/mean blur, spatial softmax,
        # and an edge-suppression window for memory-bank heatmaps.
        self.heatmap_norm = "none"
        self.heatmap_smooth_kernel = 5
        self.heatmap_edge_weight = False
        self.heatmap_edge_p = 4.0
        self.heatmap_edge_m = 4.4
        self.heatmap_edge_sigma = 1.0
        self._edge_weight_cache = None

        # Transient bbox input for the next forward pass. Reset after each forward.
        self._mask_bboxes_buf = None  # (N, 4) normalized [cx, cy, w, h]
        self._mask_batch_idx_buf = None  # (N,) long
        # Allows external callers (validator) to force mask-off mode for a single forward.
        self._mask_disabled_once = False

        # --- BackboneMemoryBank (v2.3) ---
        bb_layers_cfg = v2_cfg.get("bb_layers", None)
        bb_layers = list(bb_layers_cfg) if bb_layers_cfg else None
        self.memory_bank = (
            BackboneMemoryBank(
                temperature=3.0,
                K=5,
                max_bank_size=None,
                calibration_target_score=0.2,
                calibration_target_quantile=0.95,
                proj_dim=0,
                hmap_stretch_strength=0.0,
                holdout_max=5000,
            )
            if bb_layers
            else None
        )
        self._bb_layers = bb_layers
        self._bb_hook_handles: list = []
        self._bb_feats: dict[int, "torch.Tensor"] = {}
        if self.memory_bank is not None and self._bb_layers:
            self.memory_bank._bb_layer_indices = self._bb_layers
            self._install_backbone_taps(self._bb_layers)

        # Prior mode state (predictor/validator controlled)
        self._prior_mode: str | None = None  # None = legacy (GT bboxes -> renderer)
        self._last_heatmap: torch.Tensor | None = None

    def init_criterion(self):
        """Initialize the loss criterion.

        Returns ``AnomalyMCLoss`` when the head is an ``AnomalyMCDetect``,
        otherwise the standard detection criterion.
        """
        if isinstance(self.model[-1], AnomalyMCDetect):
            return E2ELoss(self, AnomalyMCLoss) if getattr(self, "end2end", False) else AnomalyMCLoss(self)
        return E2ELoss(self) if getattr(self, "end2end", False) else v8DetectionLoss(self)

    # ------------------------------------------------------------------
    # Mask input API
    # ------------------------------------------------------------------
    def set_mask_input(self, bboxes, batch_idx, masks=None):
        """Provide bboxes (and optionally polygon masks) used to render the prior for the NEXT forward pass.

        Cleared automatically after the forward. Typically called by:
          - ``loss()`` (auto, from batch dict — includes ``batch["masks"]`` when available)
          - the v2 validator (manual, for B-on val pass — bboxes only, no masks)
        """
        self._mask_bboxes_buf = bboxes
        self._mask_batch_idx_buf = batch_idx
        self._mask_masks_buf = masks
        self._mask_disabled_once = False

    def disable_mask_once(self):
        """Force the next forward to use no mask (bias=0 -> passthrough). Used by B-off val."""
        self._mask_bboxes_buf = None
        self._mask_batch_idx_buf = None
        self._external_mask_buf = None
        self._mask_disabled_once = True

    def set_external_mask_once(self, mask: torch.Tensor):
        """Provide a pre-computed mask (B, 1, H, W) for the next forward, bypassing the
        bbox renderer. The mask is consumed by ``heatmap_bias_fusion`` and added
        (broadcast over channels) to each PAN scale's feature.

        ``mask.shape[2:]`` may be any HxW; it is resized to each PAN scale internally.
        Typical use cases: hand-drawn masks for interactive prediction, MemoryBank
        outputs at v2.1 inference, sanity-check heatmaps.
        """
        if mask.dim() != 4 or mask.shape[1] != 1:
            raise ValueError(f"external mask must be (B, 1, H, W), got {tuple(mask.shape)}")
        self._external_mask_buf = mask
        self._mask_bboxes_buf = None
        self._mask_batch_idx_buf = None
        self._mask_disabled_once = False

    def _consume_mask_input(self):
        # Robust to being called during super().__init__()'s stride probe,
        # before our own __init__ has set these attributes.
        bb = getattr(self, "_mask_bboxes_buf", None)
        bi = getattr(self, "_mask_batch_idx_buf", None)
        ext = getattr(self, "_external_mask_buf", None)
        m = getattr(self, "_mask_masks_buf", None)
        disabled = getattr(self, "_mask_disabled_once", False)
        if hasattr(self, "_mask_bboxes_buf"):
            self._mask_bboxes_buf = None
            self._mask_batch_idx_buf = None
            self._external_mask_buf = None
            self._mask_masks_buf = None
            self._mask_disabled_once = False
        return bb, bi, ext, disabled, m

    # ------------------------------------------------------------------
    # Prior mode API (v2.3 — unified 4-mode routing)
    # ------------------------------------------------------------------
    def set_prior_mode(self, mode: str | None) -> None:
        """Set the prior source for the next forward.

        Args:
            mode: ``"none"`` (passthrough), ``"box"`` (GT bbox render), ``"mask"`` (external/GT
                mask), ``"segment"`` (segmentation prompt), ``"heatmap"`` (feature-side memory-bank
                map), or ``"anomaly_model"`` (external heatmap injected via mask seam). ``None``
                keeps the training default (pick by data).
        """
        if mode is not None and mode not in ("none", "box", "mask", "segment", "heatmap", "anomaly_model"):
            raise ValueError(f"Invalid prior_mode: {mode!r}")
        self._prior_mode = mode

    def load_support_set(
        self,
        source: str | Path | list[str],
        imgsz: int = 320,
        device=None,
        batch: int = 8,
        max_bank_size: int = 10000,
        max_images: int = 0,
        verbose: bool = True,
    ) -> int:
        """Build the BackboneMemoryBank from normal images for ``prior_mode=\"heatmap\"``.

        Iterates over images, runs the backbone to extract features, accumulates them
        into the memory bank, then compresses and freezes the bank.

        Args:
            source: Directory of normal images or list of image paths.
            imgsz: Resize images to this square size.
            device: Device for the bank (defaults to model device).
            batch: Mini-batch size for backbone feature extraction.
            max_bank_size: Maximum bank entries (coreset subsampling at freeze).
            verbose: Log progress.

        Returns:
            Final bank size (number of feature vectors).
        """
        import cv2
        from pathlib import Path

        from ultralytics.utils import LOGGER, TQDM

        mb = getattr(self, "memory_bank", None)
        if mb is None or self._bb_layers is None:
            raise RuntimeError("BackboneMemoryBank not configured. Add bb_layers to the anomaly_v2 YAML block.")
        mb.max_bank_size = max_bank_size
        mb.reset_memory_bank()
        self.set_prior_mode(None)  # Don't trigger prior routing during build

        # Resolve device
        if device is None:
            device = next(self.parameters()).device
        self.to(device)

        # Collect image paths
        if isinstance(source, (str, Path)):
            source = Path(source)
            if source.is_dir():
                exts = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"}
                paths = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in exts)
            else:
                paths = [str(source)]
        else:
            paths = list(source)

        if not paths:
            raise ValueError("No images found in source.")

        if max_images > 0 and len(paths) > max_images:
            paths = paths[:max_images]

        if verbose:
            LOGGER.info(f"Building memory bank from {len(paths)} images (imgsz={imgsz})...")

        pbar = TQDM(paths, desc="Building memory bank") if verbose else paths
        chunk = []
        n_ingested = 0  # track total images ingested for delayed temp display
        for p in pbar:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
            chunk.append(img)
            if len(chunk) >= batch:
                self._ingest_support_batch(chunk, device, mb)
                n_ingested += len(chunk)
                chunk.clear()
        if chunk:
            self._ingest_support_batch(chunk, device, mb)
            n_ingested += len(chunk)

        mb.freeze_memory_bank()
        final_size = mb.memory_bank.shape[0]
        if verbose:
            LOGGER.info(
                f"Memory bank frozen: {final_size} features, dim={mb.feature_dim}\n"
                f"  config: temp={mb.temperature:.4f}, K={mb.K}, "
                f"max_bank={mb.max_bank_size or 'unlimited'}, holdout_max={mb.holdout_max}, "
                f"bb_layers={self._bb_layers}"
            )
        return final_size

    def _ingest_support_batch(self, images, device, memory_bank):
        """Run a batch of images through the model and accumulate backbone features via hooks."""
        import numpy as np

        batch = np.stack(images, axis=0)  # (B, H, W, 3)
        batch = (torch.from_numpy(batch).permute(0, 3, 1, 2).float() / 255.0).contiguous()  # (B, 3, H, W)
        batch = batch.to(device)
        self._bb_feats = {}
        with torch.no_grad():
            self(batch)
        memory_bank.accumulate_features(self._bb_feats)

    # ------------------------------------------------------------------
    # Backbone taps (for BackboneMemoryBank)
    # ------------------------------------------------------------------
    def _install_backbone_taps(self, layer_indices: list[int]) -> None:
        """Register forward hooks that capture backbone-layer outputs into ``self._bb_feats``.

        Idempotent — removes old hooks before installing.
        """
        for h in getattr(self, "_bb_hook_handles", []):
            h.remove()
        self._bb_hook_handles: list = []
        self._bb_feats: dict[int, "torch.Tensor"] = {}

        def _make_hook(idx: int):
            def _hook(_module, _inp, out):
                # Detached capture on every forward: eval uses it for load_support_set /
                # prior_mode="heatmap". Detach keeps the graph out; cost is one small
                # held activation per tapped layer until the next forward clears the dict.
                self._bb_feats[idx] = out.detach()

            return _hook

        for idx in sorted(set(layer_indices)):
            if idx < len(self.model):
                handle = self.model[idx].register_forward_hook(_make_hook(idx))
                self._bb_hook_handles.append(handle)

    @torch.no_grad()
    def encode_bb_feats(self, img: torch.Tensor) -> dict[int, torch.Tensor]:
        """Run only the backbone prefix (layers 0..max(bb_layers)) to populate ``_bb_feats``.

        Used by the trainer to extract FIFO-queue bank features with the EMA (key) encoder at
        a fraction of a full forward's cost.

        Returns:
            Copy of the captured ``{layer_idx: (B, C, H, W)}`` dict (empty if no bb_layers).
        """
        if not getattr(self, "_bb_layers", None):
            return {}
        self._bb_feats = {}
        y, x = [], img
        for m in self.model[: max(self._bb_layers) + 1]:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
        return dict(self._bb_feats)

    def __getstate__(self):
        """Drop hooks + forward-time scratch buffers before pickle/deepcopy.

        Hook closures aren't picklable; the scratch buffers (captured backbone feats, last
        heatmap, aux/seg/mask staging) are transient — repopulated on the next forward — but if
        a forward ran just before ``torch.save`` they get pickled into the checkpoint and bloat
        it (e.g. ``_last_heatmap`` is a full batch at mask resolution).
        Null them in the returned state only (a shallow copy), so the live model is untouched.
        """
        state = super().__getstate__()
        for h in getattr(self, "_bb_hook_handles", []):
            h.remove()
        state["_bb_hook_handles"] = []
        state["_bb_feats"] = {}
        for k in (
            "_last_heatmap",
            "_mask_bboxes_buf",
            "_mask_batch_idx_buf",
            "_external_mask_buf",
            "_edge_weight_cache",
        ):
            if k in state:
                state[k] = None
        return state

    def __setstate__(self, state):
        """Re-install backbone hooks and set defaults for old-checkpoint compat."""
        super().__setstate__(state)
        self._bb_hook_handles = []
        self._bb_feats = {}
        if getattr(self, "_bb_layers", None) is not None:
            self._install_backbone_taps(self._bb_layers)
        # Backward compat: old checkpoints may lack v2.3+ attributes
        for attr, default in [
            ("memory_bank", None),
            ("_prior_mode", None),
            ("_last_heatmap", None),
            ("softmax_temperature", 1.0),
            ("heatmap_norm", "none"),
            ("heatmap_edge_weight", False),
            ("heatmap_edge_p", 4.0),
            ("heatmap_edge_m", 4.4),
            ("heatmap_edge_sigma", 1.0),
            ("_edge_weight_cache", None),
            ("fit_args", None),
            ("fit_data", None),
        ]:
            if not hasattr(self, attr):
                setattr(self, attr, default)
        # Old checkpoints may not contain the standalone augmenter module.
        if not hasattr(self, "mask_augmenter"):
            self.mask_augmenter = MaskPriorAugmenter({})

    def _resize_to_mask(self, x, mask_size):
        """Bilinearly resize a (B, 1, H, W) prior to (B, 1, mask_size, mask_size) when needed."""
        if x.shape[2] != mask_size or x.shape[3] != mask_size:
            x = torch.nn.functional.interpolate(x, size=(mask_size, mask_size), mode="bilinear", align_corners=False)
        return x

    def _prior_from_box(self, bboxes, batch_idx, batch_size, augment):
        """Prior rendered from GT bboxes (bbox->gauss), train-augmented when ``augment``, or None."""
        if bboxes is None:
            return None
        bb, bi = self.mask_augmenter.augment_prior_bboxes(bboxes, batch_idx, self.training)
        bb, bi = self.mask_augmenter.fragment_prior_bboxes(bb, bi, training=augment)
        prior = self.mask_renderer(bb, bi, batch_size)
        return self.mask_augmenter.augment_mask(prior) if augment else prior

    def _mask_union_prior(self, masks, batch_idx, batch_size):
        """Convert ``batch["masks"]`` to a ``(B, 1, mask_size, mask_size)`` binary union prior.

        Handles both overlap_mask formats:
          - ``overlap_mask=True``: ``(B, H/4, W/4)`` instance-index map (0=background)
          - ``overlap_mask=False``: ``(N, H/4, W/4)`` per-instance binary masks

        Polygon-equivalent augmentations are delegated to ``self.mask_augmenter``.
        """
        masks = self.mask_augmenter.drop_polygon_instances(masks, training=self.training, batch_size=batch_size)

        if masks.dim() == 3 and masks.shape[0] == batch_size:
            # overlap_mask=True: (B, H/r, W/r) instance-index map → binary union
            prior = (masks > 0).float().unsqueeze(1)  # (B, 1, H/r, W/r)
        else:
            # overlap_mask=False: (N, H/r, W/r) per-instance → union per image.
            # Instance drop: randomly zero individual mask rows.
            box_drop_p = self.mask_augmenter.mask_box_drop_p
            if self.training and box_drop_p > 0.0 and masks.numel() > 0:
                keep = torch.rand(masks.shape[0], device=masks.device) > box_drop_p
                masks = masks[keep]
                batch_idx = batch_idx[keep]
            prior = masks.new_zeros(batch_size, 1, masks.shape[-2], masks.shape[-1])
            for b in range(batch_size):
                sel = batch_idx == b
                if sel.any():
                    prior[b, 0] = masks[sel].amax(0)
            prior = (prior > 0).float()

        if prior.shape[2] != self.mask_size or prior.shape[3] != self.mask_size:
            prior = torch.nn.functional.interpolate(prior, size=(self.mask_size, self.mask_size), mode="nearest")

        prior = self.mask_augmenter.translate_prior(prior, training=self.training)
        prior = self.mask_augmenter.fragment_mask(prior, training=self.training)

        return prior

    def _resolve_prior(
        self,
        source,
        *,
        bboxes=None,
        batch_idx=None,
        batch_masks=None,
        external_mask=None,
        batch_size=None,
        device=None,
        augment=False,
    ):
        """Build the (B, 1, mask_size, mask_size) fusion prior from a single source, or None.

        The prior is always a mask-format hint fused into the PAN features; ``None`` means
        passthrough (no fusion -> head_a). ``augment=True`` (training only) applies
        ``mask_augmenter.augment_mask`` so the model learns to tolerate an imperfect deploy heatmap.

        Sources:
            none     -> None (explicit passthrough)
            box      -> GT bboxes rendered to a gauss mask (train default for bbox data)
            mask     -> a given mask tensor: ``external_mask`` if set, else the GT mask union
                        from ``batch_masks`` (train default for segment data)
            heatmap  -> feature-side memory-bank anomaly map
        """
        mask_size = getattr(self, "mask_size", 80)
        if source in (None, "none"):
            return None
        if source == "box":
            return self._prior_from_box(bboxes, batch_idx, batch_size, augment)
        if source == "mask":
            if external_mask is not None:
                return external_mask.to(device=device, dtype=torch.float32)
            if batch_masks is not None and batch_masks.numel() > 0:
                prior = self._mask_union_prior(batch_masks, batch_idx, batch_size).to(device=device)
                return self.mask_augmenter.augment_mask(prior) if augment else prior
            return None
        if source == "heatmap":
            return self._prior_from_heatmap(mask_size, device)
        return None

    def _prior_from_heatmap(self, mask_size, device):
        """Feature-side heatmap prior from the memory bank, or None."""
        mb = getattr(self, "memory_bank", None)
        bb_feats = getattr(self, "_bb_feats", None)
        if mb is None or not bb_feats:
            return None
        with torch.no_grad():
            hmap = mb(bb_feats)
        return self._resize_to_mask(hmap.to(device=device, dtype=torch.float32), mask_size)

    def _select_prior_source(self, external_mask, disabled, bboxes, batch_masks):
        """Pick the prior source for this forward.

        Precedence: disabled > ``_prior_mode`` (explicit deploy) > external > GT mask/box > none.
        """
        if disabled:
            return "none"
        pm = getattr(self, "_prior_mode", None)
        if pm is not None:
            return pm  # explicit deploy source (legacy aliases already translated by set_prior_mode)
        if external_mask is not None:
            return "mask"
        if bboxes is not None:
            use_polygon = (
                batch_masks is not None and batch_masks.numel() > 0 and getattr(self, "seg_target_polygon", False)
            )
            return "mask" if use_polygon else "box"
        return "none"

    def _smooth_prior(self, mask, mode, kernel):
        """Blur the prior heatmap, preserving its [0, 1] scale and spatial blob structure.

        ``mode='gaussian'`` uses a Gaussian kernel (sigma = kernel/6); ``'mean'`` uses a uniform box
        average. Both denoise a noisy memory-bank prior toward the smooth gauss masks the fusion
        conv trained on — unlike spatial softmax, which collapses scale + spatial extent.
        """
        import torch.nn.functional as F

        k = max(1, int(kernel)) | 1  # force odd so padding keeps H, W
        if k < 3:
            return mask
        if mode == "mean":
            ker = torch.ones(1, 1, k, k, device=mask.device, dtype=mask.dtype) / float(k * k)
        else:  # gaussian
            ax = torch.arange(k, device=mask.device, dtype=mask.dtype) - (k - 1) / 2.0
            g = torch.exp(-(ax**2) / (2.0 * (k / 6.0) ** 2))
            g = g / g.sum()
            ker = (g[:, None] * g[None, :])[None, None]
        return F.conv2d(mask, ker, padding=k // 2)

    def _edge_weight(self, mask):
        """Fixed squircle-Gaussian center-weight window matching ``mask``'s HxW (cached).

        ``weight = exp(-dist**m / (2*sigma**m))`` with ``dist = (|x|**p + |y|**p)**(1/p)`` over
        per-axis coords normalized to [0, 1] at the border: 1.0 at the center, decaying toward the
        edges. Defaults (p=4, m=4.4, sigma=1.0) give edge-mid ~0.61, corner ~0.34 (raise sigma -> gentler).
        Multiply into the heatmap to suppress memory-bank border noise. Returns (1, 1, H, W).
        """
        h, w = mask.shape[-2], mask.shape[-1]
        p = float(getattr(self, "heatmap_edge_p", 4.0))
        m = float(getattr(self, "heatmap_edge_m", 4.4))
        sigma = float(getattr(self, "heatmap_edge_sigma", 1.0))
        key = (h, w, p, m, sigma, mask.device, mask.dtype)
        cache = getattr(self, "_edge_weight_cache", None)
        if cache is None or cache[0] != key:
            yc = (torch.arange(h, device=mask.device, dtype=torch.float32) - (h - 1) / 2.0).abs() / max(
                (h - 1) / 2.0, 1e-6
            )
            xc = (torch.arange(w, device=mask.device, dtype=torch.float32) - (w - 1) / 2.0).abs() / max(
                (w - 1) / 2.0, 1e-6
            )
            dist = (xc[None, :] ** p + yc[:, None] ** p) ** (1.0 / p)
            wmap = torch.exp(-(dist**m) / (2.0 * sigma**m)).to(mask.dtype)
            cache = (key, wmap[None, None])
            self._edge_weight_cache = cache
        return cache[1]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def loss(self, batch, preds=None):
        """Compute loss. Sets mask input from ``batch["bboxes"]`` before forward."""
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            # Standard ultralytics batch fields: bboxes (N, 4), batch_idx (N,)
            self.set_mask_input(batch["bboxes"], batch["batch_idx"], batch.get("masks"))
            try:
                preds = self.forward(batch["img"])
            finally:
                # Forward always consumes them, but be defensive in case of exception.
                self._mask_bboxes_buf = None
                self._mask_batch_idx_buf = None
        det_loss, det_items = self.criterion(preds, batch)
        return det_loss, det_items

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """Forward with heatmap-guided fusion inserted before the Detect head."""
        batch_size = x.shape[0]
        device = x.device

        # Clear backbone feature cache so stale _ingest_support_batch / warmup
        # features don't interfere with prior_mode="heatmap".
        self._bb_feats = {}
        # AutoBackend's fuse() → .to() sequence can drop backbone hooks.
        # Re-install them defensively when any target layer has lost its hook.
        _layers = getattr(self, "_bb_layers", None)
        if _layers and hasattr(self, "_bb_hook_handles"):
            for _idx in _layers:
                if _idx < len(self.model) and len(self.model[_idx]._forward_hooks) == 0:
                    self._install_backbone_taps(_layers)
                    break

        # Consume the mask input set by loss() / set_mask_input() / disable_mask_once() /
        # set_external_mask_once().
        bboxes, batch_idx, external_mask, mask_disabled, batch_masks = self._consume_mask_input()

        # Per-sample keep mask for mask dropout (anti-shortcut). Only meaningful when a
        # rendered/blended mask is active; keep[b]=0 zeros the per-sample bias -> passthrough.
        # getattr guards the stride probe in super().__init__(), which runs before our attrs exist.
        p_drop = getattr(self, "p_drop", 0.0)
        keep = torch.ones(batch_size, device=device)
        if bboxes is not None and external_mask is None and self.training and p_drop > 0.0:
            keep = (torch.rand(batch_size, device=device) > p_drop).to(keep.dtype)

        # The fusion prior is resolved inside the loop once the PAN features are available.
        prior = None

        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        last = self.model[-1]
        for m in self.model:
            if m is last:
                # Apply soft-hint fusion to the PAN inputs before Detect.
                # m.f is a list of indices into y (the PAN P3/P4/P5 outputs).
                pan_inputs = [y[j] for j in m.f]

                # Resolve the fusion prior from a single source (picker + unified resolver).
                source = self._select_prior_source(external_mask, mask_disabled, bboxes, batch_masks)
                prior = self._resolve_prior(
                    source,
                    bboxes=bboxes,
                    batch_idx=batch_idx,
                    batch_masks=batch_masks,
                    external_mask=external_mask,
                    batch_size=batch_size,
                    device=device,
                    augment=self.training,
                )
                # Edge-suppression weight: down-weight the memory-bank heatmap toward the image
                # borders (peripheral patches score high from boundary effects, not real defects).
                if (
                    prior is not None
                    and source == "heatmap"
                    and getattr(self, "heatmap_edge_weight", False)
                ):
                    prior = prior * self._edge_weight(prior)
                # Stash RAW heatmap for validator AUROC (before softmax/augmentation).
                self._last_heatmap = prior.detach() if prior is not None else None
                # Per-image min-max normalization: stretch each sample's prior to [0, 1].
                # Boosts a soft, low-peak prior (memory bank ~0.8) so the fusion conv responds
                # like it did to binary GT masks. NOTE: on clean images with a flat prior this
                # amplifies noise to full range -> may induce false positives.
                _hn = getattr(self, "heatmap_norm", "none")
                if prior is not None and _hn == "minmax":
                    b = prior.shape[0]
                    flat = prior.reshape(b, -1)
                    lo = flat.min(dim=1, keepdim=True).values
                    hi = flat.max(dim=1, keepdim=True).values
                    prior = ((flat - lo) / (hi - lo).clamp_min(1e-6)).reshape_as(prior)
                elif prior is not None and _hn in ("gaussian", "mean"):
                    # Blur the prior: keeps [0,1] scale + blob structure (unlike softmax),
                    # denoising the MB heatmap toward the gauss masks the fusion trained on.
                    prior = self._smooth_prior(prior, _hn, getattr(self, "heatmap_smooth_kernel", 5))
                # NOTE: training-time prior augmentation (_augment_mask) is applied once inside
                # _resolve_prior (augment=self.training); do not re-apply it here.
                fused = []
                for i, p in enumerate(pan_inputs):
                    if prior is None:
                        # Pure passthrough: skip fusion entirely.
                        fused.append(p)
                        continue
                    # QueryFiLM (v0) modulates P3 (i == 0) only; P4/P5 pass through unchanged.
                    target_h, target_w = p.shape[2], p.shape[3]
                    if prior.shape[2] != target_h or prior.shape[3] != target_w:
                        m_scale = torch.nn.functional.interpolate(
                            prior, size=(target_h, target_w), mode="bilinear", align_corners=False
                        )
                    else:
                        m_scale = prior
                    delta = self.heatmap_bias_fusion(m_scale, i)
                    # Per-sample keep mask (mask dropout): dropped samples get zero increment.
                    delta = delta * keep.to(delta.dtype).view(-1, 1, 1, 1)
                    fused.append(p + delta)
                _supports_hm = hasattr(m, "_build_heatmap_gate")
                x = m(fused, heatmap=prior) if _supports_hm else m(fused)
            else:
                if m.f != -1:
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                if profile:
                    self._profile_one_layer(m, x, dt)
                x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x


class OBBModel(DetectionModel):
    """YOLO Oriented Bounding Box (OBB) model.

    This class extends DetectionModel to handle oriented bounding box detection tasks, providing specialized loss
    computation for rotated object detection.

    Methods:
        __init__: Initialize YOLO OBB model.
        init_criterion: Initialize the loss criterion for OBB detection.

    Examples:
        Initialize an OBB model
        >>> model = OBBModel("yolo26n-obb.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n-obb.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLO OBB model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the model."""
        return E2ELoss(self, v8OBBLoss) if getattr(self, "end2end", False) else v8OBBLoss(self)


class SegmentationModel(DetectionModel):
    """YOLO segmentation model.

    This class extends DetectionModel to handle instance segmentation tasks, providing specialized loss computation for
    pixel-level object detection and segmentation.

    Methods:
        __init__: Initialize YOLO segmentation model.
        init_criterion: Initialize the loss criterion for segmentation.

    Examples:
        Initialize a segmentation model
        >>> model = SegmentationModel("yolo26n-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize Ultralytics YOLO segmentation model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the SegmentationModel."""
        return E2ELoss(self, v8SegmentationLoss) if getattr(self, "end2end", False) else v8SegmentationLoss(self)


class PoseModel(DetectionModel):
    """YOLO pose model.

    This class extends DetectionModel to handle human pose estimation tasks, providing specialized loss computation for
    keypoint detection and pose estimation.

    Attributes:
        kpt_shape (tuple): Shape of keypoints data (num_keypoints, num_dimensions).

    Methods:
        __init__: Initialize YOLO pose model.
        init_criterion: Initialize the loss criterion for pose estimation.

    Examples:
        Initialize a pose model
        >>> model = PoseModel("yolo26n-pose.yaml", ch=3, nc=1, data_kpt_shape=(17, 3))
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n-pose.yaml", ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize Ultralytics YOLO Pose model.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            data_kpt_shape (tuple): Shape of keypoints data.
            verbose (bool): Whether to display model information.
        """
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg["kpt_shape"]):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg["kpt_shape"] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Initialize the loss criterion for the PoseModel."""
        return E2ELoss(self, PoseLoss26) if getattr(self, "end2end", False) else v8PoseLoss(self)


class ClassificationModel(BaseModel):
    """YOLO classification model.

    This class implements the YOLO classification architecture for image classification tasks, providing model
    initialization, configuration, and output reshaping capabilities.

    Attributes:
        yaml (dict): Model configuration dictionary.
        model (torch.nn.Sequential): The neural network model.
        stride (torch.Tensor): Model stride values.
        names (dict): Class names dictionary.

    Methods:
        __init__: Initialize ClassificationModel.
        _from_yaml: Set model configurations and define architecture.
        reshape_outputs: Update model to specified class count.
        init_criterion: Initialize the loss criterion.

    Examples:
        Initialize a classification model
        >>> model = ClassificationModel("yolo26n-cls.yaml", ch=3, nc=1000)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolo26n-cls.yaml", ch=3, nc=None, verbose=True):
        """Initialize ClassificationModel with YAML, channels, number of classes, verbose flag.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__()
        self._from_yaml(cfg, ch, nc, verbose)

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set Ultralytics YOLO model configurations and define the model architecture.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml["channels"] = self.yaml.get("channels", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override YAML value
        elif not nc and not self.yaml.get("nc", None):
            raise ValueError("nc not specified. Must specify nc in model.yaml or function arguments.")
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f"{i}" for i in range(self.yaml["nc"])}  # default names dict
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """Update a TorchVision classification model to class count 'nc' if required.

        Args:
            model (torch.nn.Module): Model to update.
            nc (int): New number of classes.
        """
        name, m = list((model.model if hasattr(model, "model") else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = torch.nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, torch.nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, torch.nn.Linear(m.in_features, nc))
        elif isinstance(m, torch.nn.Sequential):
            types = [type(x) for x in m]
            if torch.nn.Linear in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Linear)  # last torch.nn.Linear index
                if m[i].out_features != nc:
                    m[i] = torch.nn.Linear(m[i].in_features, nc)
            elif torch.nn.Conv2d in types:
                i = len(types) - 1 - types[::-1].index(torch.nn.Conv2d)  # last torch.nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = torch.nn.Conv2d(
                        m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None
                    )

    def init_criterion(self):
        """Initialize the loss criterion for the ClassificationModel."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):
    """RTDETR (Real-time DEtection and Tracking using Transformers) Detection Model class.

    This class is responsible for constructing the RTDETR architecture, defining loss functions, and facilitating both
    the training and inference processes. RTDETR is an object detection and tracking model that extends from the
    DetectionModel base class.

    Attributes:
        nc (int): Number of classes for detection.
        criterion (RTDETRDetectionLoss): Loss function for training.

    Methods:
        __init__: Initialize the RTDETRDetectionModel.
        init_criterion: Initialize the loss criterion.
        loss: Compute loss for training.
        predict: Perform forward pass through the model.

    Examples:
        Initialize an RTDETR model
        >>> model = RTDETRDetectionModel("rtdetr-l.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        """Initialize the RTDETRDetectionModel.

        Args:
            cfg (str | dict): Configuration file name or path.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Print additional information during initialization.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def _apply(self, fn):
        """Apply a function to all tensors in the model, including decoder anchors and valid mask.

        Args:
            fn (function): The function to apply to the model.

        Returns:
            (RTDETRDetectionModel): An updated RTDETRDetectionModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]
        m.anchors = fn(m.anchors)
        m.valid_mask = fn(m.valid_mask)
        return self

    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        """Compute the loss for the given batch of data.

        Args:
            batch (dict): Dictionary containing image and label data.
            preds (tuple, optional): Precomputed model predictions.

        Returns:
            (torch.Tensor): Total loss value.
            (torch.Tensor): Main three losses in a tensor.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        img = batch["img"]
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = img.shape[0]
        batch_idx = batch["batch_idx"]
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            "cls": batch["cls"].to(img.device, dtype=torch.long).view(-1),
            "bboxes": batch["bboxes"].to(device=img.device),
            "batch_idx": batch_idx.to(img.device, dtype=torch.long).view(-1),
            "gt_groups": gt_groups,
        }

        if preds is None:
            preds = self.predict(img, batch=targets)
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion(
            (dec_bboxes, dec_scores), targets, dn_bboxes=dn_bboxes, dn_scores=dn_scores, dn_meta=dn_meta
        )
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor(
            [loss[k].detach() for k in ["loss_giou", "loss_class", "loss_bbox"]], device=img.device
        )

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False, embed=None):
        """Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            batch (dict, optional): Ground truth data for evaluation.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of layer indices to return embeddings from.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x


class WorldModel(DetectionModel):
    """YOLOv8 World Model.

    This class implements the YOLOv8 World model for open-vocabulary object detection, supporting text-based class
    specification and CLIP model integration for zero-shot detection capabilities.

    Attributes:
        txt_feats (torch.Tensor): Text feature embeddings for classes.
        clip_model (torch.nn.Module): CLIP model for text encoding.

    Methods:
        __init__: Initialize YOLOv8 world model.
        set_classes: Set classes for offline inference.
        get_text_pe: Get text positional embeddings.
        predict: Perform forward pass with text features.
        loss: Compute loss with text features.

    Examples:
        Initialize a world model
        >>> model = WorldModel("yolov8s-world.yaml", ch=3, nc=80)
        >>> model.set_classes(["person", "car", "bicycle"])
        >>> results = model.predict(image_tensor)
    """

    def __init__(self, cfg="yolov8s-world.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 world model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        self.txt_feats = torch.randn(1, nc or 80, 512)  # features placeholder
        self.clip_model = None  # CLIP model placeholder
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def set_classes(self, text, batch=80, cache_clip_model=True):
        """Set classes in advance so that model could do offline-inference without clip model.

        Args:
            text (list[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.
        """
        self.txt_feats = self.get_text_pe(text, batch=batch, cache_clip_model=cache_clip_model)
        self.model[-1].nc = len(text)

    def get_text_pe(self, text, batch=80, cache_clip_model=True):
        """Get text positional embeddings using the CLIP model.

        Args:
            text (list[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.

        Returns:
            (torch.Tensor): Text positional embeddings.
        """
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # For backwards compatibility of models lacking clip_model attribute
            self.clip_model = build_text_model("clip:ViT-B/32", device=device)
        model = self.clip_model if cache_clip_model else build_text_model("clip:ViT-B/32", device=device)
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        return txt_feats.reshape(-1, len(text), txt_feats.shape[-1])

    def predict(self, x, profile=False, visualize=False, txt_feats=None, augment=False, embed=None):
        """Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            txt_feats (torch.Tensor, optional): The text features, use it if it's given.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of layer indices to return embeddings from.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        txt_feats = (self.txt_feats if txt_feats is None else txt_feats).to(device=x.device, dtype=x.dtype)
        if txt_feats.shape[0] != x.shape[0] or self.model[-1].export:
            txt_feats = txt_feats.expand(x.shape[0], -1, -1)
        ori_txt_feats = txt_feats.clone()
        y, dt, embeddings = [], [], []  # outputs
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, C2fAttn):
                x = m(x, txt_feats)
            elif isinstance(m, WorldDetect):
                x = m(x, ori_txt_feats)
            elif isinstance(m, ImagePoolingAttn):
                txt_feats = m(x, txt_feats)
            else:
                x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()

        if preds is None:
            preds = self.forward(batch["img"], txt_feats=batch["txt_feats"])
        return self.criterion(preds, batch)


class YOLOEModel(DetectionModel):
    """YOLOE detection model.

    This class implements the YOLOE architecture for efficient object detection with text and visual prompts, supporting
    both prompt-based and prompt-free inference modes.

    Attributes:
        pe (torch.Tensor): Prompt embeddings for classes.
        clip_model (torch.nn.Module): CLIP model for text encoding.

    Methods:
        __init__: Initialize YOLOE model.
        get_text_pe: Get text positional embeddings.
        get_visual_pe: Get visual embeddings.
        set_vocab: Set vocabulary for prompt-free model.
        get_vocab: Get fused vocabulary layer.
        set_classes: Set classes for offline inference.
        get_cls_pe: Get class positional embeddings.
        predict: Perform forward pass with prompts.
        loss: Compute loss with prompts.

    Examples:
        Initialize a YOLOE model
        >>> model = YOLOEModel("yoloe-v8s.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOE model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.text_model = self.yaml.get("text_model", "mobileclip:blt")

    @smart_inference_mode()
    def get_text_pe(self, text, batch=80, cache_clip_model=False, without_reprta=False):
        """Get text positional embeddings using the CLIP model.

        Args:
            text (list[str]): List of class names.
            batch (int): Batch size for processing text tokens.
            cache_clip_model (bool): Whether to cache the CLIP model.
            without_reprta (bool): Whether to return text embeddings without reprta module processing.

        Returns:
            (torch.Tensor): Text positional embeddings.
        """
        from ultralytics.nn.text_model import build_text_model

        device = next(self.model.parameters()).device
        if not getattr(self, "clip_model", None) and cache_clip_model:
            # For backwards compatibility of models lacking clip_model attribute
            self.clip_model = build_text_model(getattr(self, "text_model", "mobileclip:blt"), device=device)

        model = (
            self.clip_model
            if cache_clip_model
            else build_text_model(getattr(self, "text_model", "mobileclip:blt"), device=device)
        )
        text_token = model.tokenize(text)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(batch)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        txt_feats = txt_feats.reshape(-1, len(text), txt_feats.shape[-1])
        if without_reprta:
            return txt_feats

        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        return head.get_tpe(txt_feats)  # run auxiliary text head

    @smart_inference_mode()
    def get_visual_pe(self, img, visual):
        """Get visual positional embeddings.

        Args:
            img (torch.Tensor): Input image tensor.
            visual (torch.Tensor): Visual features.

        Returns:
            (torch.Tensor): Visual positional embeddings.
        """
        return self(img, vpe=visual, return_vpe=True)

    def set_vocab(self, vocab, names):
        """Set vocabulary for the prompt-free model.

        Args:
            vocab (nn.ModuleList): List of vocabulary items.
            names (list[str]): List of class names.
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)

        # Cache anchors for head
        device = next(self.parameters()).device
        self(torch.empty(1, 3, self.args["imgsz"], self.args["imgsz"]).to(device))  # warmup

        cv3 = getattr(head, "one2one_cv3", head.cv3)
        cv2 = getattr(head, "one2one_cv2", head.cv2)

        # re-parameterization for prompt-free model
        self.model[-1].lrpc = nn.ModuleList(
            LRPCHead(cls, pf[-1], loc[-1], enabled=i != 2) for i, (cls, pf, loc) in enumerate(zip(vocab, cv3, cv2))
        )
        for loc_head, cls_head in zip(head.cv2, head.cv3):
            assert isinstance(loc_head, nn.Sequential)
            assert isinstance(cls_head, nn.Sequential)
            del loc_head[-1]
            del cls_head[-1]
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_vocab(self, names):
        """Get fused vocabulary layer from the model.

        Args:
            names (list[str]): List of class names.

        Returns:
            (nn.ModuleList): List of vocabulary modules.
        """
        assert not self.training
        head = self.model[-1]
        assert isinstance(head, YOLOEDetect)
        assert not head.is_fused

        tpe = self.get_text_pe(names)
        self.set_classes(names, tpe)
        device = next(self.model.parameters()).device
        head.fuse(self.pe.to(device))  # fuse prompt embeddings to classify head

        cv3 = getattr(head, "one2one_cv3", head.cv3)
        vocab = nn.ModuleList()
        for cls_head in cv3:
            assert isinstance(cls_head, nn.Sequential)
            vocab.append(cls_head[-1])
        return vocab

    def set_classes(self, names, embeddings):
        """Set classes in advance so that model could do offline-inference without clip model.

        Args:
            names (list[str]): List of class names.
            embeddings (torch.Tensor): Embeddings tensor.
        """
        assert not hasattr(self.model[-1], "lrpc"), (
            "Prompt-free model does not support setting classes. Please try with Text/Visual prompt models."
        )
        assert embeddings.ndim == 3
        self.pe = embeddings
        self.model[-1].nc = len(names)
        self.names = check_class_names(names)

    def get_cls_pe(self, tpe, vpe):
        """Get class positional embeddings.

        Args:
            tpe (torch.Tensor | None): Text positional embeddings.
            vpe (torch.Tensor | None): Visual positional embeddings.

        Returns:
            (torch.Tensor): Class positional embeddings.
        """
        all_pe = []
        if tpe is not None:
            assert tpe.ndim == 3
            all_pe.append(tpe)
        if vpe is not None:
            assert vpe.ndim == 3
            all_pe.append(vpe)
        if not all_pe:
            all_pe.append(getattr(self, "pe", torch.zeros(1, 80, 512)))
        return torch.cat(all_pe, dim=1)

    def predict(
        self, x, profile=False, visualize=False, tpe=None, augment=False, embed=None, vpe=None, return_vpe=False
    ):
        """Perform a forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.
            profile (bool): If True, profile the computation time for each layer.
            visualize (bool): If True, save feature maps for visualization.
            tpe (torch.Tensor, optional): Text positional embeddings.
            augment (bool): If True, perform data augmentation during inference.
            embed (list, optional): A list of layer indices to return embeddings from.
            vpe (torch.Tensor, optional): Visual positional embeddings.
            return_vpe (bool): If True, return visual positional embeddings.

        Returns:
            (torch.Tensor): Model's output tensor.
        """
        y, dt, embeddings = [], [], []  # outputs
        b = x.shape[0]
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            if isinstance(m, YOLOEDetect):
                vpe = m.get_vpe(x, vpe) if vpe is not None else None
                if return_vpe:
                    assert vpe is not None
                    assert not self.training
                    return vpe
                cls_pe = self.get_cls_pe(m.get_tpe(tpe), vpe).to(device=x[0].device, dtype=x[0].dtype)
                if cls_pe.shape[0] != b or m.export:
                    cls_pe = cls_pe.expand(b, -1, -1)
                x.append(cls_pe)  # adding cls embedding
            x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPDetectLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = (
                (E2ELoss(self, TVPDetectLoss) if getattr(self, "end2end", False) else TVPDetectLoss(self))
                if visual_prompt
                else self.init_criterion()
            )
        if preds is None:
            preds = self.forward(
                batch["img"],
                tpe=None if "visuals" in batch else batch.get("txt_feats", None),
                vpe=batch.get("visuals", None),
            )
        return self.criterion(preds, batch)


class YOLOESegModel(YOLOEModel, SegmentationModel):
    """YOLOE segmentation model.

    This class extends YOLOEModel to handle instance segmentation tasks with text and visual prompts, providing
    specialized loss computation for pixel-level object detection and segmentation.

    Methods:
        __init__: Initialize YOLOE segmentation model.
        loss: Compute loss with prompts for segmentation.

    Examples:
        Initialize a YOLOE segmentation model
        >>> model = YOLOESegModel("yoloe-v8s-seg.yaml", ch=3, nc=80)
        >>> results = model.predict(image_tensor, tpe=text_embeddings)
    """

    def __init__(self, cfg="yoloe-v8s-seg.yaml", ch=3, nc=None, verbose=True):
        """Initialize YOLOE segmentation model with given config and parameters.

        Args:
            cfg (str | dict): Model configuration file path or dictionary.
            ch (int): Number of input channels.
            nc (int, optional): Number of classes.
            verbose (bool): Whether to display model information.
        """
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        if not hasattr(self, "criterion"):
            from ultralytics.utils.loss import TVPSegmentLoss

            visual_prompt = batch.get("visuals", None) is not None  # TODO
            self.criterion = (
                (E2ELoss(self, TVPSegmentLoss) if getattr(self, "end2end", False) else TVPSegmentLoss(self))
                if visual_prompt
                else self.init_criterion()
            )

        if preds is None:
            preds = self.forward(batch["img"], tpe=batch.get("txt_feats", None), vpe=batch.get("visuals", None))
        return self.criterion(preds, batch)


class Ensemble(torch.nn.ModuleList):
    """Ensemble of models.

    This class allows combining multiple YOLO models into an ensemble for improved performance through model averaging
    or other ensemble techniques.

    Methods:
        __init__: Initialize an ensemble of models.
        forward: Generate predictions from all models in the ensemble.

    Examples:
        Create an ensemble of models
        >>> ensemble = Ensemble()
        >>> ensemble.append(model1)
        >>> ensemble.append(model2)
        >>> results = ensemble(image_tensor)
    """

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Run ensemble forward pass and concatenate predictions from all models.

        Args:
            x (torch.Tensor): Input tensor.
            augment (bool): Whether to augment the input.
            profile (bool): Whether to profile the model.
            visualize (bool): Whether to visualize the features.

        Returns:
            (torch.Tensor): Concatenated predictions from all models.
            (None): Always None for ensemble inference.
        """
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C*num_models)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None, attributes=None):
    """Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code, where you've
    moved a module from one location to another, but you still want to support the old import paths for backwards
    compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.
        attributes (dict, optional): A dictionary mapping old module attributes to new module attributes.

    Examples:
        >>> with temporary_modules({"old.module": "new.module"}, {"old.module.attribute": "new.module.attribute"}):
        >>> import old.module  # this will now import new.module
        >>> from old.module import attribute  # this will now import new.module.attribute

    Notes:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if modules is None:
        modules = {}
    if attributes is None:
        attributes = {}
    import sys
    from importlib import import_module

    try:
        # Set attributes in sys.modules under their old name
        for old, new in attributes.items():
            old_module, old_attr = old.rsplit(".", 1)
            new_module, new_attr = new.rsplit(".", 1)
            setattr(import_module(old_module), old_attr, getattr(import_module(new_module), new_attr))

        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


class SafeClass:
    """A placeholder class to replace unknown classes during unpickling."""

    def __init__(self, *args, **kwargs):
        """Initialize SafeClass instance, ignoring all arguments."""
        pass

    def __call__(self, *args, **kwargs):
        """Run SafeClass instance, ignoring all arguments."""
        pass


class SafeUnpickler(pickle.Unpickler):
    """Custom Unpickler that replaces unknown classes with SafeClass."""

    def find_class(self, module, name):
        """Attempt to find a class, returning SafeClass if not among safe modules.

        Args:
            module (str): Module name.
            name (str): Class name.

        Returns:
            (type): Found class or SafeClass.
        """
        safe_modules = (
            "torch",
            "collections",
            "collections.abc",
            "builtins",
            "math",
            "numpy",
            # Add other modules considered safe
        )
        if module in safe_modules:
            return super().find_class(module, name)
        else:
            return SafeClass


def torch_safe_load(weight, safe_only=False):
    """Attempt to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised, it catches
    the error, logs a warning message, and attempts to install the missing module via the check_requirements()
    function. After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str | Path): The file path of the PyTorch model.
        safe_only (bool): If True, replace unknown classes with SafeClass during loading.

    Returns:
        (dict): The loaded model checkpoint.
        (str): The loaded filename.

    Examples:
        >>> from ultralytics.nn.tasks import torch_safe_load
        >>> ckpt, file = torch_safe_load("path/to/best.pt", safe_only=True)
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix=".pt")
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",  # YOLOv9e
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",  # YOLOv10
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",  # YOLOv10
                # resolve cross-platform pathlib pickle incompatibility
                **(
                    {"pathlib.PosixPath": "pathlib.WindowsPath"}
                    if WINDOWS
                    else {"pathlib.WindowsPath": "pathlib.PosixPath"}
                ),
            },
        ):
            if safe_only:
                # Load via custom pickle module
                safe_pickle = types.ModuleType("safe_pickle")
                safe_pickle.Unpickler = SafeUnpickler
                safe_pickle.load = lambda file_obj: SafeUnpickler(file_obj).load()
                with open(file, "rb") as f:
                    ckpt = torch_load(f, pickle_module=safe_pickle)
            else:
                ckpt = torch_load(file, map_location="cpu")

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == "models":
            raise TypeError(
                emojis(
                    f"ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained "
                    f"with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with "
                    f"YOLOv8 at https://github.com/ultralytics/ultralytics."
                    f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                    f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo26n.pt'"
                )
            ) from e
        elif e.name == "numpy._core":
            raise ModuleNotFoundError(
                emojis(
                    f"ERROR ❌️ {weight} requires numpy>=1.26.1, however numpy=={__import__('numpy').__version__} is installed."
                )
            ) from e
        LOGGER.warning(
            f"{weight} appears to require '{e.name}', which is not in Ultralytics requirements."
            f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
            f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
            f"run a command with an official Ultralytics model, i.e. 'yolo predict model=yolo26n.pt'"
        )
        check_requirements(e.name)  # install missing module
        ckpt = torch_load(file, map_location="cpu")

    if not isinstance(ckpt, dict):
        # File is likely a YOLO instance saved with i.e. torch.save(model, "saved_model.pt")
        LOGGER.warning(
            f"The file '{weight}' appears to be improperly saved or formatted. "
            f"For optimal results, use model.save('filename.pt') to correctly save YOLO models."
        )
        ckpt = {"model": ckpt.model}

    return ckpt, file


def load_checkpoint(weight, device=None, inplace=True, fuse=False):
    """Load single model weights.

    Args:
        weight (str | Path): Model weight path.
        device (torch.device, optional): Device to load model to.
        inplace (bool): Whether to do inplace operations.
        fuse (bool): Whether to fuse model.

    Returns:
        (torch.nn.Module): Loaded model.
        (dict): Model checkpoint dictionary.
    """
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
    model = (ckpt.get("ema") or ckpt["model"]).float()  # FP32 model

    # Model compatibility updates
    model.args = args  # attach args to model
    model.pt_path = str(weight)  # attach *.pt file path to model as string (avoids WindowsPath pickle issues)
    model.task = getattr(model, "task", guess_model_task(model))
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])

    model = (model.fuse() if fuse and hasattr(model, "fuse") else model).eval().to(device)  # model in eval mode

    # Module updates
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = inplace
        elif isinstance(m, torch.nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True):
    """Parse a YOLO model.yaml dictionary into a PyTorch model.

    Args:
        d (dict): Model dictionary.
        ch (int): Input channels.
        verbose (bool): Whether to print model details.

    Returns:
        (torch.nn.Sequential): PyTorch model.
        (list): Sorted list of layer indices whose outputs need to be saved.
    """
    import ast

    # Args
    legacy = True  # backward compatibility for v3/v5/v8/v9 models
    max_channels = float("inf")
    nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
    reg_max = d.get("reg_max", 16)
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = torch.nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
        }
    )
    repeat_modules = frozenset(  # modules with 'repeat' arguments
        {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fPSA,
            C2fCIB,
            C2PSA,
            A2C2f,
        }
    )
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 != nc (e.g., Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:  # set 1) embed channels and 2) num heads
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)  # number of repeats
                n = 1
            if m is C3k2:  # for M/L/X sizes
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":  # for L/X sizes
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in frozenset(
            {
                Detect,
                AnomalyMCDetect,
                WorldDetect,
                YOLOEDetect,
                Segment,
                Segment26,
                YOLOESegment,
                YOLOESegment26,
                Pose,
                Pose26,
                OBB,
                OBB26,
            }
        ):
            args.extend([reg_max, end2end, [ch[x] for x in f]])
            if m is Segment or m is YOLOESegment or m is Segment26 or m is YOLOESegment26:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {
                Detect,
                AnomalyMCDetect,
                YOLOEDetect,
                Segment,
                Segment26,
                YOLOESegment,
                YOLOESegment26,
                Pose,
                Pose26,
                OBB,
                OBB26,
            }:
                m.legacy = legacy
        elif m is v10Detect:
            args.append([ch[x] for x in f])
        elif m is ImagePoolingAttn:
            args.insert(1, [ch[x] for x in f])  # channels as second arg
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        m_.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    """Load a YOLO model from a YAML file.

    Args:
        path (str | Path): Path to the YAML file.

    Returns:
        (dict): Model dictionary.
    """
    path = Path(path)
    if path.stem in (f"yolov{d}{x}6" for x in "nsmlx" for d in (5, 8)):
        new_stem = re.sub(r"(\d+)([nslmx])6(.+)?$", r"\1\2-p6\3", path.stem)
        LOGGER.warning(f"Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.")
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r"(\d+)([nslmx])(.+)?$", r"\1\3", str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = YAML.load(yaml_file)  # model dict
    d["scale"] = guess_model_scale(path)
    d["yaml_file"] = str(path)
    return d


def guess_model_scale(model_path):
    """Extract the size character n, s, m, l, or x of the model's scale from the model path.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale (n, s, m, l, or x), or empty string if not found.
    """
    try:
        return re.search(r"yolo(e-)?[v]?\d+([nslmx])", Path(model_path).stem).group(2)
    except AttributeError:
        return ""


def guess_model_task(model):
    """Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (torch.nn.Module | dict | str | Path): PyTorch model, model configuration dict, or model file path.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose', 'obb').
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        # YOLO Anomaly v2 marks itself with a dedicated config block (see yolo26-anomaly-v2.yaml).
        if isinstance(cfg, dict) and "anomaly_v2" in cfg:
            return "anomaly_v2"
        m = cfg["head"][-1][-2].lower()  # output module name
        if m in {"classify", "classifier", "cls", "fc"}:
            return "classify"
        if "detect" in m:
            return "detect"
        if "segment" in m:
            return "segment"
        if "pose" in m:
            return "pose"
        if "obb" in m:
            return "obb"

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)
    # Guess from PyTorch model
    if isinstance(model, torch.nn.Module):  # PyTorch model
        # YOLOAnomalyV2Model carries v2-specific submodules; check class first so we don't
        # fall through to "detect" via the Detect-head module check below.
        if isinstance(model, YOLOAnomalyV2Model):
            return "anomaly_v2"
        for x in "model.args", "model.model.args", "model.model.model.args":
            with contextlib.suppress(Exception):
                return eval(x)["task"]  # nosec B307: safe eval of known attribute paths
        for x in "model.yaml", "model.model.yaml", "model.model.model.yaml":
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))  # nosec B307: safe eval of known attribute paths
        for m in model.modules():
            if isinstance(m, (Segment, YOLOESegment)):
                return "segment"
            elif isinstance(m, Classify):
                return "classify"
            elif isinstance(m, Pose):
                return "pose"
            elif isinstance(m, OBB):
                return "obb"
            elif isinstance(m, (Detect, WorldDetect, YOLOEDetect, v10Detect)):
                return "detect"

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if "-anomaly-v2" in model.stem or "anomaly_v2" in model.parts:
            return "anomaly_v2"
        elif "-seg" in model.stem or "segment" in model.parts:
            return "segment"
        elif "-cls" in model.stem or "classify" in model.parts:
            return "classify"
        elif "-pose" in model.stem or "pose" in model.parts:
            return "pose"
        elif "-obb" in model.stem or "obb" in model.parts:
            return "obb"
        elif "detect" in model.parts:
            return "detect"

    # Unable to determine task from model
    LOGGER.warning(
        "Unable to automatically guess model task, assuming 'task=detect'. "
        "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify','pose' or 'obb'."
    )
    return "detect"  # assume detect
