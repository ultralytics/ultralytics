# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model head modules."""

from __future__ import annotations

import copy
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils import LOGGER, NOT_MACOS14
from ultralytics.utils.tal import dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import TORCH_1_11, fuse_conv_and_bn, smart_inference_mode

from .block import DFL, SAVPE, BNContrastiveHead, ContrastiveHead, Proto, Proto26, RealNVP, Residual, SwiGLUFFN
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = "OBB", "Classify", "Detect", "Pose", "RTDETRDecoder", "Segment", "YOLOEDetect", "YOLOESegment", "v10Detect"


class Detect(nn.Module):
    """YOLO Detect head for object detection models.

    This class implements the detection head used in YOLO models for predicting bounding boxes and class probabilities.
    It supports both training and inference modes, with optional end-to-end detection capabilities.

    Attributes:
        dynamic (bool): Force grid reconstruction.
        export (bool): Export mode flag.
        format (str): Export format.
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum detections per image.
        shape (tuple): Input shape.
        anchors (torch.Tensor): Anchor points.
        strides (torch.Tensor): Feature map strides.
        legacy (bool): Backward compatibility for v3/v5/v8/v9/v11 models.
        xyxy (bool): Output format, xyxy or xywh.
        nc (int): Number of classes.
        nl (int): Number of detection layers.
        reg_max (int): DFL channels.
        no (int): Number of outputs per anchor.
        stride (torch.Tensor): Strides computed during build.
        cv2 (nn.ModuleList): Convolution layers for box regression.
        cv3 (nn.ModuleList): Convolution layers for classification.
        dfl (nn.Module): Distribution Focal Loss layer.
        one2one_cv2 (nn.ModuleList): One-to-one convolution layers for box regression.
        one2one_cv3 (nn.ModuleList): One-to-one convolution layers for classification.

    Methods:
        forward: Perform forward pass and return predictions.
        bias_init: Initialize detection head biases.
        decode_bboxes: Decode bounding boxes from predictions.
        postprocess: Post-process model predictions.

    Examples:
        Create a detection head for 80 classes
        >>> detect = Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = detect(x)
    """

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    max_det = 300  # max_det
    agnostic_nms = False
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    xyxy = False  # xyxy or xywh output

    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO detection layer with specified number of classes and channels.

        Args:
            nc (int): Number of classes.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)

    @property
    def end2end(self):
        """Checks if the model has one2one for v3/v5/v8/v9/v11 backward compatibility."""
        return getattr(self, "_end2end", True) and hasattr(self, "one2one")

    @end2end.setter
    def end2end(self, value):
        """Override the end-to-end detection mode."""
        self._end2end = value

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]  # batch size
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

        Args:
            x (dict[str, torch.Tensor]): Dictionary of predictions from detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
        """
        # Inference path
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get decoded boxes based on anchors and strides."""
        shape = x["feats"][0].shape  # BCHW
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape

        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return dbox

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):  # from
            a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / self.stride[i]) ** 2
            )  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):  # from
                a[-1].bias.data[:] = 2.0  # box
                b[-1].bias.data[: self.nc] = math.log(
                    5 / self.nc / (640 / self.stride[i]) ** 2
                )  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """Decode bounding boxes from predictions."""
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not self.end2end and not self.xyxy,
            dim=1,
        )

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x1, y1, x2, y2, class_probs].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x1, y1, x2, y2, max_class_prob, class_index].
        """
        boxes, scores = preds.split([4, self.nc], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores, conf], dim=-1)

    def get_topk_index(self, scores: torch.Tensor, max_det: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get top-k indices from scores.

        Args:
            scores (torch.Tensor): Scores tensor with shape (batch_size, num_anchors, num_classes).
            max_det (int): Maximum detections per image.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): Top scores, class indices, and filtered indices.
        """
        batch_size, anchors, nc = scores.shape  # i.e. shape(16,8400,84)
        # Use max_det directly during export for TensorRT compatibility (requires k to be constant),
        # otherwise use min(max_det, anchors) for safety with small inputs during Python inference
        k = max_det if self.export else min(max_det, anchors)
        if self.agnostic_nms:
            scores, labels = scores.max(dim=-1, keepdim=True)
            scores, indices = scores.topk(k, dim=1)
            labels = labels.gather(1, indices)
            return scores, labels, indices
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(k)
        idx = ori_index[torch.arange(batch_size)[..., None], index // nc]  # original index
        return scores[..., None], (index % nc)[..., None].float(), idx

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = None


class Segment(Detect):
    """YOLO Segment head for segmentation models.

    This class extends the Detect head to include mask prediction capabilities for instance segmentation tasks.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv4 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a segmentation head
        >>> segment = Segment(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, mask_head=self.one2one_cv4)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with mask coefficients."""
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, mask_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and mask coefficients."""
        preds = super().forward_head(x, box_head, cls_head)
        if mask_head is not None:
            bs = x[0].shape[0]  # batch size
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nm) with last dimension
                format [x1, y1, x2, y2, class_probs, mask_coefficient].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + nm) and last
                dimension format [x1, y1, x2, y2, max_class_prob, class_index, mask_coefficient].
        """
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = None


class Segment26(Segment):
    """YOLO26 Segment head for segmentation models.

    This class extends the Segment head with Proto26 for mask prediction in instance segmentation tasks.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto26): Prototype generation module.
        cv4 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a segmentation head
        >>> segment = Segment26(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, nm, npr, reg_max, end2end, ch)
        self.proto = Proto26(ch, self.npr, self.nm, nc)  # protos

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = Detect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x)  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = (
                    tuple(p.detach() for p in proto) if isinstance(proto, tuple) else proto.detach()
                )
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def fuse(self) -> None:
        """Remove the one2many head and extra part of proto module for inference optimization."""
        super().fuse()
        if hasattr(self.proto, "fuse"):
            self.proto.fuse()


class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models.

    This class extends the Detect head to include oriented bounding box prediction with rotation angles.

    Attributes:
        ne (int): Number of extra parameters.
        cv4 (nn.ModuleList): Convolution layers for angle prediction.
        angle (torch.Tensor): Predicted rotation angles.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        decode_bboxes: Decode rotated bounding boxes.

    Examples:
        Create an OBB detection head
        >>> obb = OBB(nc=80, ne=1, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb(x)
    """

    def __init__(self, nc: int = 80, ne: int = 1, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`.

        Args:
            nc (int): Number of classes.
            ne (int): Number of extra parameters.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, angle_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, angle_head=self.one2one_cv4)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with rotation angles."""
        # For decode_bboxes convenience
        self.angle = x["angle"]  # TODO: need to test obb
        preds = super()._inference(x)
        return torch.cat([preds, x["angle"]], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, angle_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and angles."""
        preds = super().forward_head(x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]  # batch size
            angle = torch.cat(
                [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )  # OBB theta logits
            angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
            preds["angle"] = angle
        return preds

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + ne) with last dimension
                format [x, y, w, h, class_probs, angle].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 7) and last
                dimension format [x, y, w, h, max_class_prob, class_index, angle].
        """
        boxes, scores, angle = preds.split([4, self.nc, self.ne], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        angle = angle.gather(dim=1, index=idx.repeat(1, 1, self.ne))
        return torch.cat([boxes, scores, conf, angle], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = None


class OBB26(OBB):
    """YOLO26 OBB detection head for detection with rotation models. This class extends the OBB head with modified angle
    processing that outputs raw angle predictions without sigmoid transformation, compared to the original
    OBB class.

    Attributes:
        ne (int): Number of extra parameters.
        cv4 (nn.ModuleList): Convolution layers for angle prediction.
        angle (torch.Tensor): Predicted rotation angles.

    Methods:
        forward_head: Concatenate and return predicted bounding boxes, class probabilities, and raw angles.

    Examples:
        Create an OBB26 detection head
        >>> obb26 = OBB26(nc=80, ne=1, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb26(x)
    """

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, angle_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and raw angles."""
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if angle_head is not None:
            bs = x[0].shape[0]  # batch size
            angle = torch.cat(
                [angle_head[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2
            )  # OBB theta logits (raw output without sigmoid transformation)
            preds["angle"] = angle
        return preds


class Pose(Detect):
    """YOLO Pose head for keypoints models.

    This class extends the Detect head to include keypoint prediction capabilities for pose estimation tasks.

    Attributes:
        kpt_shape (tuple): Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible).
        nk (int): Total number of keypoint values.
        cv4 (nn.ModuleList): Convolution layers for keypoint prediction.

    Methods:
        forward: Perform forward pass through YOLO model and return predictions.
        kpts_decode: Decode keypoints from predictions.

    Examples:
        Create a pose detection head
        >>> pose = Pose(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize YOLO network with default parameters and Convolutional Layers.

        Args:
            nc (int): Number of classes.
            kpt_shape (tuple): Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible).
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, pose_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, pose_head=self.one2one_cv4)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with keypoints."""
        preds = super()._inference(x)
        return torch.cat([preds, self.kpts_decode(x["kpts"])], dim=1)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, pose_head: torch.nn.Module
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and keypoints."""
        preds = super().forward_head(x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]  # batch size
            preds["kpts"] = torch.cat([pose_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nk) with last dimension
                format [x1, y1, x2, y2, class_probs, keypoints].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + self.nk) and
                last dimension format [x1, y1, x2, y2, max_class_prob, class_index, keypoints].
        """
        boxes, scores, kpts = preds.split([4, self.nc, self.nk], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        kpts = kpts.gather(dim=1, index=idx.repeat(1, 1, self.nk))
        return torch.cat([boxes, scores, conf, kpts], dim=-1)

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = self.cv4 = None

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions."""
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Pose26(Pose):
    """YOLO26 Pose head for keypoints models.

    This class extends the Pose head with normalizing flow for keypoint prediction in pose estimation tasks.

    Attributes:
        kpt_shape (tuple): Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible).
        nk (int): Total number of keypoint values.
        cv4 (nn.ModuleList): Convolution layers for keypoint prediction.

    Methods:
        forward: Perform forward pass through YOLO model and return predictions.
        kpts_decode: Decode keypoints from predictions.

    Examples:
        Create a pose detection head
        >>> pose = Pose26(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize YOLO network with default parameters and Convolutional Layers.

        Args:
            nc (int): Number of classes.
            kpt_shape (tuple): Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible).
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, kpt_shape, reg_max, end2end, ch)
        self.flow_model = RealNVP()

        c4 = max(ch[0] // 4, kpt_shape[0] * (kpt_shape[1] + 2))
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3)) for x in ch)

        self.cv4_kpts = nn.ModuleList(nn.Conv2d(c4, self.nk, 1) for _ in ch)
        self.nk_sigma = kpt_shape[0] * 2  # sigma_x, sigma_y for each keypoint
        self.cv4_sigma = nn.ModuleList(nn.Conv2d(c4, self.nk_sigma, 1) for _ in ch)

        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)
            self.one2one_cv4_kpts = copy.deepcopy(self.cv4_kpts)
            self.one2one_cv4_sigma = copy.deepcopy(self.cv4_sigma)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for backward compatibility."""
        return dict(
            box_head=self.cv2,
            cls_head=self.cv3,
            pose_head=self.cv4,
            kpts_head=self.cv4_kpts,
            kpts_sigma_head=self.cv4_sigma,
        )

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            pose_head=self.one2one_cv4,
            kpts_head=self.one2one_cv4_kpts,
            kpts_sigma_head=self.one2one_cv4_sigma,
        )

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        pose_head: torch.nn.Module,
        kpts_head: torch.nn.Module,
        kpts_sigma_head: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and keypoints."""
        preds = Detect.forward_head(self, x, box_head, cls_head)
        if pose_head is not None:
            bs = x[0].shape[0]  # batch size
            features = [pose_head[i](x[i]) for i in range(self.nl)]
            preds["kpts"] = torch.cat([kpts_head[i](features[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
            if self.training:
                preds["kpts_sigma"] = torch.cat(
                    [kpts_sigma_head[i](features[i]).view(bs, self.nk_sigma, -1) for i in range(self.nl)], 2
                )
        return preds

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        super().fuse()
        self.cv4_kpts = self.cv4_sigma = self.flow_model = self.one2one_cv4_sigma = None

    def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions."""
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            y = kpts.view(bs, *self.kpt_shape, -1)
            # NCNN fix
            a = (y[:, :, :2] + self.anchors) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] + self.anchors[0]) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] + self.anchors[1]) * self.strides
            return y


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2).

    This class implements a classification head that transforms feature maps into class predictions.

    Attributes:
        export (bool): Export mode flag.
        conv (Conv): Convolutional layer for feature transformation.
        pool (nn.AdaptiveAvgPool2d): Global average pooling layer.
        drop (nn.Dropout): Dropout layer for regularization.
        linear (nn.Linear): Linear layer for final classification.

    Methods:
        forward: Perform forward pass on input feature maps.

    Examples:
        Create a classification head
        >>> classify = Classify(c1=1024, c2=1000)
        >>> x = torch.randn(1, 1024, 20, 20)
        >>> output = classify(x)
    """

    export = False  # export mode

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """Initialize YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output classes.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor | tuple:
        """Perform forward pass on input feature maps."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to incorporate text embeddings for enhanced semantic understanding in
    object detection tasks.

    Attributes:
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        bias_init: Initialize detection head biases.

    Examples:
        Create a WorldDetect head
        >>> world_detect = WorldDetect(nc=80, embed=512, with_bn=False, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = world_detect(x, text)
    """

    def __init__(
        self,
        nc: int = 80,
        embed: int = 512,
        with_bn: bool = False,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        """Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max=reg_max, end2end=end2end, ch=ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> dict[str, torch.Tensor] | tuple:
        """Concatenate and return predicted bounding boxes and class probabilities."""
        feats = [xi.clone() for xi in x]  # save original features for anchor generation
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        bs = x[0].shape[0]
        x_cat = torch.cat([xi.view(bs, self.no, -1) for xi in x], 2)
        boxes, scores = x_cat.split((self.reg_max * 4, self.nc), 1)
        preds = dict(boxes=boxes, scores=scores, feats=feats)
        if self.training:
            return preds
        y = self._inference(preds)
        return y if self.export else (y, preds)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class LRPCHead(nn.Module):
    """Lightweight Region Proposal and Classification Head for efficient object detection.

    This head combines region proposal filtering with classification to enable efficient detection with dynamic
    vocabulary support.

    Attributes:
        vocab (nn.Module): Vocabulary/classification layer.
        pf (nn.Module): Proposal filter module.
        loc (nn.Module): Localization module.
        enabled (bool): Whether the head is enabled.

    Methods:
        conv2linear: Convert a 1x1 convolutional layer to a linear layer.
        forward: Process classification and localization features to generate detection proposals.

    Examples:
        Create an LRPC head
        >>> vocab = nn.Conv2d(256, 80, 1)
        >>> pf = nn.Conv2d(256, 1, 1)
        >>> loc = nn.Conv2d(256, 4, 1)
        >>> head = LRPCHead(vocab, pf, loc, enabled=True)
    """

    def __init__(self, vocab: nn.Module, pf: nn.Module, loc: nn.Module, enabled: bool = True):
        """Initialize LRPCHead with vocabulary, proposal filter, and localization components.

        Args:
            vocab (nn.Module): Vocabulary/classification module.
            pf (nn.Module): Proposal filter module.
            loc (nn.Module): Localization module.
            enabled (bool): Whether to enable the head functionality.
        """
        super().__init__()
        self.vocab = self.conv2linear(vocab) if enabled else vocab
        self.pf = pf
        self.loc = loc
        self.enabled = enabled

    @staticmethod
    def conv2linear(conv: nn.Conv2d) -> nn.Linear:
        """Convert a 1x1 convolutional layer to a linear layer."""
        assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
        linear = nn.Linear(conv.in_channels, conv.out_channels)
        linear.weight.data = conv.weight.view(conv.out_channels, -1).data
        linear.bias.data = conv.bias.data
        return linear

    def forward(self, cls_feat: torch.Tensor, loc_feat: torch.Tensor, conf: float) -> tuple[tuple, torch.Tensor]:
        """Process classification and localization features to generate detection proposals."""
        if self.enabled:
            pf_score = self.pf(cls_feat)[0, 0].flatten(0)
            mask = pf_score.sigmoid() > conf
            cls_feat = cls_feat.flatten(2).transpose(-1, -2)
            cls_feat = self.vocab(cls_feat[:, mask] if conf else cls_feat * mask.unsqueeze(-1).int())
            return self.loc(loc_feat), cls_feat.transpose(-1, -2), mask
        else:
            cls_feat = self.vocab(cls_feat)
            loc_feat = self.loc(loc_feat)
            return (
                loc_feat,
                cls_feat.flatten(2),
                torch.ones(cls_feat.shape[2] * cls_feat.shape[3], device=cls_feat.device, dtype=torch.bool),
            )






class YOLOEDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to support text-guided detection with enhanced semantic understanding
    through text embeddings and visual prompt embeddings.

    Attributes:
        is_fused (bool): Whether the model is fused for inference.
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.
        reprta (Residual): Residual block for text prompt embeddings.
        savpe (SAVPE): Spatial-aware visual prompt embeddings module.
        embed (int): Embedding dimension.

    Methods:
        fuse: Fuse text features with model weights for efficient inference.
        get_tpe: Get text prompt embeddings with normalization.
        get_vpe: Get visual prompt embeddings with spatial awareness.
        forward_lrpc: Process features with fused text embeddings for prompt-free model.
        forward: Process features with class prompt embeddings to generate detections.
        bias_init: Initialize biases for detection heads.

    Examples:
        Create a YOLOEDetect head
        >>> yoloe_detect = YOLOEDetect(nc=80, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> cls_pe = torch.randn(1, 80, 512)
        >>> outputs = yoloe_detect(x, cls_pe)
    """

    is_fused = False

    def __init__(
        self, nc: int = 80, embed: int = 512, with_bn: bool = False, reg_max=16, end2end=False, ch: tuple = ()
    ):
        """Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        c3 = max(ch[0], min(self.nc, 100))
        assert c3 <= embed
        assert with_bn
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, embed, 1),
                )
                for x in ch
            )
        )
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)
        if end2end:
            self.one2one_cv3 = copy.deepcopy(self.cv3)  # overwrite with new cv3
            self.one2one_cv4 = copy.deepcopy(self.cv4)

        self.reprta = Residual(SwiGLUFFN(embed, embed))
        self.savpe = SAVPE(ch, c3, embed)
        self.embed = embed

    @smart_inference_mode()
    def fuse(self, txt_feats: torch.Tensor = None):
        """Fuse text features with model weights for efficient inference."""
        if txt_feats is None:  # means eliminate one2many branch
            self.cv2 = self.cv3 = self.cv4 = None
            return
        if self.is_fused:
            return

        assert not self.training
        txt_feats = txt_feats.to(torch.float32).squeeze(0)
        self._fuse_tp(txt_feats, self.cv3, self.cv4)
        if self.end2end:
            self._fuse_tp(txt_feats, self.one2one_cv3, self.one2one_cv4)
        del self.reprta
        self.reprta = nn.Identity()
        self.is_fused = True

    def _fuse_tp(self, txt_feats: torch.Tensor, cls_head: torch.nn.Module, bn_head: torch.nn.Module) -> None:
        """Fuse text prompt embeddings with model weights for efficient inference."""
        for cls_h, bn_h in zip(cls_head, bn_head):
            assert isinstance(cls_h, nn.Sequential)
            assert isinstance(bn_h, BNContrastiveHead)
            conv = cls_h[-1]
            assert isinstance(conv, nn.Conv2d)
            logit_scale = bn_h.logit_scale
            bias = bn_h.bias
            norm = bn_h.norm

            t = txt_feats * logit_scale.exp()
            conv: nn.Conv2d = fuse_conv_and_bn(conv, norm)

            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data

            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bias

            conv = (
                nn.Conv2d(
                    conv.in_channels,
                    w.shape[0],
                    kernel_size=1,
                )
                .requires_grad_(False)
                .to(conv.weight.device)
            )

            conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            conv.bias.data.copy_(b1 + b2)
            cls_h[-1] = conv

            bn_h.fuse()

    def get_tpe(self, tpe: torch.Tensor | None) -> torch.Tensor | None:
        """Get text prompt embeddings with normalization."""
        return None if tpe is None else F.normalize(self.reprta(tpe), dim=-1, p=2)

    def get_vpe(self, x: list[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
        """Get visual prompt embeddings with spatial awareness."""
        if vpe.shape[1] == 0:  # no visual prompt embeddings
            return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device)
        if vpe.ndim == 4:  # (B, N, H, W)
            vpe = self.savpe(x, vpe)
        assert vpe.ndim == 3  # (B, N, D)
        return vpe

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with class prompt embeddings to generate detections."""
        if hasattr(self, "lrpc"):  # for prompt-free inference
            return self.forward_lrpc(x[:3])
        return super().forward(x)

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with fused text embeddings to generate detections for prompt-free model."""
        boxes, scores, index = [], [], []
        bs = x[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv3
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                0 if self.export and not self.dynamic else getattr(self, "conf", 0.001),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        preds = dict(boxes=torch.cat(boxes, 2), scores=torch.cat(scores, 2), feats=x, index=torch.cat(index))
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _get_decode_boxes(self, x):
        """Decode predicted bounding boxes for inference."""
        dbox = super()._get_decode_boxes(x)
        if hasattr(self, "lrpc"):
            dbox = dbox if self.export and not self.dynamic else dbox[..., x["index"]]
        return dbox

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, contrastive_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, contrastive_head=self.one2one_cv4)

    def forward_head(self, x, box_head, cls_head, contrastive_head):
        """Concatenates and returns predicted bounding boxes, class probabilities, and contrastive scores."""
        assert len(x) == 4, f"Expected 4 features including 3 feature maps and 1 text embeddings, but got {len(x)}."
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]  # batch size
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        self.nc = x[-1].shape[1]
        scores = torch.cat(
            [contrastive_head[i](cls_head[i](x[i]), x[-1]).reshape(bs, self.nc, -1) for i in range(self.nl)], dim=-1
        )
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        return dict(boxes=boxes, scores=scores, feats=x[:3])

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for i, (a, b, c) in enumerate(
            zip(self.one2many["box_head"], self.one2many["cls_head"], self.one2many["contrastive_head"])
        ):
            a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[:] = 0.0
            c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b, c) in enumerate(
                zip(self.one2one["box_head"], self.one2one["cls_head"], self.one2one["contrastive_head"])
            ):
                a[-1].bias.data[:] = 2.0  # box
                b[-1].bias.data[:] = 0.0
                c.bias.data[:] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)


class YOLOESegment(YOLOEDetect):
    """YOLO segmentation head with text embedding capabilities.

    This class extends YOLOEDetect to include mask prediction capabilities for instance segmentation tasks with
    text-guided semantic understanding.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv5 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a YOLOESegment head
        >>> yoloe_segment = YOLOESegment(nc=80, nm=32, npr=256, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = yoloe_segment(x, text)
    """

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        """Initialize YOLOESegment with class count, mask parameters, and embedding dimensions.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv5, contrastive_head=self.cv4)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(
            box_head=self.one2one_cv2,
            cls_head=self.one2one_cv3,
            mask_head=self.one2one_cv5,
            contrastive_head=self.one2one_cv4,
        )

    def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Process features with fused text embeddings to generate detections for prompt-free model."""
        boxes, scores, index = [], [], []
        bs = x[0].shape[0]
        cv2 = self.cv2 if not self.end2end else self.one2one_cv2
        cv3 = self.cv3 if not self.end2end else self.one2one_cv3
        cv5 = self.cv5 if not self.end2end else self.one2one_cv5
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            box, score, idx = self.lrpc[i](
                cls_feat,
                loc_feat,
                0 if self.export and not self.dynamic else getattr(self, "conf", 0.001),
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        mc = torch.cat([cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        index = torch.cat(index)
        preds = dict(
            boxes=torch.cat(boxes, 2),
            scores=torch.cat(scores, 2),
            feats=x,
            index=index,
            mask_coefficient=mc * index.int() if self.export and not self.dynamic else mc[..., index],
        )
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = super().forward(x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])  # mask protos
        if isinstance(preds, dict):  # training and validating during training
            if self.end2end:
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities, concatenated with mask coefficients."""
        preds = super()._inference(x)
        return torch.cat([preds, x["mask_coefficient"]], dim=1)

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: torch.nn.Module,
        cls_head: torch.nn.Module,
        mask_head: torch.nn.Module,
        contrastive_head: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes, class probabilities, and mask coefficients."""
        preds = super().forward_head(x, box_head, cls_head, contrastive_head)
        if mask_head is not None:
            bs = x[0].shape[0]  # batch size
            preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        return preds

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc + nm) with last dimension
                format [x1, y1, x2, y2, class_probs, mask_coefficient].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + nm) and last
                dimension format [x1, y1, x2, y2, max_class_prob, class_index, mask_coefficient].
        """
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def fuse(self, txt_feats: torch.Tensor = None):
        """Fuse text features with model weights for efficient inference."""
        super().fuse(txt_feats)
        if txt_feats is None:  # means eliminate one2many branch
            self.cv5 = None
            if hasattr(self.proto, "fuse"):
                self.proto.fuse()
            return


class YOLOESegment26(YOLOESegment):
    """YOLOE-style segmentation head module using Proto26 for mask generation.

    This class extends the YOLOESegment functionality to include segmentation capabilities by integrating a Proto26
    generation module and convolutional layers to predict mask coefficients.

    Args:
        nc (int): Number of classes. Defaults to 80.
        nm (int): Number of masks. Defaults to 32.
        npr (int): Number of prototype channels. Defaults to 256.
        embed (int): Embedding dimensionality. Defaults to 512.
        with_bn (bool): Whether to use Batch Normalization. Defaults to False.
        reg_max (int): Maximum number of DFL channels. Defaults to 16.
        end2end (bool): Whether to use end-to-end detection mode. Defaults to False.
        ch (tuple[int, ...]): Input channels for each scale.

    Attributes:
        nm (int): Number of segmentation masks.
        npr (int): Number of prototype channels.
        proto (Proto26): Prototype generation module for segmentation.
        cv5 (nn.ModuleList): Convolutional layers for generating mask coefficients from features.
        one2one_cv5 (nn.ModuleList, optional): Deep copy of cv5 for end-to-end detection branches.
    """

    def __init__(
        self,
        nc: int = 80,
        nm: int = 32,
        npr: int = 256,
        embed: int = 512,
        with_bn: bool = False,
        reg_max=16,
        end2end=False,
        ch: tuple = (),
    ):
        """Initialize YOLOESegment26 with class count, mask parameters, and embedding dimensions."""
        YOLOEDetect.__init__(self, nc, embed, with_bn, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto26(ch, self.npr, self.nm, nc)  # protos

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)
        if end2end:
            self.one2one_cv5 = copy.deepcopy(self.cv5)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        outputs = YOLOEDetect.forward(self, x)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto([xi.detach() for xi in x], return_semseg=False)  # mask protos

        if isinstance(preds, dict):  # training and validating during training
            if self.end2end and not hasattr(self, "lrpc"):  # not prompt-free
                preds["one2many"]["proto"] = proto
                preds["one2one"]["proto"] = proto.detach()
            else:
                preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)


class RTDETRDecoder(nn.Module):
    """Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.

    Attributes:
        export (bool): Export mode flag.
        hidden_dim (int): Dimension of hidden layers.
        nhead (int): Number of heads in multi-head attention.
        nl (int): Number of feature levels.
        nc (int): Number of classes.
        num_queries (int): Number of query points.
        num_decoder_layers (int): Number of decoder layers.
        input_proj (nn.ModuleList): Input projection layers for backbone features.
        decoder (DeformableTransformerDecoder): Transformer decoder module.
        denoising_class_embed (nn.Embedding): Class embeddings for denoising.
        num_denoising (int): Number of denoising queries.
        label_noise_ratio (float): Label noise ratio for training.
        box_noise_scale (float): Box noise scale for training.
        learnt_init_query (bool): Whether to learn initial query embeddings.
        tgt_embed (nn.Embedding): Target embeddings for queries.
        query_pos_head (MLP): Query position head.
        enc_output (nn.Sequential): Encoder output layers.
        enc_score_head (nn.Linear): Encoder score prediction head.
        enc_bbox_head (MLP): Encoder bbox prediction head.
        dec_score_head (nn.ModuleList): Decoder score prediction heads.
        dec_bbox_head (nn.ModuleList): Decoder bbox prediction heads.

    Methods:
        forward: Run forward pass and return bounding box and classification scores.

    Examples:
        Create an RTDETRDecoder
        >>> decoder = RTDETRDecoder(nc=80, ch=(512, 1024, 2048), hd=256, nq=300)
        >>> x = [torch.randn(1, 512, 64, 64), torch.randn(1, 1024, 32, 32), torch.randn(1, 2048, 16, 16)]
        >>> outputs = decoder(x)
    """

    export = False  # export mode
    shapes = []
    anchors = torch.empty(0)
    valid_mask = torch.empty(0)
    dynamic = False

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,  # hidden dim
        nq: int = 300,  # num queries
        ndp: int = 4,  # num decoder points
        nh: int = 8,  # num head
        ndl: int = 6,  # num decoder layers
        d_ffn: int = 1024,  # dim of feedforward
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        # Training args
        nd: int = 100,  # num denoising
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
    ):
        """Initialize the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes.
            ch (tuple): Channels in the backbone feature maps.
            hd (int): Dimension of hidden layers.
            nq (int): Number of query points.
            ndp (int): Number of decoder points.
            nh (int): Number of heads in multi-head attention.
            ndl (int): Number of decoder layers.
            d_ffn (int): Dimension of the feed-forward networks.
            dropout (float): Dropout rate.
            act (nn.Module): Activation function.
            eval_idx (int): Evaluation index.
            nd (int): Number of denoising.
            label_noise_ratio (float): Label noise ratio.
            box_noise_scale (float): Box noise scale.
            learnt_init_query (bool): Whether to learn initial query embeddings.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor:
        """Run the forward pass of the module, returning bounding box and classification scores for the input.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.
            batch (dict, optional): Batch information for training.

        Returns:
            outputs (tuple | torch.Tensor): During training, returns a tuple of bounding boxes, scores, and other
                metadata. During inference, returns a tensor of shape (bs, 300, 4+nc) containing bounding boxes and
                class scores.
        """
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    @staticmethod
    def _generate_anchors(
        shapes: list[list[int]],
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        eps: float = 1e-2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate anchor bounding boxes for given shapes with specific grid size and validate them.

        Args:
            shapes (list): List of feature map shapes.
            grid_size (float, optional): Base size of grid cells.
            dtype (torch.dtype, optional): Data type for tensors.
            device (str, optional): Device to create tensors on.
            eps (float, optional): Small value for numerical stability.

        Returns:
            anchors (torch.Tensor): Generated anchor boxes.
            valid_mask (torch.Tensor): Valid mask for anchors.
        """
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, list[list[int]]]:
        """Process and return encoder inputs by getting projection features from input and concatenating them.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.

        Returns:
            feats (torch.Tensor): Processed features.
            shapes (list): List of feature map shapes.
        """
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(
        self,
        feats: torch.Tensor,
        shapes: list[list[int]],
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate and prepare the input required for the decoder from the provided features and shapes.

        Args:
            feats (torch.Tensor): Processed features from encoder.
            shapes (list): List of feature map shapes.
            dn_embed (torch.Tensor, optional): Denoising embeddings.
            dn_bbox (torch.Tensor, optional): Denoising bounding boxes.

        Returns:
            embeddings (torch.Tensor): Query embeddings for decoder.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            enc_bboxes (torch.Tensor): Encoded bounding boxes.
            enc_scores (torch.Tensor): Encoded scores.
        """
        bs = feats.shape[0]
        if self.dynamic or self.shapes != shapes:
            self.anchors, self.valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
            self.shapes = shapes

        # Prepare input for decoder
        features = self.enc_output(self.valid_mask * feats)  # bs, h*w, 256
        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs*num_queries,)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs*num_queries,)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = self.anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def _reset_parameters(self):
        """Initialize or reset the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    """v10 Detection head from https://arxiv.org/pdf/2405.14458.

    This class implements the YOLOv10 detection head with dual-assignment training and consistent dual predictions for
    improved efficiency and performance.

    Attributes:
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum number of detections.
        cv3 (nn.ModuleList): Light classification head layers.
        one2one_cv3 (nn.ModuleList): One-to-one classification head layers.

    Methods:
        __init__: Initialize the v10Detect object with specified number of classes and input channels.
        forward: Perform forward pass of the v10Detect module.
        bias_init: Initialize biases of the Detect module.
        fuse: Remove the one2many head for inference optimization.

    Examples:
        Create a v10Detect head
        >>> v10_detect = v10Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = v10_detect(x)
    """

    end2end = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """Initialize the v10Detect object with the specified number of classes and input channels.

        Args:
            nc (int): Number of classes.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, end2end=True, ch=ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    def fuse(self):
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = None


def _coreset_subsample(mem: "torch.Tensor", max_size: int) -> "torch.Tensor":
    """Greedy k-center coreset selection on an L2-normalised feature matrix.

    Selects *max_size* rows from *mem* that maximally cover the feature space
    (minimises the largest nearest-neighbour distance to any un-selected point).
    Uses cosine distance (1 - cos_sim) as the metric; features must be L2-normalised.

    Complexity: O(max_size × M) dot-products — runs once at freeze time.

    Args:
        mem: [M, C] L2-normalised feature matrix.
        max_size: Target number of entries to keep.

    Returns:
        [max_size, C] selected subset.
    """
    import torch
    M = mem.shape[0]
    if M <= max_size:
        return mem
    device = mem.device
    # dist_to_selected[i] = cosine distance from point i to its nearest selected centre.
    # Initialise to inf (nothing selected yet).
    dist = torch.full((M,), float("inf"), device=device, dtype=torch.float32)
    selected: list[int] = []
    # Seed: pick the point closest to the global mean (most "typical").
    mean = mem.mean(dim=0)
    mean = mean / mean.norm().clamp(min=1e-8)
    seed = int((mem @ mean).argmax().item())
    selected.append(seed)
    for _ in range(max_size - 1):
        centre = mem[selected[-1]].unsqueeze(0)          # [1, C]
        cos_sim = (mem @ centre.t()).squeeze(1)           # [M]  in [-1, 1]
        new_dist = (1.0 - cos_sim).clamp(min=0.0)        # cosine distance
        dist = torch.minimum(dist, new_dist)
        selected.append(int(dist.argmax().item()))
    idx = torch.tensor(selected, device=device)
    return mem[idx]


class ADMBHead(nn.Module):
    """Memory-bank anomaly detection head.

    Scores each anchor position by cosine similarity against stored normal-image
    features, replacing the standard proposal-filter (pf) module.

    During a warmup pass (``update=True``) features are accumulated into the memory
    bank.  During inference (``update=False``) each position receives a Noisy-OR
    anomaly score; positions that exceed ``conf`` are kept as proposals.

    Attributes:
        vocab_linear (nn.Linear): Linear classifier used in detect-mode scoring.
        loc (nn.Module): Box-regression (localization) module.
        memory_bank (Tensor): L2-normalised normal-image features [N, C].
        feature_dim (int | None): Channel depth, inferred on first accumulation.
        update (bool): Accumulate features when True; score positions when False.
        temperature (float): Noisy-OR temperature exponent (default 1.0).
        anomaly_mode (bool): Output single-channel anomaly logit (True) or
            nc-class vocabulary scores (False).
    """

    def __init__(self, vocab: nn.Module, loc: nn.Module,
                 anomaly_args: dict | None = None,
                 # Legacy individual params — used when anomaly_args is None (backward compat).
                 temperature: float = 3.0, accumulate_thresh: float = 0.4,
                 K: int = 15, score_filter_kernel: int = 1) -> None:
        super().__init__()
        # Accept either a 1x1 Conv2d or a pre-built Linear (e.g. copied from another head).
        self.vocab_linear = vocab if isinstance(vocab, nn.Linear) else self._conv2linear(vocab)
        self.loc = loc
        self.register_buffer("memory_bank", torch.empty(0, 0), persistent=True)
        self.feature_dim: int | None = None
        self.update = True
        self.enabled = True  # set False to skip this head entirely (no accumulation, no proposals)
        self._calibrated = False
        self._calibration_image_count = 0

        # All scoring / calibration params — seeded from anomaly_args dict (or legacy kwargs).
        args = anomaly_args if anomaly_args is not None else {
            "temperature": temperature,
            "accumulate_thresh": accumulate_thresh,
            "K": K,
            "score_filter_kernel": score_filter_kernel,
        }
        self.temperature               = args.get("temperature",               3.0)
        self.K                         = args.get("K",                         15)
        self.accumulate_thresh         = args.get("accumulate_thresh",         0.4)
        self.score_filter_kernel       = args.get("score_filter_kernel",       1)
        self.auto_temperature          = args.get("auto_temperature",          True)
        self.calibration_interval      = args.get("calibration_interval",      0)
        self.min_calibration_bank_size = args.get("min_calibration_bank_size", 50)
        self.calibration_target_score  = args.get("calibration_target_score",  0.2)
        self.em_iters                  = args.get("em_iters",                  1)
        self.max_bank_size             = args.get("max_bank_size",             None)
        self.score_aggregation         = args.get("score_aggregation",         "max")

        assert self.calibration_target_score < self.accumulate_thresh, (
            f"calibration_target_score ({self.calibration_target_score}) must be "
            f"< accumulate_thresh ({self.accumulate_thresh})"
        )


    # ── configuration ────────────────────────────────────────────────────────

    def set_update(self, update: bool) -> None:
        """Toggle memory-bank accumulation on (True) / off (False)."""
        self.update = update

    def reset_memory_bank(self) -> None:
        """Discard all accumulated normal features and reset calibration state."""
        self.memory_bank = torch.empty((0, 0), device=self.memory_bank.device)
        self.feature_dim = None
        self._calibrated = False
        self._calibration_image_count = 0

    def compress_memory_bank(self, max_size: int) -> None:
        """Compress the memory bank to *max_size* entries using greedy k-center coreset selection.

        Features are already L2-normalised, so cosine distance  (1 − cos_sim) is used as the
        distance metric.  Greedy k-center guarantees that the selected subset maximally covers
        the original feature space (minimises the largest gap), which is far better than random
        subsampling.  Complexity: O(max_size × M) dot-products — runs once, on the stored bank.

        Called automatically after each OBMA pass when ``max_bank_size`` is set.
        Can also be called manually at any time, e.g. after ``load_support_set``.
        """
        mem = self._effective_memory_bank()
        if mem.shape[0] <= max_size:
            return
        self.memory_bank = _coreset_subsample(mem, max_size)

    def get_memory_bank_stats(self) -> dict:
        """Return size and feature dimension of the current memory bank."""
        mem = self._effective_memory_bank()
        return {
            "size": mem.shape[0],
            "feature_dim": self.feature_dim,
        }

    def _effective_memory_bank(self) -> torch.Tensor:
        """Return the real memory-bank entries, excluding zero-padding placeholders."""
        mem = self._memory_tensor()
        if self.feature_dim is None or mem.numel() == 0 or mem.shape[0] == 0:
            return mem[:0]
        valid = mem.norm(dim=1) > 0
        return mem[valid]

    def _prepare_cls_feat(self, cls_feat: torch.Tensor) -> torch.Tensor:
        """Apply the configured pre-scoring spatial filter."""
        if self.score_filter_kernel > 1:
            pad = self.score_filter_kernel // 2
            cls_feat = F.avg_pool2d(
                cls_feat,
                kernel_size=self.score_filter_kernel,
                stride=1,
                padding=pad,
                count_include_pad=True,
            )
        return cls_feat

    def forward_dense(self, cls_feat: torch.Tensor, anomaly_mode: bool = True, return_logits: bool = False) -> torch.Tensor:
        """Return dense anomaly or classification scores without proposal filtering."""
        if not self.enabled:
            channels = 1 if anomaly_mode else self.vocab_linear.out_features
            return torch.zeros(
                cls_feat.shape[0],
                channels,
                cls_feat.shape[2],
                cls_feat.shape[3],
                device=cls_feat.device,
                dtype=cls_feat.dtype,
            )

        batch_size, channels, height, width = cls_feat.shape
        if self.feature_dim is None:
            self.feature_dim = channels

        cls_feat = self._prepare_cls_feat(cls_feat)
        if self.update:
            self._online_bootstrapped_memory_accumulation(cls_feat, batch_size, height, width, self.accumulate_thresh)

        if anomaly_mode:
            heatmap = self._anomaly_scores(cls_feat, mem=self._memory_tensor()).view(batch_size, 1, height, width)
            if return_logits:
                heatmap = heatmap.clamp(1e-6, 1 - 1e-6)
                heatmap = torch.log(heatmap / (1 - heatmap))
            return heatmap

        cls_scores = self.vocab_linear(cls_feat.flatten(2).transpose(-1, -2)).transpose(-1, -2)
        return cls_scores.view(batch_size, self.vocab_linear.out_features, height, width)

    # ── internals ─────────────────────────────────────────────────────────────

    def _calibrate_temperature(self, normed_features: torch.Tensor, mem: torch.Tensor) -> None:
        """Auto-calibrate the Noisy-OR temperature β from the current memory bank.

        Solves for β such that a "typical normal" feature (at the 90th-percentile
        of mean top-k cosine similarities) receives an anomaly score equal to
        ``self.calibration_target_score``.  This keeps the score distribution
        consistent regardless of how similar or dissimilar the dataset features are.

        Formula (from the Noisy-OR single-entry approximation):
            score ≈ 1 - exp(-β·(1 - s_typical))
            β = -ln(1 - target_score) / (1 - s_typical)

        Args:
            normed_features: L2-normalised feature matrix [N, C] from the current batch.
            mem: Current memory bank [M, C].
        """
        with torch.no_grad():
            k = min(self.K, mem.shape[0])
            # Sub-sample query rows to cap cost: 512 rows × M cols
            sample = normed_features
            if sample.shape[0] > 512:
                idx = torch.randperm(sample.shape[0], device=sample.device)[:512]
                sample = sample[idx]
            sim = sample @ mem.t()                                  # [n, M]
            topk_sim = sim.topk(k=k, dim=1).values                 # [n, k]
            mean_topk = topk_sim.mean(dim=1)                       # [n]  — avg similarity per position
            s_typical = torch.quantile(mean_topk, 0.90).clamp(0.0, 1.0 - 1e-4)
            beta = -math.log(1.0 - self.calibration_target_score) / (1.0 - s_typical.item())
            beta = max(0.1, min(20.0, beta))                        # safety clamp
        old_temp = self.temperature
        self.temperature = beta
        LOGGER.info(
            "ADMBHead: auto-calibrated temperature %.3f → %.3f  (s_90=%.4f, target_score=%.2f)",
            old_temp, beta, s_typical.item(), self.calibration_target_score,
        )

    @staticmethod
    def _conv2linear(conv: nn.Conv2d) -> nn.Linear:
        """Convert a 1×1 Conv2d to an equivalent nn.Linear."""
        assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
        linear = nn.Linear(conv.in_channels, conv.out_channels)
        linear.weight.data = conv.weight.view(conv.out_channels, -1).data
        linear.bias.data = conv.bias.data
        return linear

    def _memory_tensor(self) -> torch.Tensor:
        device = self.memory_bank.device
        embed_dim = self.feature_dim if self.feature_dim is not None else self.vocab_linear.in_features

        if self.memory_bank.dim() != 2 or self.memory_bank.shape[1] != embed_dim or self.memory_bank.shape[0] == 0:
            self.memory_bank = torch.zeros((10, embed_dim), device=device)
        return self.memory_bank

    def _anomaly_scores(self, features: torch.Tensor, mem: torch.Tensor | None = None, chunk_size: int | None = None) -> torch.Tensor:
        """Return Noisy-OR anomaly scores ∈ [0, 1] for every spatial position.

        Shape: [B*H*W].  0 = normal, 1 = anomalous.
        Falls back to 0.5 (neutral) when the memory bank is empty.

        Args:
            chunk_size: Number of query positions processed at once to bound peak memory.
                        Peak sim matrix = chunk_size × mem_size × 4 bytes
                        (e.g. 4096 × 57 000 ≈ 936 MB instead of N × M all at once).
                        Set to None to disable chunking (processes all N positions at once —
                        risks OOM if N × M is large, e.g. N=102 400, M=57 000 → ~23 GB).
        """
        if mem is None:
            mem = self._effective_memory_bank()
        elif mem.numel() and mem.dim() == 2:
            mem = mem[mem.norm(dim=1) > 0]
        if mem is None or mem.numel() == 0 or mem.shape[0] == 0:
            n = features.numel() // features.shape[1] if features.dim() == 4 else features.shape[0]
            return torch.full((n,), 0.5, device=features.device)
        if features.dim() == 4:
            B, C, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(-1, C)
        q = F.normalize(features.view(-1, self.feature_dim), p=2, dim=1)

        N = q.shape[0]
        k = min(self.K, mem.shape[0])
        prob = torch.empty(N, device=q.device, dtype=q.dtype)

        # chunk_size=None → single-shot (original behaviour, risks OOM for large N×M).
        # Any positive integer caps the sim-matrix rows, bounding peak memory to
        # chunk_size × M × 4 bytes regardless of batch size or spatial resolution.
        _chunk = N if chunk_size is None else chunk_size
        for start in range(0, N, _chunk):
            end = min(start + _chunk, N)
            q_chunk = q[start:end]                              # [chunk, C]

            # calculate cosine similarity between query features and memory bank features
            sim = q_chunk @ mem.t()                             # [chunk, M]

            # ψ(x) = exp(−β(1 − x)): maps cosine similarity → [0, 1] with temperature β
            sim = torch.exp(-self.temperature * (1 - sim))     # [chunk, M]

            # Top-k most similar bank entries per query position
            topk_sim = sim.topk(k=k, dim=1).values             # [chunk, k]

            # Noisy-OR (uniform weights): P(anomaly) = exp(mean(log(1 - sim_i)))
            score = (1 - topk_sim).clamp(min=1e-8)
            log_prob = (topk_sim.new_ones(1, k) / k * torch.log(score)).sum(dim=1)
            prob[start:end] = torch.exp(log_prob).clamp(0, 1)

        return prob

    def _online_bootstrapped_memory_accumulation(
        self,
        cls_feat: torch.Tensor,
        B: int,
        H: int,
        W: int,
        accumulate_thresh: float,
    ) -> torch.Tensor:
        """Online Bootstrapped Memory Accumulation (OBMA) with sequential deduplication.

        For each image in the batch, candidate positions (initial anomaly score >
        ``accumulate_thresh``) are processed one-by-one in descending novelty order.
        After each accepted feature is added to the memory bank the bank is updated
        immediately, so subsequent candidates from the **same image** are re-scored
        against the already-updated bank.  This guarantees that no two features
        contributed by the same image are mutually redundant.

        Returns:
            keep: bool [H*W] — positions accepted from the first batch image.
        """
        keep_flags = torch.zeros(H * W, dtype=torch.bool, device=cls_feat.device)

        # Pre-cache normalised features — reused across all EM passes.
        # Cost: B × H*W × C floats (e.g. 4 × 6400 × 256 ≈ 26 MB per layer — acceptable).
        all_normed = [
            F.normalize(
                cls_feat[b].permute(1, 2, 0).reshape(H * W, self.feature_dim).detach(),
                p=2, dim=1,
            )
            for b in range(B)
        ]  # list of [H*W, C]

        # ── Coreset fast-path ────────────────────────────────────────────────
        # When max_bank_size is set, collect ALL spatial features from every image
        # (no threshold filtering) so that freeze_memory_bank() has the full picture
        # before running k-center coreset once across all batches.
        if self.max_bank_size is not None:
            cur_bank = self._effective_memory_bank()
            all_new = torch.cat(
                [n.to(cls_feat.device, dtype=cls_feat.dtype) for n in all_normed], dim=0
            )  # [B*H*W, C] — everything, no filtering
            self.memory_bank = (
                torch.cat([cur_bank, all_new], dim=0)
                if cur_bank.shape[0] > 0
                else all_new
            )
            LOGGER.debug(
                "ADMBHead(coreset fast-path): batch collected +%d, bank_size=%d (compress deferred to freeze)",
                all_new.shape[0],
                self.memory_bank.shape[0],
            )
            return keep_flags
        # ── /Coreset fast-path ───────────────────────────────────────────────

        mem = self._effective_memory_bank()
        total_added = 0

        def _run_obma_pass(mem_in: torch.Tensor, is_first_pass: bool):
            """One E-step: iterate over all images, add novel features to mem_in."""
            mem = mem_in
            added = 0
            for b, normed_b in enumerate(all_normed):
                scores = self._anomaly_scores(normed_b, mem=mem)  # [H*W]
                cand_idx = (scores > accumulate_thresh).nonzero(as_tuple=True)[0]
                if cand_idx.numel() == 0:
                    continue
                cand_idx = cand_idx[scores[cand_idx].argsort(descending=True)]
                cand_feats = normed_b[cand_idx]
                for i in range(cand_feats.shape[0]):
                    feat = cand_feats[i : i + 1]
                    if self._anomaly_scores(feat, mem=mem).item() > accumulate_thresh:
                        mem = torch.cat((mem, feat.to(mem.device, dtype=mem.dtype)), dim=0)
                        added += 1
                        if b == 0 and is_first_pass:
                            keep_flags[cand_idx[i]] = True
            return mem, added

        def _maybe_calibrate(mem: torch.Tensor) -> bool:
            """M-step: recalibrate β from the current bank if conditions are met.

            Returns True if calibration actually ran (β was updated).
            """
            if not self.auto_temperature:
                return False
            if mem.shape[0] < self.min_calibration_bank_size:
                return False
            if self._calibrated and not (
                self.calibration_interval > 0
                and self._calibration_image_count % self.calibration_interval == 0
            ):
                return False
            self._calibrate_temperature(torch.cat(all_normed, dim=0), mem)
            self._calibrated = True
            return True

        # Unified EM loop: E-step → M-step every iteration.
        #
        # Reset-and-rebuild (when em_iters >= 2 and auto_temperature):
        #   Pass 0: seed the bank with β=initial (may be over-inclusive).
        #   M-step: calibrate β from the seeded bank.
        #   If calibration ran → discard the seed bank and rebuild from scratch
        #   with the correct β on pass 1.  This ensures the final bank was built
        #   entirely under the calibrated temperature, not the default 3.0.
        #   Pass 2+: standard convergence check.
        self._calibration_image_count += B
        em_iters_run = 0
        for t in range(self.em_iters):
            is_first_pass = (t == 0)

            # E-step
            mem, added = _run_obma_pass(mem, is_first_pass=is_first_pass)
            total_added += added
            em_iters_run = t + 1

            # M-step — calibrate β from the bank just built
            just_calibrated = _maybe_calibrate(mem)

            # Reset-and-rebuild: after seeding on pass 0 and calibrating β,
            # throw away the seed bank so pass 1 builds cleanly with correct β.
            if is_first_pass and just_calibrated and self.em_iters >= 2:
                mem = torch.zeros((0, self.feature_dim), device=cls_feat.device, dtype=cls_feat.dtype)
                keep_flags.zero_()  # visualisation flags will be set properly in pass 1
                total_added = 0     # restart counter — seed bank is discarded
                continue            # go straight to pass 1 E-step

            # Convergence check — skip on pass 0 (bank was just seeded)
            if not is_first_pass and added == 0:
                LOGGER.debug("OBMA-EM: converged at iter %d.", t)
                break

        self.memory_bank = mem
        LOGGER.info(
            "OBMA: total_added=%d keep=%.1f%% mem_size=%d temperature=%.3f em_extra_iters=%d/%d",
            total_added,
            100.0 * keep_flags.sum() / max(H * W, 1),
            self.memory_bank.shape[0],
            self.temperature,
            em_iters_run - 1,
            max(0, self.em_iters - 1),
        )
        return keep_flags

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, cls_feat: torch.Tensor, loc_feat: torch.Tensor, conf: float = 0.5, anomaly_mode: bool = True) -> tuple:
        """Forward pass.

        If ``self.enabled`` is False, skips accumulation and returns empty proposals
        so this head contributes nothing to detections.

        Memory-bank building (``update=True``): accumulates features and returns
        dummy proposals (discarded by the caller).

        Inference (``update=False``): filters positions by anomaly score and returns:
          - ``anomaly_mode=True``  → logit-encoded anomaly probability [B, 1, k]
          - ``anomaly_mode=False`` → vocabulary classification scores  [B, nc, k]

        Returns:
            tuple: (loc_preds [B, reg_max*4, H, W],
                    cls_scores [B, 1|nc, k],
                    mask       bool [H*W])
        """
        B, C, H, W = cls_feat.shape
        if not self.enabled:
            nc = 1 if anomaly_mode else self.vocab_linear.out_features
            # Export/dense (conf==0): return H*W zeros so output shape is fixed.
            # Normal inference: return 0-width tensor (no proposals from disabled head).
            k = H * W if conf == 0 else 0
            empty_scores = torch.zeros(B, nc, k, device=cls_feat.device, dtype=cls_feat.dtype)
            empty_mask = torch.ones(H * W, dtype=torch.bool, device=cls_feat.device) if conf == 0 \
                else torch.zeros(H * W, dtype=torch.bool, device=cls_feat.device)
            return self.loc(loc_feat), empty_scores, empty_mask

        if self.feature_dim is None:
            self.feature_dim = C

        cls_feat = self._prepare_cls_feat(cls_feat)

        # compute per-image anomaly scores: [B, H*W]
        scores_per_image = self._anomaly_scores(cls_feat, mem=self._memory_tensor()).view(B, -1)

        # The shared mask (union across batch) determines WHICH positions are candidates.
        # Using max(dim=0) means: flag a position if ANY image in the batch scores it high.
        # Each image then gets its OWN score at those positions so normal images are not
        # falsely assigned the high score of an anomalous batch-mate.
        scores_hw = scores_per_image.max(dim=0).values  # [H*W] — for mask only

        # accumulate high-confidence normal features into the memory bank (only when self.update is True)
        if self.update:
            accumulate_mask = self._online_bootstrapped_memory_accumulation(cls_feat, B, H, W,
                                                                             self.accumulate_thresh)

        # infer flow — two paths:
        #   conf == 0 (export/dense): return all H*W positions with real scores, no boolean indexing.
        #     Fixed output shape [B, 1|nc, H*W] is required for ONNX. Downstream topk + predictor
        #     conf filter handle selection. Mirrors the conf=0 pattern in LRPCHead/YOLOEDetect.
        #   conf > 0 (normal inference): sparse boolean mask → variable-k output (fast, memory-efficient).
        if conf == 0:
            mask = torch.ones(H * W, dtype=torch.bool, device=cls_feat.device)
            if anomaly_mode:
                per_img_scores = scores_per_image.clamp(1e-6, 1 - 1e-6)                     # [B, H*W]
                cls_scores = torch.log(per_img_scores / (1 - per_img_scores)).unsqueeze(1)  # [B, 1, H*W]
            else:
                cls_flat = cls_feat.flatten(2).transpose(-1, -2)                            # [B, H*W, C]
                cls_scores = self.vocab_linear(cls_flat).transpose(-1, -2)                  # [B, nc, H*W]
        else:
            mask = scores_hw > conf                                                          # [H*W]
            if anomaly_mode:
                # Per-image logits: each image uses its own anomaly probability at the selected
                # positions, NOT the shared batch-max score. This prevents a single anomalous
                # image from contaminating normal images in the same batch with its high scores.
                per_img_scores = scores_per_image[:, mask].clamp(1e-6, 1 - 1e-6)           # [B, k]
                cls_scores = torch.log(per_img_scores / (1 - per_img_scores)).unsqueeze(1)  # [B, 1, k]
            else:
                cls_flat = cls_feat.flatten(2).transpose(-1, -2)                            # [B, H*W, C]
                cls_scores = self.vocab_linear(cls_flat[:, mask]).transpose(-1, -2)         # [B, nc, k]

        return self.loc(loc_feat), cls_scores, mask


class AnomalyDetection(Detect):
    """
    Anomaly Detection head based on YOLO/YOLOE architecture.

    Arguments:
    
    
    """
    is_fused = False
    _fixed_nc=None

    def __init__(
        self, nc: int = 80, embed: int = 512, with_bn: bool = True, reg_max=16, end2end=False, ch: tuple = ()
    ):
        """
        Initialize Anomaly Detection head with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension (kept for YAML compatibility with YOLOEDetect).
            with_bn (bool): Batch normalization flag (kept for YAML compatibility).
            reg_max (int): DFL bins.
            end2end (bool): Whether to use end-to-end mode.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, reg_max, end2end, ch)
        self.adhead = None  # Anomaly detection head will be built separately with build_adhead()
        self.fused_adhead = None
        self.last_heatmap = None

        # Central parameter store — all anomaly-specific params live here.
        # Use set_anomaly_args(**kwargs) or direct attribute access (via @property) to update.
        self.anomaly_args: dict = {
            "ad_conf": 0.5, "ad_max_det": 9, "anomaly_mode": True,
            "feature_mode": "per_level", "return_heatmap": False, "heatmap_logits": False,
            "fused_layers": [0, 1], "fused_use_pre_clshead": False,
            "accumulate_thresh": 0.4, "temperature": 3.0, "K": 15, "score_filter_kernel": 1,
            "auto_temperature": True, "calibration_interval": 0, "calibration_target_score": 0.2,
            "min_calibration_bank_size": 50, "em_iters": 1, "max_bank_size": None,
            "score_aggregation": "max",
        }

        # Auto-build ADMBHead sub-modules when constructed from YAML (not from from_detect_head)
        if ch:
            self.build_adhead()

    def __getattr__(self, name: str):
        """Backward-compat fallback: build anomaly_args from old flat attrs on legacy .pt models.

        Also provides transparent dict-key access for any key in anomaly_args that doesn't have
        an explicit @property (e.g. ``head.em_iters``, ``head.auto_temperature``).
        """
        if name == "anomaly_args":
            _defaults = {
                "ad_conf": 0.5, "ad_max_det": 9, "anomaly_mode": True,
                "feature_mode": "per_level", "return_heatmap": False, "heatmap_logits": False,
                "fused_layers": [0, 1], "fused_use_pre_clshead": False,
                "accumulate_thresh": 0.4, "temperature": 3.0, "K": 15, "score_filter_kernel": 1,
                "auto_temperature": True, "calibration_interval": 0, "calibration_target_score": 0.2,
                "min_calibration_bank_size": 50, "em_iters": 1, "max_bank_size": None,
                "score_aggregation": "max",
            }
            args = {k: (list(v) if isinstance(v, list) else v) for k, v in _defaults.items()}
            for k in _defaults:
                v = self.__dict__.get(k)
                if v is not None:
                    args[k] = list(v) if isinstance(v, list) else v
            # Clean up stale flat attrs from __dict__ to avoid confusion.
            for k in _defaults:
                self.__dict__.pop(k, None)
            self.__dict__["anomaly_args"] = args
            return args
        if "anomaly_args" in self.__dict__ and name in self.__dict__["anomaly_args"]:
            return self.__dict__["anomaly_args"][name]
        return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        """Route anomaly_args key writes to the dict; handle side-effects for anomaly_mode/feature_mode."""
        if "anomaly_args" in self.__dict__ and name in self.__dict__["anomaly_args"]:
            if name == "anomaly_mode":
                if getattr(self, "adhead", None) is not None:
                    self.set_anomaly_mode(bool(value))
                else:
                    self.__dict__["anomaly_args"]["anomaly_mode"] = bool(value)
            elif name == "feature_mode":
                if value not in {"per_level", "fused_heatmap"}:
                    raise ValueError(f"Unsupported feature_mode={value!r}.")
                self.__dict__["anomaly_args"]["feature_mode"] = value
                if value == "fused_heatmap" and getattr(self, "adhead", None) is not None:
                    if not isinstance(getattr(self, "fused_adhead", None), ADMBHead):
                        self.fused_adhead = self._build_fused_head(copy.deepcopy(self.adhead[0].vocab_linear))
            else:
                self.__dict__["anomaly_args"][name] = value
        else:
            super().__setattr__(name, value)

    @classmethod
    def from_detect_head(cls, head: "Detect") -> "AnomalyDetection":
        """Create an AnomalyDetection from an existing Detect/YOLOEDetect head, reusing all trained weights.

        Unlike ``head.__class__ = AnomalyDetection``, this creates a genuine new object so the
        original head reference is cleanly discarded after ``self.model[-1] = new_head``.

        Implementation notes:
          * ``nn.Module.__init__`` is called first to get a properly initialised Module
            (fresh hook-dicts, empty registries, etc.).
          * The exact set of keys created by ``nn.Module.__init__`` is captured as ``_skip``
            so the code is robust across PyTorch versions.
          * ``training`` is always copied from *head* (overrides the default ``True`` set by
            ``nn.Module.__init__``).
          * ``_modules`` / ``_parameters`` / ``_buffers`` are copied element-by-element so
            both the new head and the discarded old head never share the same dict object.

        Args:
            head: An existing ``Detect`` (or subclass) head module.

        Returns:
            AnomalyDetection: New instance sharing trained submodule objects with *head*.
        """
        new = object.__new__(cls)
        nn.Module.__init__(new)  # sets up _modules, _parameters, _buffers, hooks, training=True

        # Capture all keys initialised by nn.Module.__init__ so we can skip them
        # when copying plain Python attrs (version-agnostic).
        _skip = frozenset(new.__dict__)

        # Copy plain Python instance attributes (nc, nl, reg_max, stride, inplace, …)
        for k, v in head.__dict__.items():
            if k not in _skip:
                new.__dict__[k] = v

        # training is in _skip (nn.Module.__init__ sets it to True), so override explicitly.
        new.training = head.training

        # Transfer registered submodules / params / buffers (the actual trained weights).
        new._modules.update(head._modules)
        new._parameters.update(head._parameters)
        new._buffers.update(head._buffers)
        new._non_persistent_buffers_set.update(getattr(head, "_non_persistent_buffers_set", set()))

        # Set AD-specific defaults (absent on a raw Detect/YOLOEDetect head).
        new.adhead = None
        new.fused_adhead = getattr(head, "fused_adhead", None)
        new.last_heatmap = getattr(head, "last_heatmap", None)
        # Build anomaly_args: migrate from old flat attrs if head is pre-refactor style.
        if "anomaly_args" not in new.__dict__:
            _defaults = {
                "ad_conf": 0.5, "ad_max_det": 9, "anomaly_mode": True,
                "feature_mode": "per_level", "return_heatmap": False, "heatmap_logits": False,
                "fused_layers": [0, 1], "fused_use_pre_clshead": False,
                "accumulate_thresh": 0.4, "temperature": 3.0, "K": 15, "score_filter_kernel": 1,
                "auto_temperature": True, "calibration_interval": 0, "calibration_target_score": 0.2,
                "min_calibration_bank_size": 50, "em_iters": 1, "max_bank_size": None,
                "score_aggregation": "max",
            }
            args = {k: (list(v) if isinstance(v, list) else v) for k, v in _defaults.items()}
            for k in _defaults:
                v = new.__dict__.pop(k, None)
                if v is not None:
                    args[k] = list(v) if isinstance(v, list) else v
            new.__dict__["anomaly_args"] = args
        new.__dict__.setdefault("original_nc", head.nc)
        return new

    def _get_feature_heads(self) -> tuple[nn.ModuleList, nn.ModuleList]:
        """Return truncated localization and classification branches for anomaly scoring."""
        has_end2end = hasattr(self, "one2one_cv2")
        return (self.one2one_cv2, self.one2one_cv3) if has_end2end else (self.cv2, self.cv3)

    def _build_fused_head(self, vocab: "nn.Module | nn.Linear") -> ADMBHead:
        """Build the dedicated fused memory-bank head for dense anomaly heatmaps.

        ``vocab`` may be either the original 1×1 Conv2d (from ``build_adhead``) or a
        pre-built ``nn.Linear`` copied from an existing ADMBHead (for backward-compat
        with models saved before ``fused_adhead`` existed).
        """
        fused_head = ADMBHead(
            vocab=vocab,
            loc=nn.Identity(),
            anomaly_args=self.anomaly_args,
        )
        fused_head._memory_tensor()
        return fused_head

    def build_adhead(self):
        """Build anomaly detection sub-heads from self's cv2/cv3 layers.

        Saves the final conv layers BEFORE deep-copying and truncating, so adhead
        always receives the original (pre-truncation) final layers.

        End2end models (yolo26, YOLOE-seg): use `one2one_cv2` / `one2one_cv3`.
        Non-end2end models (yolov8, yolov5 …): use `cv2` / `cv3`.
        """
        import copy
        self.original_nc = self.nc  # save for set_anomaly_mode(False)

        _e2e = hasattr(self, "one2one_cv2")
        src_cv2 = self.one2one_cv2 if _e2e else self.cv2
        src_cv3 = self.one2one_cv3 if _e2e else self.cv3
        assert len(src_cv2) == self.nl, "Number of heads must match number of feature levels."

        # Save references to the original final layers BEFORE any rebinding.
        # These become the vocab / loc modules of each ADMBHead.
        saved_vocab = [src_cv3[i][-1] for i in range(self.nl)]
        saved_loc   = [src_cv2[i][-1] for i in range(self.nl)]

        # Deep-copy and truncate the cv branches so the backbone features are
        # forwarded up to (but not including) the original final projection.
        if _e2e:
            self.one2one_cv2 = copy.deepcopy(self.one2one_cv2)
            self.one2one_cv3 = copy.deepcopy(self.one2one_cv3)
            cv2, cv3 = self.one2one_cv2, self.one2one_cv3
        else:
            self.cv2 = copy.deepcopy(self.cv2)
            self.cv3 = copy.deepcopy(self.cv3)
            cv2, cv3 = self.cv2, self.cv3

        for loc_head, cls_head in zip(cv2, cv3):
            assert isinstance(loc_head, nn.Sequential)
            assert isinstance(cls_head, nn.Sequential)
            del loc_head[-1]
            del cls_head[-1]

        self.adhead = nn.ModuleList(
            ADMBHead(vocab=saved_vocab[i], loc=saved_loc[i], anomaly_args=self.anomaly_args)
            for i in range(self.nl)
        )
        self.fused_adhead = self._build_fused_head(saved_vocab[0])
        # Pre-fill memory banks with zeros so forward() never hits an empty-bank edge case
        # (e.g. during stride computation in DetectionModel.__init__).
        for h in self.adhead:
            h._memory_tensor()

    def bias_init(self):
        """No-op: ADMBHead manages its own scoring; standard Detect bias init does not apply."""
        pass

    def _propagate_to_subheads(self) -> None:
        """Sync shared params from anomaly_args to every ADMBHead instance."""
        if getattr(self, "adhead", None) is None:
            return
        for h in self.iter_ad_heads():
            for k, v in self.anomaly_args.items():
                if hasattr(h, k):
                    setattr(h, k, v)
        # Invariant: calibration_target_score < accumulate_thresh.
        for h in self.iter_ad_heads():
            assert h.calibration_target_score < h.accumulate_thresh, (
                f"calibration_target_score ({h.calibration_target_score}) must be "
                f"< accumulate_thresh ({h.accumulate_thresh})"
            )

    def set_anomaly_mode(self, anomaly_mode: bool) -> None:
        """Switch between anomaly scoring (nc=1) and original classification (nc=original_nc).

        Args:
            anomaly_mode (bool): True  = memory-bank cosine-similarity scoring, single class.
                                 False = original nc-class classification (e.g., 80 COCO classes).
        """
        assert self.adhead is not None, "Call build_adhead() first."
        self.anomaly_args["anomaly_mode"] = bool(anomaly_mode)  # keep dict in sync
        self.nc = 1 if anomaly_mode else getattr(self, "original_nc", self.nc)
        for h in self.adhead:
            h.anomaly_mode = anomaly_mode
        fused = getattr(self, "fused_adhead", None)
        if fused is not None:
            fused.anomaly_mode = anomaly_mode

    def iter_ad_heads(self, include_fused: bool = True) -> list[ADMBHead]:
        """Return all anomaly memory-bank heads, optionally including the fused heatmap head."""
        heads = [h for h in (self.adhead or []) if isinstance(h, ADMBHead)]
        fused = getattr(self, "fused_adhead", None)
        if include_fused and isinstance(fused, ADMBHead):
            heads.append(fused)
        return heads

    def _build_fused_feature_map(self, x: list[torch.Tensor], cls_heads: nn.ModuleList) -> torch.Tensor:
        """Fuse multi-scale anomaly features onto the highest-resolution selected scale.

        Source features:
          * ``self.fused_use_pre_clshead = False`` (default) — fuse ``cls_heads[i](x[i])``.
          * ``self.fused_use_pre_clshead = True``           — fuse raw backbone features ``x[i]``.

        Layer selection:
          * ``self.fused_layers = None`` (default) — fuse all ``self.nl`` layers.
          * ``self.fused_layers = [1, 2]`` (e.g.)  — fuse only those indices.
        The first selected index becomes the base resolution; remaining feats are
        nearest-interpolated up to it. Channels must match across selected layers.
        """
        # Use getattr so models loaded from .pt files saved before these attrs existed still work.
        fused_layers = getattr(self, "fused_layers", None)
        use_pre_clshead = getattr(self, "fused_use_pre_clshead", False)
        layers = fused_layers if fused_layers else list(range(self.nl))
        if not layers:
            raise ValueError("fused_layers is empty; need at least one layer to fuse.")
        # use_pre_clshead=True # for debug
        if use_pre_clshead:
            feats = [x[i] for i in layers]
        else:
            feats = [cls_heads[i](x[i]) for i in layers]

        base = feats[0]
        target_size = base.shape[-2:]
        target_channels = base.shape[1]
        fused = base
        for feat in feats[1:]:
            if feat.shape[1] != target_channels:
                raise ValueError(
                    f"Fused anomaly heatmap requires matching channels across fused_layers={layers}, "
                    f"got {target_channels} and {feat.shape[1]}. "
                    f"With fused_use_pre_clshead={use_pre_clshead}, channels per layer are: "
                    f"{[x[i].shape[1] if use_pre_clshead else cls_heads[i](x[i]).shape[1] for i in layers]}."
                )
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="nearest")
            fused = fused + feat
        return fused

    def forward_heatmap(self, x: list[torch.Tensor], cls_heads: nn.ModuleList | None = None) -> torch.Tensor:
        """Build a dense anomaly heatmap from fused multi-scale features."""
        fused_adhead = getattr(self, "fused_adhead", None)
        if fused_adhead is None:
            raise RuntimeError("Call build_adhead() before forward_heatmap().")
        if cls_heads is None:
            _, cls_heads = self._get_feature_heads()
        heatmap = fused_adhead.forward_dense(
            self._build_fused_feature_map(x, cls_heads),
            anomaly_mode=self.anomaly_mode,
            return_logits=self.heatmap_logits,
        )
        self.last_heatmap = heatmap
        return heatmap


    def set_anomaly_args(
        self,
        active_layers: "list[int] | None" = None,
        mode: "str | None" = None,
        **kwargs,
    ) -> None:
        """Set anomaly-detection parameters from keyword arguments.

        All parameter names map directly to keys in ``self.anomaly_args``.
        Property setters fire automatically via ``setattr``, handling side-effects
        (e.g. ``feature_mode`` auto-builds ``fused_adhead``, ``anomaly_mode`` updates ``self.nc``).

        Special non-dict args:
            active_layers (list[int] | None): Indices of adheads to enable; None = all.
            mode (str | None): Shortcut to switch mode (``'anomaly'`` or ``'detect'``).

        Valid keys: see ``self.anomaly_args`` for the full list.
        """
        unknown = set(kwargs) - self.anomaly_args.keys()
        if unknown:
            raise ValueError(f"Unknown anomaly args: {unknown}. Valid keys: {set(self.anomaly_args)}")
        for k, v in kwargs.items():
            setattr(self, k, v)  # routes through __setattr__ → handles side-effects for anomaly_mode/feature_mode
        self._propagate_to_subheads()
        if active_layers is not None and self.adhead is not None:
            for i, h in enumerate(self.adhead):
                h.enabled = (i in active_layers)
        if mode is not None:
            self.set_anomaly_mode(mode == "anomaly")

    def _get_decode_boxes(self, x):
        """Decode boxes; for end2end filter to anomaly-selected positions, for non-end2end keep all."""
        dbox = super()._get_decode_boxes(x)
        if self.adhead is not None and "index" in x:
            dbox = dbox if (self.export and not self.dynamic) else dbox[..., x["index"]]
        return dbox

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Run anomaly detection forward pass.

        Sparse filtered output via memory-bank cosine-similarity scoring.
        End2end models apply postprocess (top-k); non-end2end uses predictor NMS.
        Memory bank is pre-filled with zeros by build_adhead(), so this works
        even during stride computation in DetectionModel.__init__.
        """
        assert self.adhead is not None, "Call build_adhead() before forward()."

        bs = x[0].shape[0]
        # Always use the branches that build_adhead() truncated — determined by architecture,
        # not by self.end2end.  self.end2end only controls the *output format* (top-k vs raw).
        cv2, cv3 = self._get_feature_heads()

        # Export/dense mode: conf=0 routes each ADMBHead to the fixed-shape path (no boolean
        # indexing), producing [B, 1|nc, H*W] scores. Downstream postprocess topk + predictor
        # conf threshold do all selection. Mirrors YOLOEDetect.forward_lrpc (conf=0 on export).
        ad_conf = 0 if (self.export and not self.dynamic) else self.ad_conf

        heatmap = None
        if self.feature_mode == "fused_heatmap" or self.return_heatmap:
            heatmap = self.forward_heatmap(x, cls_heads=cv3)

        boxes, scores, index = [], [], []
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            box, score, idx = self.adhead[i](
                cls_feat,
                loc_feat,
                ad_conf,
                self.anomaly_mode
            )
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)

        preds = dict(boxes=torch.cat(boxes, 2), scores=torch.cat(scores, 2), feats=x, index=torch.cat(index))
        if heatmap is not None:
            preds["heatmap"] = heatmap
            preds["feature_mode"] = self.feature_mode
        if self.training:
            if self.end2end:
                one2one = {
                    key: value.detach() if isinstance(value, torch.Tensor) else value
                    for key, value in preds.items()
                }
                return {"one2many": preds, "one2one": one2one}
            return preds
        # Do not mutate self.nc from tensor shape during export — set_anomaly_mode() keeps it correct.
        if not (self.export and not self.dynamic):
            self.nc = preds["scores"].shape[1]
        y = self._inference(preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)
    

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x1, y1, x2, y2, class_probs].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x1, y1, x2, y2, max_class_prob, class_index].
        """
        boxes, scores = preds.split([4, self.nc], dim=-1)

        scores, conf, idx = self.get_topk_index(scores, self.ad_max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores, conf], dim=-1)
    