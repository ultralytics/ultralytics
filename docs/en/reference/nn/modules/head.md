---
description: Explore docs covering Ultralytics YOLO detection, pose & RTDETRDecoder. Comprehensive guides to help you understand Ultralytics nn modules.
keywords: Ultralytics, YOLO, Detection, Pose, RTDETRDecoder, nn modules, guides
---

# Reference for `ultralytics/nn/modules/head.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`Detect`](#ultralytics.nn.modules.head.Detect)
        - [`Segment`](#ultralytics.nn.modules.head.Segment)
        - [`Segment26`](#ultralytics.nn.modules.head.Segment26)
        - [`OBB`](#ultralytics.nn.modules.head.OBB)
        - [`OBB26`](#ultralytics.nn.modules.head.OBB26)
        - [`Pose`](#ultralytics.nn.modules.head.Pose)
        - [`Pose26`](#ultralytics.nn.modules.head.Pose26)
        - [`Classify`](#ultralytics.nn.modules.head.Classify)
        - [`WorldDetect`](#ultralytics.nn.modules.head.WorldDetect)
        - [`LRPCHead`](#ultralytics.nn.modules.head.LRPCHead)
        - [`YOLOEDetect`](#ultralytics.nn.modules.head.YOLOEDetect)
        - [`YOLOESegment`](#ultralytics.nn.modules.head.YOLOESegment)
        - [`YOLOESegment26`](#ultralytics.nn.modules.head.YOLOESegment26)
        - [`RTDETRDecoder`](#ultralytics.nn.modules.head.RTDETRDecoder)
        - [`v10Detect`](#ultralytics.nn.modules.head.v10Detect)

    === "<span class="doc-kind doc-kind-property">Properties</span>"

        - [`Detect.one2many`](#ultralytics.nn.modules.head.Detect.one2many)
        - [`Detect.one2one`](#ultralytics.nn.modules.head.Detect.one2one)
        - [`Detect.end2end`](#ultralytics.nn.modules.head.Detect.end2end)
        - [`Segment.one2many`](#ultralytics.nn.modules.head.Segment.one2many)
        - [`Segment.one2one`](#ultralytics.nn.modules.head.Segment.one2one)
        - [`OBB.one2many`](#ultralytics.nn.modules.head.OBB.one2many)
        - [`OBB.one2one`](#ultralytics.nn.modules.head.OBB.one2one)
        - [`Pose.one2many`](#ultralytics.nn.modules.head.Pose.one2many)
        - [`Pose.one2one`](#ultralytics.nn.modules.head.Pose.one2one)
        - [`Pose26.one2many`](#ultralytics.nn.modules.head.Pose26.one2many)
        - [`Pose26.one2one`](#ultralytics.nn.modules.head.Pose26.one2one)
        - [`YOLOEDetect.one2many`](#ultralytics.nn.modules.head.YOLOEDetect.one2many)
        - [`YOLOEDetect.one2one`](#ultralytics.nn.modules.head.YOLOEDetect.one2one)
        - [`YOLOESegment.one2many`](#ultralytics.nn.modules.head.YOLOESegment.one2many)
        - [`YOLOESegment.one2one`](#ultralytics.nn.modules.head.YOLOESegment.one2one)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`Detect.end2end`](#ultralytics.nn.modules.head.Detect.end2end)
        - [`Detect.forward_head`](#ultralytics.nn.modules.head.Detect.forward_head)
        - [`Detect.forward`](#ultralytics.nn.modules.head.Detect.forward)
        - [`Detect._inference`](#ultralytics.nn.modules.head.Detect._inference)
        - [`Detect._get_decode_boxes`](#ultralytics.nn.modules.head.Detect._get_decode_boxes)
        - [`Detect.bias_init`](#ultralytics.nn.modules.head.Detect.bias_init)
        - [`Detect.decode_bboxes`](#ultralytics.nn.modules.head.Detect.decode_bboxes)
        - [`Detect.postprocess`](#ultralytics.nn.modules.head.Detect.postprocess)
        - [`Detect.get_topk_index`](#ultralytics.nn.modules.head.Detect.get_topk_index)
        - [`Detect.fuse`](#ultralytics.nn.modules.head.Detect.fuse)
        - [`Segment.forward`](#ultralytics.nn.modules.head.Segment.forward)
        - [`Segment._inference`](#ultralytics.nn.modules.head.Segment._inference)
        - [`Segment.forward_head`](#ultralytics.nn.modules.head.Segment.forward_head)
        - [`Segment.postprocess`](#ultralytics.nn.modules.head.Segment.postprocess)
        - [`Segment.fuse`](#ultralytics.nn.modules.head.Segment.fuse)
        - [`Segment26.forward`](#ultralytics.nn.modules.head.Segment26.forward)
        - [`Segment26.fuse`](#ultralytics.nn.modules.head.Segment26.fuse)
        - [`OBB._inference`](#ultralytics.nn.modules.head.OBB._inference)
        - [`OBB.forward_head`](#ultralytics.nn.modules.head.OBB.forward_head)
        - [`OBB.decode_bboxes`](#ultralytics.nn.modules.head.OBB.decode_bboxes)
        - [`OBB.postprocess`](#ultralytics.nn.modules.head.OBB.postprocess)
        - [`OBB.fuse`](#ultralytics.nn.modules.head.OBB.fuse)
        - [`OBB26.forward_head`](#ultralytics.nn.modules.head.OBB26.forward_head)
        - [`Pose._inference`](#ultralytics.nn.modules.head.Pose._inference)
        - [`Pose.forward_head`](#ultralytics.nn.modules.head.Pose.forward_head)
        - [`Pose.postprocess`](#ultralytics.nn.modules.head.Pose.postprocess)
        - [`Pose.fuse`](#ultralytics.nn.modules.head.Pose.fuse)
        - [`Pose.kpts_decode`](#ultralytics.nn.modules.head.Pose.kpts_decode)
        - [`Pose26.forward_head`](#ultralytics.nn.modules.head.Pose26.forward_head)
        - [`Pose26.fuse`](#ultralytics.nn.modules.head.Pose26.fuse)
        - [`Pose26.kpts_decode`](#ultralytics.nn.modules.head.Pose26.kpts_decode)
        - [`Classify.forward`](#ultralytics.nn.modules.head.Classify.forward)
        - [`WorldDetect.forward`](#ultralytics.nn.modules.head.WorldDetect.forward)
        - [`WorldDetect.bias_init`](#ultralytics.nn.modules.head.WorldDetect.bias_init)
        - [`LRPCHead.conv2linear`](#ultralytics.nn.modules.head.LRPCHead.conv2linear)
        - [`LRPCHead.forward`](#ultralytics.nn.modules.head.LRPCHead.forward)
        - [`YOLOEDetect.fuse`](#ultralytics.nn.modules.head.YOLOEDetect.fuse)
        - [`YOLOEDetect._fuse_tp`](#ultralytics.nn.modules.head.YOLOEDetect._fuse_tp)
        - [`YOLOEDetect.get_tpe`](#ultralytics.nn.modules.head.YOLOEDetect.get_tpe)
        - [`YOLOEDetect.get_vpe`](#ultralytics.nn.modules.head.YOLOEDetect.get_vpe)
        - [`YOLOEDetect.forward`](#ultralytics.nn.modules.head.YOLOEDetect.forward)
        - [`YOLOEDetect.forward_lrpc`](#ultralytics.nn.modules.head.YOLOEDetect.forward_lrpc)
        - [`YOLOEDetect._get_decode_boxes`](#ultralytics.nn.modules.head.YOLOEDetect._get_decode_boxes)
        - [`YOLOEDetect.forward_head`](#ultralytics.nn.modules.head.YOLOEDetect.forward_head)
        - [`YOLOEDetect.bias_init`](#ultralytics.nn.modules.head.YOLOEDetect.bias_init)
        - [`YOLOESegment.forward_lrpc`](#ultralytics.nn.modules.head.YOLOESegment.forward_lrpc)
        - [`YOLOESegment.forward`](#ultralytics.nn.modules.head.YOLOESegment.forward)
        - [`YOLOESegment._inference`](#ultralytics.nn.modules.head.YOLOESegment._inference)
        - [`YOLOESegment.forward_head`](#ultralytics.nn.modules.head.YOLOESegment.forward_head)
        - [`YOLOESegment.postprocess`](#ultralytics.nn.modules.head.YOLOESegment.postprocess)
        - [`YOLOESegment.fuse`](#ultralytics.nn.modules.head.YOLOESegment.fuse)
        - [`YOLOESegment26.forward`](#ultralytics.nn.modules.head.YOLOESegment26.forward)
        - [`RTDETRDecoder.forward`](#ultralytics.nn.modules.head.RTDETRDecoder.forward)
        - [`RTDETRDecoder._generate_anchors`](#ultralytics.nn.modules.head.RTDETRDecoder._generate_anchors)
        - [`RTDETRDecoder._get_encoder_input`](#ultralytics.nn.modules.head.RTDETRDecoder._get_encoder_input)
        - [`RTDETRDecoder._get_decoder_input`](#ultralytics.nn.modules.head.RTDETRDecoder._get_decoder_input)
        - [`RTDETRDecoder._reset_parameters`](#ultralytics.nn.modules.head.RTDETRDecoder._reset_parameters)
        - [`v10Detect.fuse`](#ultralytics.nn.modules.head.v10Detect.fuse)


## Class `ultralytics.nn.modules.head.Detect` {#ultralytics.nn.modules.head.Detect}

```python
Detect(self, nc: int = 80, reg_max = 16, end2end = False, ch: tuple = ())
```

**Bases:** `nn.Module`

YOLO Detect head for object detection models.

This class implements the detection head used in YOLO models for predicting bounding boxes and class probabilities. It supports both training and inference modes, with optional end-to-end detection capabilities.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `nc` | `int` | Number of classes. | `80` |
| `reg_max` | `int` | Maximum number of DFL channels. | `16` |
| `end2end` | `bool` | Whether to use end-to-end NMS-free detection. | `False` |
| `ch` | `tuple` | Tuple of channel sizes from backbone feature maps. | `()` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `dynamic` | `bool` | Force grid reconstruction. |
| `export` | `bool` | Export mode flag. |
| `format` | `str` | Export format. |
| `end2end` | `bool` | End-to-end detection mode. |
| `max_det` | `int` | Maximum detections per image. |
| `shape` | `tuple` | Input shape. |
| `anchors` | `torch.Tensor` | Anchor points. |
| `strides` | `torch.Tensor` | Feature map strides. |
| `legacy` | `bool` | Backward compatibility for v3/v5/v8/v9/v11 models. |
| `xyxy` | `bool` | Output format, xyxy or xywh. |
| `nc` | `int` | Number of classes. |
| `nl` | `int` | Number of detection layers. |
| `reg_max` | `int` | DFL channels. |
| `no` | `int` | Number of outputs per anchor. |
| `stride` | `torch.Tensor` | Strides computed during build. |
| `cv2` | `nn.ModuleList` | Convolution layers for box regression. |
| `cv3` | `nn.ModuleList` | Convolution layers for classification. |
| `dfl` | `nn.Module` | Distribution Focal Loss layer. |
| `one2one_cv2` | `nn.ModuleList` | One-to-one convolution layers for box regression. |
| `one2one_cv3` | `nn.ModuleList` | One-to-one convolution layers for classification. |

**Methods**

| Name | Description |
| --- | --- |
| [`one2many`](#ultralytics.nn.modules.head.Detect.one2many) | Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility. |
| [`one2one`](#ultralytics.nn.modules.head.Detect.one2one) | Returns the one-to-one head components. |
| [`end2end`](#ultralytics.nn.modules.head.Detect.end2end) | Checks if the model has one2one for v3/v5/v8/v9/v11 backward compatibility. |
| [`_get_decode_boxes`](#ultralytics.nn.modules.head.Detect._get_decode_boxes) | Get decoded boxes based on anchors and strides. |
| [`_inference`](#ultralytics.nn.modules.head.Detect._inference) | Decode predicted bounding boxes and class probabilities based on multiple-level feature maps. |
| [`bias_init`](#ultralytics.nn.modules.head.Detect.bias_init) | Initialize Detect() biases, WARNING: requires stride availability. |
| [`decode_bboxes`](#ultralytics.nn.modules.head.Detect.decode_bboxes) | Decode bounding boxes from predictions. |
| [`end2end`](#ultralytics.nn.modules.head.Detect.end2end) | Override the end-to-end detection mode. |
| [`forward`](#ultralytics.nn.modules.head.Detect.forward) | Concatenates and returns predicted bounding boxes and class probabilities. |
| [`forward_head`](#ultralytics.nn.modules.head.Detect.forward_head) | Concatenates and returns predicted bounding boxes and class probabilities. |
| [`fuse`](#ultralytics.nn.modules.head.Detect.fuse) | Remove the one2many head for inference optimization. |
| [`get_topk_index`](#ultralytics.nn.modules.head.Detect.get_topk_index) | Get top-k indices from scores. |
| [`postprocess`](#ultralytics.nn.modules.head.Detect.postprocess) | Post-processes YOLO model predictions. |

**Examples**

```python
Create a detection head for 80 classes
>>> detect = Detect(nc=80, ch=(256, 512, 1024))
>>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
>>> outputs = detect(x)
```

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L26-L251"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Property `ultralytics.nn.modules.head.Detect.one2many` {#ultralytics.nn.modules.head.Detect.one2many}

```python
def one2many(self)
```

Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L116-L118"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def one2many(self):
    """Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility."""
    return dict(box_head=self.cv2, cls_head=self.cv3)
```
</details>

<br>

### Property `ultralytics.nn.modules.head.Detect.one2one` {#ultralytics.nn.modules.head.Detect.one2one}

```python
def one2one(self)
```

Returns the one-to-one head components.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L121-L123"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def one2one(self):
    """Returns the one-to-one head components."""
    return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)
```
</details>

<br>

### Property `ultralytics.nn.modules.head.Detect.end2end` {#ultralytics.nn.modules.head.Detect.end2end}

```python
def end2end(self)
```

Checks if the model has one2one for v3/v5/v8/v9/v11 backward compatibility.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L126-L128"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def end2end(self):
    """Checks if the model has one2one for v3/v5/v8/v9/v11 backward compatibility."""
    return getattr(self, "_end2end", True) and hasattr(self, "one2one")
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Detect._get_decode_boxes` {#ultralytics.nn.modules.head.Detect.\_get\_decode\_boxes}

```python
def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor
```

Get decoded boxes based on anchors and strides.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `dict[str, torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L175-L183"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
    """Get decoded boxes based on anchors and strides."""
    shape = x["feats"][0].shape  # BCHW
    if self.dynamic or self.shape != shape:
        self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
        self.shape = shape

    dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
    return dbox
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Detect._inference` {#ultralytics.nn.modules.head.Detect.\_inference}

```python
def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor
```

Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `dict[str, torch.Tensor]` | Dictionary of predictions from detection layers. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Concatenated tensor of decoded bounding boxes and class probabilities. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L162-L173"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Detect.bias_init` {#ultralytics.nn.modules.head.Detect.bias\_init}

```python
def bias_init(self)
```

Initialize Detect() biases, WARNING: requires stride availability.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L185-L197"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Detect.decode_bboxes` {#ultralytics.nn.modules.head.Detect.decode\_bboxes}

```python
def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor
```

Decode bounding boxes from predictions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `bboxes` | `torch.Tensor` |  | *required* |
| `anchors` | `torch.Tensor` |  | *required* |
| `xywh` | `bool` |  | `True` |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L199-L206"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
    """Decode bounding boxes from predictions."""
    return dist2bbox(
        bboxes,
        anchors,
        xywh=xywh and not self.end2end and not self.xyxy,
        dim=1,
    )
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Detect.end2end` {#ultralytics.nn.modules.head.Detect.end2end}

```python
def end2end(self, value)
```

Override the end-to-end detection mode.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `value` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L131-L133"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@end2end.setter
def end2end(self, value):
    """Override the end-to-end detection mode."""
    self._end2end = value
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Detect.forward` {#ultralytics.nn.modules.head.Detect.forward}

```python
def forward(
    self, x: list[torch.Tensor]
) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]
```

Concatenates and returns predicted bounding boxes and class probabilities.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L146-L160"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Detect.forward_head` {#ultralytics.nn.modules.head.Detect.forward\_head}

```python
def forward_head(
    self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
) -> dict[str, torch.Tensor]
```

Concatenates and returns predicted bounding boxes and class probabilities.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` |  | *required* |
| `box_head` | `torch.nn.Module` |  | `None` |
| `cls_head` | `torch.nn.Module` |  | `None` |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L135-L144"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Detect.fuse` {#ultralytics.nn.modules.head.Detect.fuse}

```python
def fuse(self) -> None
```

Remove the one2many head for inference optimization.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L249-L251"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def fuse(self) -> None:
    """Remove the one2many head for inference optimization."""
    self.cv2 = self.cv3 = None
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Detect.get_topk_index` {#ultralytics.nn.modules.head.Detect.get\_topk\_index}

```python
def get_topk_index(self, scores: torch.Tensor, max_det: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

Get top-k indices from scores.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `scores` | `torch.Tensor` | Scores tensor with shape (batch_size, num_anchors, num_classes). | *required* |
| `max_det` | `int` | Maximum detections per image. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor, torch.Tensor, torch.Tensor` | Top scores, class indices, and filtered indices. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L224-L247"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Detect.postprocess` {#ultralytics.nn.modules.head.Detect.postprocess}

```python
def postprocess(self, preds: torch.Tensor) -> torch.Tensor
```

Post-processes YOLO model predictions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `torch.Tensor` | Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension<br>    format [x1, y1, x2, y2, class_probs]. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L208-L222"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.head.Segment` {#ultralytics.nn.modules.head.Segment}

```python
Segment(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max = 16, end2end = False, ch: tuple = ())
```

**Bases:** `Detect`

YOLO Segment head for segmentation models.

This class extends the Detect head to include mask prediction capabilities for instance segmentation tasks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `nc` | `int` | Number of classes. | `80` |
| `nm` | `int` | Number of masks. | `32` |
| `npr` | `int` | Number of protos. | `256` |
| `reg_max` | `int` | Maximum number of DFL channels. | `16` |
| `end2end` | `bool` | Whether to use end-to-end NMS-free detection. | `False` |
| `ch` | `tuple` | Tuple of channel sizes from backbone feature maps. | `()` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `nm` | `int` | Number of masks. |
| `npr` | `int` | Number of protos. |
| `proto` | `Proto` | Prototype generation module. |
| `cv4` | `nn.ModuleList` | Convolution layers for mask coefficients. |

**Methods**

| Name | Description |
| --- | --- |
| [`one2many`](#ultralytics.nn.modules.head.Segment.one2many) | Returns the one-to-many head components, here for backward compatibility. |
| [`one2one`](#ultralytics.nn.modules.head.Segment.one2one) | Returns the one-to-one head components. |
| [`_inference`](#ultralytics.nn.modules.head.Segment._inference) | Decode predicted bounding boxes and class probabilities, concatenated with mask coefficients. |
| [`forward`](#ultralytics.nn.modules.head.Segment.forward) | Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients. |
| [`forward_head`](#ultralytics.nn.modules.head.Segment.forward_head) | Concatenates and returns predicted bounding boxes, class probabilities, and mask coefficients. |
| [`fuse`](#ultralytics.nn.modules.head.Segment.fuse) | Remove the one2many head for inference optimization. |
| [`postprocess`](#ultralytics.nn.modules.head.Segment.postprocess) | Post-process YOLO model predictions. |

**Examples**

```python
Create a segmentation head
>>> segment = Segment(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
>>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
>>> outputs = segment(x)
```

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L254-L355"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Property `ultralytics.nn.modules.head.Segment.one2many` {#ultralytics.nn.modules.head.Segment.one2many}

```python
def one2many(self)
```

Returns the one-to-many head components, here for backward compatibility.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L297-L299"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def one2many(self):
    """Returns the one-to-many head components, here for backward compatibility."""
    return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv4)
```
</details>

<br>

### Property `ultralytics.nn.modules.head.Segment.one2one` {#ultralytics.nn.modules.head.Segment.one2one}

```python
def one2one(self)
```

Returns the one-to-one head components.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L302-L304"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def one2one(self):
    """Returns the one-to-one head components."""
    return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, mask_head=self.one2one_cv4)
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Segment._inference` {#ultralytics.nn.modules.head.Segment.\_inference}

```python
def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor
```

Decode predicted bounding boxes and class probabilities, concatenated with mask coefficients.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `dict[str, torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L321-L324"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
    """Decode predicted bounding boxes and class probabilities, concatenated with mask coefficients."""
    preds = super()._inference(x)
    return torch.cat([preds, x["mask_coefficient"]], dim=1)
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Segment.forward` {#ultralytics.nn.modules.head.Segment.forward}

```python
def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]
```

Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L306-L319"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Segment.forward_head` {#ultralytics.nn.modules.head.Segment.forward\_head}

```python
def forward_head(
    self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, mask_head: torch.nn.Module
) -> dict[str, torch.Tensor]
```

Concatenates and returns predicted bounding boxes, class probabilities, and mask coefficients.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` |  | *required* |
| `box_head` | `torch.nn.Module` |  | *required* |
| `cls_head` | `torch.nn.Module` |  | *required* |
| `mask_head` | `torch.nn.Module` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L326-L334"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward_head(
    self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, mask_head: torch.nn.Module
) -> dict[str, torch.Tensor]:
    """Concatenates and returns predicted bounding boxes, class probabilities, and mask coefficients."""
    preds = super().forward_head(x, box_head, cls_head)
    if mask_head is not None:
        bs = x[0].shape[0]  # batch size
        preds["mask_coefficient"] = torch.cat([mask_head[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
    return preds
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Segment.fuse` {#ultralytics.nn.modules.head.Segment.fuse}

```python
def fuse(self) -> None
```

Remove the one2many head for inference optimization.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L353-L355"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def fuse(self) -> None:
    """Remove the one2many head for inference optimization."""
    self.cv2 = self.cv3 = self.cv4 = None
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Segment.postprocess` {#ultralytics.nn.modules.head.Segment.postprocess}

```python
def postprocess(self, preds: torch.Tensor) -> torch.Tensor
```

Post-process YOLO model predictions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `torch.Tensor` | Raw predictions with shape (batch_size, num_anchors, 4 + nc + nm) with last dimension<br>    format [x1, y1, x2, y2, class_probs, mask_coefficient]. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + nm) and last |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L336-L351"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.head.Segment26` {#ultralytics.nn.modules.head.Segment26}

```python
Segment26(self, nc: int = 80, nm: int = 32, npr: int = 256, reg_max = 16, end2end = False, ch: tuple = ())
```

**Bases:** `Segment`

YOLO26 Segment head for segmentation models.

This class extends the Segment head with Proto26 for mask prediction in instance segmentation tasks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `nc` | `int` | Number of classes. | `80` |
| `nm` | `int` | Number of masks. | `32` |
| `npr` | `int` | Number of protos. | `256` |
| `reg_max` | `int` | Maximum number of DFL channels. | `16` |
| `end2end` | `bool` | Whether to use end-to-end NMS-free detection. | `False` |
| `ch` | `tuple` | Tuple of channel sizes from backbone feature maps. | `()` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `nm` | `int` | Number of masks. |
| `npr` | `int` | Number of protos. |
| `proto` | `Proto26` | Prototype generation module. |
| `cv4` | `nn.ModuleList` | Convolution layers for mask coefficients. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.head.Segment26.forward) | Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients. |
| [`fuse`](#ultralytics.nn.modules.head.Segment26.fuse) | Remove the one2many head and extra part of proto module for inference optimization. |

**Examples**

```python
Create a segmentation head
>>> segment = Segment26(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
>>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
>>> outputs = segment(x)
```

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L358-L414"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Segment26.forward` {#ultralytics.nn.modules.head.Segment26.forward}

```python
def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]
```

Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L393-L408"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Segment26.fuse` {#ultralytics.nn.modules.head.Segment26.fuse}

```python
def fuse(self) -> None
```

Remove the one2many head and extra part of proto module for inference optimization.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L410-L414"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def fuse(self) -> None:
    """Remove the one2many head and extra part of proto module for inference optimization."""
    super().fuse()
    if hasattr(self.proto, "fuse"):
        self.proto.fuse()
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.head.OBB` {#ultralytics.nn.modules.head.OBB}

```python
OBB(self, nc: int = 80, ne: int = 1, reg_max = 16, end2end = False, ch: tuple = ())
```

**Bases:** `Detect`

YOLO OBB detection head for detection with rotation models.

This class extends the Detect head to include oriented bounding box prediction with rotation angles.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `nc` | `int` | Number of classes. | `80` |
| `ne` | `int` | Number of extra parameters. | `1` |
| `reg_max` | `int` | Maximum number of DFL channels. | `16` |
| `end2end` | `bool` | Whether to use end-to-end NMS-free detection. | `False` |
| `ch` | `tuple` | Tuple of channel sizes from backbone feature maps. | `()` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `ne` | `int` | Number of extra parameters. |
| `cv4` | `nn.ModuleList` | Convolution layers for angle prediction. |
| `angle` | `torch.Tensor` | Predicted rotation angles. |

**Methods**

| Name | Description |
| --- | --- |
| [`one2many`](#ultralytics.nn.modules.head.OBB.one2many) | Returns the one-to-many head components, here for backward compatibility. |
| [`one2one`](#ultralytics.nn.modules.head.OBB.one2one) | Returns the one-to-one head components. |
| [`_inference`](#ultralytics.nn.modules.head.OBB._inference) | Decode predicted bounding boxes and class probabilities, concatenated with rotation angles. |
| [`decode_bboxes`](#ultralytics.nn.modules.head.OBB.decode_bboxes) | Decode rotated bounding boxes. |
| [`forward_head`](#ultralytics.nn.modules.head.OBB.forward_head) | Concatenates and returns predicted bounding boxes, class probabilities, and angles. |
| [`fuse`](#ultralytics.nn.modules.head.OBB.fuse) | Remove the one2many head for inference optimization. |
| [`postprocess`](#ultralytics.nn.modules.head.OBB.postprocess) | Post-process YOLO model predictions. |

**Examples**

```python
Create an OBB detection head
>>> obb = OBB(nc=80, ne=1, ch=(256, 512, 1024))
>>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
>>> outputs = obb(x)
```

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L417-L510"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Property `ultralytics.nn.modules.head.OBB.one2many` {#ultralytics.nn.modules.head.OBB.one2many}

```python
def one2many(self)
```

Returns the one-to-many head components, here for backward compatibility.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L457-L459"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def one2many(self):
    """Returns the one-to-many head components, here for backward compatibility."""
    return dict(box_head=self.cv2, cls_head=self.cv3, angle_head=self.cv4)
```
</details>

<br>

### Property `ultralytics.nn.modules.head.OBB.one2one` {#ultralytics.nn.modules.head.OBB.one2one}

```python
def one2one(self)
```

Returns the one-to-one head components.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L462-L464"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def one2one(self):
    """Returns the one-to-one head components."""
    return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, angle_head=self.one2one_cv4)
```
</details>

<br>

### Method `ultralytics.nn.modules.head.OBB._inference` {#ultralytics.nn.modules.head.OBB.\_inference}

```python
def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor
```

Decode predicted bounding boxes and class probabilities, concatenated with rotation angles.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `dict[str, torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L466-L471"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
    """Decode predicted bounding boxes and class probabilities, concatenated with rotation angles."""
    # For decode_bboxes convenience
    self.angle = x["angle"]  # TODO: need to test obb
    preds = super()._inference(x)
    return torch.cat([preds, x["angle"]], dim=1)
```
</details>

<br>

### Method `ultralytics.nn.modules.head.OBB.decode_bboxes` {#ultralytics.nn.modules.head.OBB.decode\_bboxes}

```python
def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor
```

Decode rotated bounding boxes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `bboxes` | `torch.Tensor` |  | *required* |
| `anchors` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L487-L489"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Decode rotated bounding boxes."""
    return dist2rbox(bboxes, self.angle, anchors, dim=1)
```
</details>

<br>

### Method `ultralytics.nn.modules.head.OBB.forward_head` {#ultralytics.nn.modules.head.OBB.forward\_head}

```python
def forward_head(
    self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, angle_head: torch.nn.Module
) -> dict[str, torch.Tensor]
```

Concatenates and returns predicted bounding boxes, class probabilities, and angles.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` |  | *required* |
| `box_head` | `torch.nn.Module` |  | *required* |
| `cls_head` | `torch.nn.Module` |  | *required* |
| `angle_head` | `torch.nn.Module` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L473-L485"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.OBB.fuse` {#ultralytics.nn.modules.head.OBB.fuse}

```python
def fuse(self) -> None
```

Remove the one2many head for inference optimization.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L508-L510"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def fuse(self) -> None:
    """Remove the one2many head for inference optimization."""
    self.cv2 = self.cv3 = self.cv4 = None
```
</details>

<br>

### Method `ultralytics.nn.modules.head.OBB.postprocess` {#ultralytics.nn.modules.head.OBB.postprocess}

```python
def postprocess(self, preds: torch.Tensor) -> torch.Tensor
```

Post-process YOLO model predictions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `torch.Tensor` | Raw predictions with shape (batch_size, num_anchors, 4 + nc + ne) with last dimension<br>    format [x, y, w, h, class_probs, angle]. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Processed predictions with shape (batch_size, min(max_det, num_anchors), 7) and last |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L491-L506"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.head.OBB26` {#ultralytics.nn.modules.head.OBB26}

```python
OBB26()
```

**Bases:** `OBB`

YOLO26 OBB detection head for detection with rotation models. This class extends the OBB head with modified angle

processing that outputs raw angle predictions without sigmoid transformation, compared to the original OBB class.

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `ne` | `int` | Number of extra parameters. |
| `cv4` | `nn.ModuleList` | Convolution layers for angle prediction. |
| `angle` | `torch.Tensor` | Predicted rotation angles. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward_head`](#ultralytics.nn.modules.head.OBB26.forward_head) | Concatenates and returns predicted bounding boxes, class probabilities, and raw angles. |

**Examples**

```python
Create an OBB26 detection head
>>> obb26 = OBB26(nc=80, ne=1, ch=(256, 512, 1024))
>>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
>>> outputs = obb26(x)
```

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L513-L544"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class OBB26(OBB):
```
</details>

<br>

### Method `ultralytics.nn.modules.head.OBB26.forward_head` {#ultralytics.nn.modules.head.OBB26.forward\_head}

```python
def forward_head(
    self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, angle_head: torch.nn.Module
) -> dict[str, torch.Tensor]
```

Concatenates and returns predicted bounding boxes, class probabilities, and raw angles.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` |  | *required* |
| `box_head` | `torch.nn.Module` |  | *required* |
| `cls_head` | `torch.nn.Module` |  | *required* |
| `angle_head` | `torch.nn.Module` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L533-L544"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.head.Pose` {#ultralytics.nn.modules.head.Pose}

```python
Pose(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max = 16, end2end = False, ch: tuple = ())
```

**Bases:** `Detect`

YOLO Pose head for keypoints models.

This class extends the Detect head to include keypoint prediction capabilities for pose estimation tasks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `nc` | `int` | Number of classes. | `80` |
| `kpt_shape` | `tuple` | Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible). | `(17, 3)` |
| `reg_max` | `int` | Maximum number of DFL channels. | `16` |
| `end2end` | `bool` | Whether to use end-to-end NMS-free detection. | `False` |
| `ch` | `tuple` | Tuple of channel sizes from backbone feature maps. | `()` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `kpt_shape` | `tuple` | Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible). |
| `nk` | `int` | Total number of keypoint values. |
| `cv4` | `nn.ModuleList` | Convolution layers for keypoint prediction. |

**Methods**

| Name | Description |
| --- | --- |
| [`one2many`](#ultralytics.nn.modules.head.Pose.one2many) | Returns the one-to-many head components, here for backward compatibility. |
| [`one2one`](#ultralytics.nn.modules.head.Pose.one2one) | Returns the one-to-one head components. |
| [`_inference`](#ultralytics.nn.modules.head.Pose._inference) | Decode predicted bounding boxes and class probabilities, concatenated with keypoints. |
| [`forward_head`](#ultralytics.nn.modules.head.Pose.forward_head) | Concatenates and returns predicted bounding boxes, class probabilities, and keypoints. |
| [`fuse`](#ultralytics.nn.modules.head.Pose.fuse) | Remove the one2many head for inference optimization. |
| [`kpts_decode`](#ultralytics.nn.modules.head.Pose.kpts_decode) | Decode keypoints from predictions. |
| [`postprocess`](#ultralytics.nn.modules.head.Pose.postprocess) | Post-process YOLO model predictions. |

**Examples**

```python
Create a pose detection head
>>> pose = Pose(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
>>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
>>> outputs = pose(x)
```

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L547-L652"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Property `ultralytics.nn.modules.head.Pose.one2many` {#ultralytics.nn.modules.head.Pose.one2many}

```python
def one2many(self)
```

Returns the one-to-many head components, here for backward compatibility.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L588-L590"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def one2many(self):
    """Returns the one-to-many head components, here for backward compatibility."""
    return dict(box_head=self.cv2, cls_head=self.cv3, pose_head=self.cv4)
```
</details>

<br>

### Property `ultralytics.nn.modules.head.Pose.one2one` {#ultralytics.nn.modules.head.Pose.one2one}

```python
def one2one(self)
```

Returns the one-to-one head components.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L593-L595"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def one2one(self):
    """Returns the one-to-one head components."""
    return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, pose_head=self.one2one_cv4)
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Pose._inference` {#ultralytics.nn.modules.head.Pose.\_inference}

```python
def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor
```

Decode predicted bounding boxes and class probabilities, concatenated with keypoints.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `dict[str, torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L597-L600"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
    """Decode predicted bounding boxes and class probabilities, concatenated with keypoints."""
    preds = super()._inference(x)
    return torch.cat([preds, self.kpts_decode(x["kpts"])], dim=1)
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Pose.forward_head` {#ultralytics.nn.modules.head.Pose.forward\_head}

```python
def forward_head(
    self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, pose_head: torch.nn.Module
) -> dict[str, torch.Tensor]
```

Concatenates and returns predicted bounding boxes, class probabilities, and keypoints.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` |  | *required* |
| `box_head` | `torch.nn.Module` |  | *required* |
| `cls_head` | `torch.nn.Module` |  | *required* |
| `pose_head` | `torch.nn.Module` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L602-L610"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward_head(
    self, x: list[torch.Tensor], box_head: torch.nn.Module, cls_head: torch.nn.Module, pose_head: torch.nn.Module
) -> dict[str, torch.Tensor]:
    """Concatenates and returns predicted bounding boxes, class probabilities, and keypoints."""
    preds = super().forward_head(x, box_head, cls_head)
    if pose_head is not None:
        bs = x[0].shape[0]  # batch size
        preds["kpts"] = torch.cat([pose_head[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], 2)
    return preds
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Pose.fuse` {#ultralytics.nn.modules.head.Pose.fuse}

```python
def fuse(self) -> None
```

Remove the one2many head for inference optimization.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L629-L631"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def fuse(self) -> None:
    """Remove the one2many head for inference optimization."""
    self.cv2 = self.cv3 = self.cv4 = None
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Pose.kpts_decode` {#ultralytics.nn.modules.head.Pose.kpts\_decode}

```python
def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor
```

Decode keypoints from predictions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `kpts` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L633-L652"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Pose.postprocess` {#ultralytics.nn.modules.head.Pose.postprocess}

```python
def postprocess(self, preds: torch.Tensor) -> torch.Tensor
```

Post-process YOLO model predictions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `torch.Tensor` | Raw predictions with shape (batch_size, num_anchors, 4 + nc + nk) with last dimension<br>    format [x1, y1, x2, y2, class_probs, keypoints]. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + self.nk) and |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L612-L627"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.head.Pose26` {#ultralytics.nn.modules.head.Pose26}

```python
Pose26(self, nc: int = 80, kpt_shape: tuple = (17, 3), reg_max = 16, end2end = False, ch: tuple = ())
```

**Bases:** `Pose`

YOLO26 Pose head for keypoints models.

This class extends the Pose head with normalizing flow for keypoint prediction in pose estimation tasks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `nc` | `int` | Number of classes. | `80` |
| `kpt_shape` | `tuple` | Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible). | `(17, 3)` |
| `reg_max` | `int` | Maximum number of DFL channels. | `16` |
| `end2end` | `bool` | Whether to use end-to-end NMS-free detection. | `False` |
| `ch` | `tuple` | Tuple of channel sizes from backbone feature maps. | `()` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `kpt_shape` | `tuple` | Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible). |
| `nk` | `int` | Total number of keypoint values. |
| `cv4` | `nn.ModuleList` | Convolution layers for keypoint prediction. |

**Methods**

| Name | Description |
| --- | --- |
| [`one2many`](#ultralytics.nn.modules.head.Pose26.one2many) | Returns the one-to-many head components, here for backward compatibility. |
| [`one2one`](#ultralytics.nn.modules.head.Pose26.one2one) | Returns the one-to-one head components. |
| [`forward_head`](#ultralytics.nn.modules.head.Pose26.forward_head) | Concatenates and returns predicted bounding boxes, class probabilities, and keypoints. |
| [`fuse`](#ultralytics.nn.modules.head.Pose26.fuse) | Remove the one2many head for inference optimization. |
| [`kpts_decode`](#ultralytics.nn.modules.head.Pose26.kpts_decode) | Decode keypoints from predictions. |

**Examples**

```python
Create a pose detection head
>>> pose = Pose26(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
>>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
>>> outputs = pose(x)
```

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L655-L769"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Property `ultralytics.nn.modules.head.Pose26.one2many` {#ultralytics.nn.modules.head.Pose26.one2many}

```python
def one2many(self)
```

Returns the one-to-many head components, here for backward compatibility.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L702-L710"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Property `ultralytics.nn.modules.head.Pose26.one2one` {#ultralytics.nn.modules.head.Pose26.one2one}

```python
def one2one(self)
```

Returns the one-to-one head components.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L713-L721"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Pose26.forward_head` {#ultralytics.nn.modules.head.Pose26.forward\_head}

```python
def forward_head(
    self,
    x: list[torch.Tensor],
    box_head: torch.nn.Module,
    cls_head: torch.nn.Module,
    pose_head: torch.nn.Module,
    kpts_head: torch.nn.Module,
    kpts_sigma_head: torch.nn.Module,
) -> dict[str, torch.Tensor]
```

Concatenates and returns predicted bounding boxes, class probabilities, and keypoints.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` |  | *required* |
| `box_head` | `torch.nn.Module` |  | *required* |
| `cls_head` | `torch.nn.Module` |  | *required* |
| `pose_head` | `torch.nn.Module` |  | *required* |
| `kpts_head` | `torch.nn.Module` |  | *required* |
| `kpts_sigma_head` | `torch.nn.Module` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L723-L742"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Pose26.fuse` {#ultralytics.nn.modules.head.Pose26.fuse}

```python
def fuse(self) -> None
```

Remove the one2many head for inference optimization.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L744-L747"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def fuse(self) -> None:
    """Remove the one2many head for inference optimization."""
    super().fuse()
    self.cv4_kpts = self.cv4_sigma = self.flow_model = self.one2one_cv4_sigma = None
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Pose26.kpts_decode` {#ultralytics.nn.modules.head.Pose26.kpts\_decode}

```python
def kpts_decode(self, kpts: torch.Tensor) -> torch.Tensor
```

Decode keypoints from predictions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `kpts` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L749-L769"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.head.Classify` {#ultralytics.nn.modules.head.Classify}

```python
Classify(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1)
```

**Bases:** `nn.Module`

YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2).

This class implements a classification head that transforms feature maps into class predictions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `c1` | `int` | Number of input channels. | *required* |
| `c2` | `int` | Number of output classes. | *required* |
| `k` | `int` | Kernel size. | `1` |
| `s` | `int` | Stride. | `1` |
| `p` | `int, optional` | Padding. | `None` |
| `g` | `int` | Groups. | `1` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `export` | `bool` | Export mode flag. |
| `conv` | `Conv` | Convolutional layer for feature transformation. |
| `pool` | `nn.AdaptiveAvgPool2d` | Global average pooling layer. |
| `drop` | `nn.Dropout` | Dropout layer for regularization. |
| `linear` | `nn.Linear` | Linear layer for final classification. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.head.Classify.forward) | Perform forward pass on input feature maps. |

**Examples**

```python
Create a classification head
>>> classify = Classify(c1=1024, c2=1000)
>>> x = torch.randn(1, 1024, 20, 20)
>>> output = classify(x)
```

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L772-L822"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.Classify.forward` {#ultralytics.nn.modules.head.Classify.forward}

```python
def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor | tuple
```

Perform forward pass on input feature maps.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor] | torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L814-L822"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor | tuple:
    """Perform forward pass on input feature maps."""
    if isinstance(x, list):
        x = torch.cat(x, 1)
    x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
    if self.training:
        return x
    y = x.softmax(1)  # get final output
    return y if self.export else (y, x)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.head.WorldDetect` {#ultralytics.nn.modules.head.WorldDetect}

```python
def __init__(
    self,
    nc: int = 80,
    embed: int = 512,
    with_bn: bool = False,
    reg_max: int = 16,
    end2end: bool = False,
    ch: tuple = (),
)
```

**Bases:** `Detect`

Head for integrating YOLO detection models with semantic understanding from text embeddings.

This class extends the standard Detect head to incorporate text embeddings for enhanced semantic understanding in object detection tasks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `nc` | `int` | Number of classes. | `80` |
| `embed` | `int` | Embedding dimension. | `512` |
| `with_bn` | `bool` | Whether to use batch normalization in contrastive head. | `False` |
| `reg_max` | `int` | Maximum number of DFL channels. | `16` |
| `end2end` | `bool` | Whether to use end-to-end NMS-free detection. | `False` |
| `ch` | `tuple` | Tuple of channel sizes from backbone feature maps. | `()` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `cv3` | `nn.ModuleList` | Convolution layers for embedding features. |
| `cv4` | `nn.ModuleList` | Contrastive head layers for text-vision alignment. |

**Methods**

| Name | Description |
| --- | --- |
| [`bias_init`](#ultralytics.nn.modules.head.WorldDetect.bias_init) | Initialize Detect() biases, WARNING: requires stride availability. |
| [`forward`](#ultralytics.nn.modules.head.WorldDetect.forward) | Concatenate and return predicted bounding boxes and class probabilities. |

**Examples**

```python
Create a WorldDetect head
>>> world_detect = WorldDetect(nc=80, embed=512, with_bn=False, ch=(256, 512, 1024))
>>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
>>> text = torch.randn(1, 80, 512)
>>> outputs = world_detect(x, text)
```

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L825-L892"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.WorldDetect.bias_init` {#ultralytics.nn.modules.head.WorldDetect.bias\_init}

```python
def bias_init(self)
```

Initialize Detect() biases, WARNING: requires stride availability.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L886-L892"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def bias_init(self):
    """Initialize Detect() biases, WARNING: requires stride availability."""
    m = self  # self.model[-1]  # Detect() module
    # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
    # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
    for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
        a[-1].bias.data[:] = 1.0  # box
```
</details>

<br>

### Method `ultralytics.nn.modules.head.WorldDetect.forward` {#ultralytics.nn.modules.head.WorldDetect.forward}

```python
def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> dict[str, torch.Tensor] | tuple
```

Concatenate and return predicted bounding boxes and class probabilities.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` |  | *required* |
| `text` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L871-L884"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.head.LRPCHead` {#ultralytics.nn.modules.head.LRPCHead}

```python
LRPCHead(self, vocab: nn.Module, pf: nn.Module, loc: nn.Module, enabled: bool = True)
```

**Bases:** `nn.Module`

Lightweight Region Proposal and Classification Head for efficient object detection.

This head combines region proposal filtering with classification to enable efficient detection with dynamic vocabulary support.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `vocab` | `nn.Module` | Vocabulary/classification module. | *required* |
| `pf` | `nn.Module` | Proposal filter module. | *required* |
| `loc` | `nn.Module` | Localization module. | *required* |
| `enabled` | `bool` | Whether to enable the head functionality. | `True` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `vocab` | `nn.Module` | Vocabulary/classification layer. |
| `pf` | `nn.Module` | Proposal filter module. |
| `loc` | `nn.Module` | Localization module. |
| `enabled` | `bool` | Whether the head is enabled. |

**Methods**

| Name | Description |
| --- | --- |
| [`conv2linear`](#ultralytics.nn.modules.head.LRPCHead.conv2linear) | Convert a 1x1 convolutional layer to a linear layer. |
| [`forward`](#ultralytics.nn.modules.head.LRPCHead.forward) | Process classification and localization features to generate detection proposals. |

**Examples**

```python
Create an LRPC head
>>> vocab = nn.Conv2d(256, 80, 1)
>>> pf = nn.Conv2d(256, 1, 1)
>>> loc = nn.Conv2d(256, 4, 1)
>>> head = LRPCHead(vocab, pf, loc, enabled=True)
```

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L896-L959"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.LRPCHead.conv2linear` {#ultralytics.nn.modules.head.LRPCHead.conv2linear}

```python
def conv2linear(conv: nn.Conv2d) -> nn.Linear
```

Convert a 1x1 convolutional layer to a linear layer.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `conv` | `nn.Conv2d` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L936-L942"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@staticmethod
def conv2linear(conv: nn.Conv2d) -> nn.Linear:
    """Convert a 1x1 convolutional layer to a linear layer."""
    assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
    linear = nn.Linear(conv.in_channels, conv.out_channels)
    linear.weight.data = conv.weight.view(conv.out_channels, -1).data
    linear.bias.data = conv.bias.data
    return linear
```
</details>

<br>

### Method `ultralytics.nn.modules.head.LRPCHead.forward` {#ultralytics.nn.modules.head.LRPCHead.forward}

```python
def forward(self, cls_feat: torch.Tensor, loc_feat: torch.Tensor, conf: float) -> tuple[tuple, torch.Tensor]
```

Process classification and localization features to generate detection proposals.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cls_feat` | `torch.Tensor` |  | *required* |
| `loc_feat` | `torch.Tensor` |  | *required* |
| `conf` | `float` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L944-L959"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.head.YOLOEDetect` {#ultralytics.nn.modules.head.YOLOEDetect}

```python
YOLOEDetect(self, nc: int = 80, embed: int = 512, with_bn: bool = False, reg_max = 16, end2end = False, ch: tuple = ())
```

**Bases:** `Detect`

Head for integrating YOLO detection models with semantic understanding from text embeddings.

This class extends the standard Detect head to support text-guided detection with enhanced semantic understanding through text embeddings and visual prompt embeddings.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `nc` | `int` | Number of classes. | `80` |
| `embed` | `int` | Embedding dimension. | `512` |
| `with_bn` | `bool` | Whether to use batch normalization in contrastive head. | `False` |
| `reg_max` | `int` | Maximum number of DFL channels. | `16` |
| `end2end` | `bool` | Whether to use end-to-end NMS-free detection. | `False` |
| `ch` | `tuple` | Tuple of channel sizes from backbone feature maps. | `()` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `is_fused` | `bool` | Whether the model is fused for inference. |
| `cv3` | `nn.ModuleList` | Convolution layers for embedding features. |
| `cv4` | `nn.ModuleList` | Contrastive head layers for text-vision alignment. |
| `reprta` | `Residual` | Residual block for text prompt embeddings. |
| `savpe` | `SAVPE` | Spatial-aware visual prompt embeddings module. |
| `embed` | `int` | Embedding dimension. |

**Methods**

| Name | Description |
| --- | --- |
| [`one2many`](#ultralytics.nn.modules.head.YOLOEDetect.one2many) | Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility. |
| [`one2one`](#ultralytics.nn.modules.head.YOLOEDetect.one2one) | Returns the one-to-one head components. |
| [`_fuse_tp`](#ultralytics.nn.modules.head.YOLOEDetect._fuse_tp) | Fuse text prompt embeddings with model weights for efficient inference. |
| [`_get_decode_boxes`](#ultralytics.nn.modules.head.YOLOEDetect._get_decode_boxes) | Decode predicted bounding boxes for inference. |
| [`bias_init`](#ultralytics.nn.modules.head.YOLOEDetect.bias_init) | Initialize Detect() biases, WARNING: requires stride availability. |
| [`forward`](#ultralytics.nn.modules.head.YOLOEDetect.forward) | Process features with class prompt embeddings to generate detections. |
| [`forward_head`](#ultralytics.nn.modules.head.YOLOEDetect.forward_head) | Concatenates and returns predicted bounding boxes, class probabilities, and contrastive scores. |
| [`forward_lrpc`](#ultralytics.nn.modules.head.YOLOEDetect.forward_lrpc) | Process features with fused text embeddings to generate detections for prompt-free model. |
| [`fuse`](#ultralytics.nn.modules.head.YOLOEDetect.fuse) | Fuse text features with model weights for efficient inference. |
| [`get_tpe`](#ultralytics.nn.modules.head.YOLOEDetect.get_tpe) | Get text prompt embeddings with normalization. |
| [`get_vpe`](#ultralytics.nn.modules.head.YOLOEDetect.get_vpe) | Get visual prompt embeddings with spatial awareness. |

**Examples**

```python
Create a YOLOEDetect head
>>> yoloe_detect = YOLOEDetect(nc=80, embed=512, with_bn=True, ch=(256, 512, 1024))
>>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
>>> cls_pe = torch.randn(1, 80, 512)
>>> outputs = yoloe_detect(x, cls_pe)
```

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L962-L1175"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Property `ultralytics.nn.modules.head.YOLOEDetect.one2many` {#ultralytics.nn.modules.head.YOLOEDetect.one2many}

```python
def one2many(self)
```

Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1138-L1140"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def one2many(self):
    """Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility."""
    return dict(box_head=self.cv2, cls_head=self.cv3, contrastive_head=self.cv4)
```
</details>

<br>

### Property `ultralytics.nn.modules.head.YOLOEDetect.one2one` {#ultralytics.nn.modules.head.YOLOEDetect.one2one}

```python
def one2one(self)
```

Returns the one-to-one head components.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1143-L1145"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def one2one(self):
    """Returns the one-to-one head components."""
    return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3, contrastive_head=self.one2one_cv4)
```
</details>

<br>

### Method `ultralytics.nn.modules.head.YOLOEDetect._fuse_tp` {#ultralytics.nn.modules.head.YOLOEDetect.\_fuse\_tp}

```python
def _fuse_tp(self, txt_feats: torch.Tensor, cls_head: torch.nn.Module, bn_head: torch.nn.Module) -> None
```

Fuse text prompt embeddings with model weights for efficient inference.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `txt_feats` | `torch.Tensor` |  | *required* |
| `cls_head` | `torch.nn.Module` |  | *required* |
| `bn_head` | `torch.nn.Module` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1050-L1085"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.YOLOEDetect._get_decode_boxes` {#ultralytics.nn.modules.head.YOLOEDetect.\_get\_decode\_boxes}

```python
def _get_decode_boxes(self, x)
```

Decode predicted bounding boxes for inference.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1130-L1135"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _get_decode_boxes(self, x):
    """Decode predicted bounding boxes for inference."""
    dbox = super()._get_decode_boxes(x)
    if hasattr(self, "lrpc"):
        dbox = dbox if self.export and not self.dynamic else dbox[..., x["index"]]
    return dbox
```
</details>

<br>

### Method `ultralytics.nn.modules.head.YOLOEDetect.bias_init` {#ultralytics.nn.modules.head.YOLOEDetect.bias\_init}

```python
def bias_init(self)
```

Initialize Detect() biases, WARNING: requires stride availability.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1161-L1175"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.YOLOEDetect.forward` {#ultralytics.nn.modules.head.YOLOEDetect.forward}

```python
def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple
```

Process features with class prompt embeddings to generate detections.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1100-L1104"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
    """Process features with class prompt embeddings to generate detections."""
    if hasattr(self, "lrpc"):  # for prompt-free inference
        return self.forward_lrpc(x[:3])
    return super().forward(x)
```
</details>

<br>

### Method `ultralytics.nn.modules.head.YOLOEDetect.forward_head` {#ultralytics.nn.modules.head.YOLOEDetect.forward\_head}

```python
def forward_head(self, x, box_head, cls_head, contrastive_head)
```

Concatenates and returns predicted bounding boxes, class probabilities, and contrastive scores.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` |  |  | *required* |
| `box_head` |  |  | *required* |
| `cls_head` |  |  | *required* |
| `contrastive_head` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1147-L1159"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.YOLOEDetect.forward_lrpc` {#ultralytics.nn.modules.head.YOLOEDetect.forward\_lrpc}

```python
def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple
```

Process features with fused text embeddings to generate detections for prompt-free model.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1106-L1128"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.YOLOEDetect.fuse` {#ultralytics.nn.modules.head.YOLOEDetect.fuse}

```python
def fuse(self, txt_feats: torch.Tensor = None)
```

Fuse text features with model weights for efficient inference.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `txt_feats` | `torch.Tensor` |  | `None` |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1033-L1048"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.YOLOEDetect.get_tpe` {#ultralytics.nn.modules.head.YOLOEDetect.get\_tpe}

```python
def get_tpe(self, tpe: torch.Tensor | None) -> torch.Tensor | None
```

Get text prompt embeddings with normalization.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `tpe` | `torch.Tensor | None` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1087-L1089"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_tpe(self, tpe: torch.Tensor | None) -> torch.Tensor | None:
    """Get text prompt embeddings with normalization."""
    return None if tpe is None else F.normalize(self.reprta(tpe), dim=-1, p=2)
```
</details>

<br>

### Method `ultralytics.nn.modules.head.YOLOEDetect.get_vpe` {#ultralytics.nn.modules.head.YOLOEDetect.get\_vpe}

```python
def get_vpe(self, x: list[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor
```

Get visual prompt embeddings with spatial awareness.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` |  | *required* |
| `vpe` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1091-L1098"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_vpe(self, x: list[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
    """Get visual prompt embeddings with spatial awareness."""
    if vpe.shape[1] == 0:  # no visual prompt embeddings
        return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device)
    if vpe.ndim == 4:  # (B, N, H, W)
        vpe = self.savpe(x, vpe)
    assert vpe.ndim == 3  # (B, N, D)
    return vpe
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.head.YOLOESegment` {#ultralytics.nn.modules.head.YOLOESegment}

```python
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
)
```

**Bases:** `YOLOEDetect`

YOLO segmentation head with text embedding capabilities.

This class extends YOLOEDetect to include mask prediction capabilities for instance segmentation tasks with text-guided semantic understanding.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `nc` | `int` | Number of classes. | `80` |
| `nm` | `int` | Number of masks. | `32` |
| `npr` | `int` | Number of protos. | `256` |
| `embed` | `int` | Embedding dimension. | `512` |
| `with_bn` | `bool` | Whether to use batch normalization in contrastive head. | `False` |
| `reg_max` | `int` | Maximum number of DFL channels. | `16` |
| `end2end` | `bool` | Whether to use end-to-end NMS-free detection. | `False` |
| `ch` | `tuple` | Tuple of channel sizes from backbone feature maps. | `()` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `nm` | `int` | Number of masks. |
| `npr` | `int` | Number of protos. |
| `proto` | `Proto` | Prototype generation module. |
| `cv5` | `nn.ModuleList` | Convolution layers for mask coefficients. |

**Methods**

| Name | Description |
| --- | --- |
| [`one2many`](#ultralytics.nn.modules.head.YOLOESegment.one2many) | Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility. |
| [`one2one`](#ultralytics.nn.modules.head.YOLOESegment.one2one) | Returns the one-to-one head components. |
| [`_inference`](#ultralytics.nn.modules.head.YOLOESegment._inference) | Decode predicted bounding boxes and class probabilities, concatenated with mask coefficients. |
| [`forward`](#ultralytics.nn.modules.head.YOLOESegment.forward) | Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients. |
| [`forward_head`](#ultralytics.nn.modules.head.YOLOESegment.forward_head) | Concatenates and returns predicted bounding boxes, class probabilities, and mask coefficients. |
| [`forward_lrpc`](#ultralytics.nn.modules.head.YOLOESegment.forward_lrpc) | Process features with fused text embeddings to generate detections for prompt-free model. |
| [`fuse`](#ultralytics.nn.modules.head.YOLOESegment.fuse) | Fuse text features with model weights for efficient inference. |
| [`postprocess`](#ultralytics.nn.modules.head.YOLOESegment.postprocess) | Post-process YOLO model predictions. |

**Examples**

```python
Create a YOLOESegment head
>>> yoloe_segment = YOLOESegment(nc=80, nm=32, npr=256, embed=512, with_bn=True, ch=(256, 512, 1024))
>>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
>>> text = torch.randn(1, 80, 512)
>>> outputs = yoloe_segment(x, text)
```

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1178-L1341"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Property `ultralytics.nn.modules.head.YOLOESegment.one2many` {#ultralytics.nn.modules.head.YOLOESegment.one2many}

```python
def one2many(self)
```

Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1235-L1237"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def one2many(self):
    """Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility."""
    return dict(box_head=self.cv2, cls_head=self.cv3, mask_head=self.cv5, contrastive_head=self.cv4)
```
</details>

<br>

### Property `ultralytics.nn.modules.head.YOLOESegment.one2one` {#ultralytics.nn.modules.head.YOLOESegment.one2one}

```python
def one2one(self)
```

Returns the one-to-one head components.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1240-L1247"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@property
def one2one(self):
    """Returns the one-to-one head components."""
    return dict(
        box_head=self.one2one_cv2,
        cls_head=self.one2one_cv3,
        mask_head=self.one2one_cv5,
        contrastive_head=self.one2one_cv4,
    )
```
</details>

<br>

### Method `ultralytics.nn.modules.head.YOLOESegment._inference` {#ultralytics.nn.modules.head.YOLOESegment.\_inference}

```python
def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor
```

Decode predicted bounding boxes and class probabilities, concatenated with mask coefficients.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `dict[str, torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1297-L1300"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
    """Decode predicted bounding boxes and class probabilities, concatenated with mask coefficients."""
    preds = super()._inference(x)
    return torch.cat([preds, x["mask_coefficient"]], dim=1)
```
</details>

<br>

### Method `ultralytics.nn.modules.head.YOLOESegment.forward` {#ultralytics.nn.modules.head.YOLOESegment.forward}

```python
def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]
```

Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1282-L1295"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.YOLOESegment.forward_head` {#ultralytics.nn.modules.head.YOLOESegment.forward\_head}

```python
def forward_head(
    self,
    x: list[torch.Tensor],
    box_head: torch.nn.Module,
    cls_head: torch.nn.Module,
    mask_head: torch.nn.Module,
    contrastive_head: torch.nn.Module,
) -> dict[str, torch.Tensor]
```

Concatenates and returns predicted bounding boxes, class probabilities, and mask coefficients.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` |  | *required* |
| `box_head` | `torch.nn.Module` |  | *required* |
| `cls_head` | `torch.nn.Module` |  | *required* |
| `mask_head` | `torch.nn.Module` |  | *required* |
| `contrastive_head` | `torch.nn.Module` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1302-L1315"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.YOLOESegment.forward_lrpc` {#ultralytics.nn.modules.head.YOLOESegment.forward\_lrpc}

```python
def forward_lrpc(self, x: list[torch.Tensor]) -> torch.Tensor | tuple
```

Process features with fused text embeddings to generate detections for prompt-free model.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1249-L1280"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.YOLOESegment.fuse` {#ultralytics.nn.modules.head.YOLOESegment.fuse}

```python
def fuse(self, txt_feats: torch.Tensor = None)
```

Fuse text features with model weights for efficient inference.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `txt_feats` | `torch.Tensor` |  | `None` |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1334-L1341"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def fuse(self, txt_feats: torch.Tensor = None):
    """Fuse text features with model weights for efficient inference."""
    super().fuse(txt_feats)
    if txt_feats is None:  # means eliminate one2many branch
        self.cv5 = None
        if hasattr(self.proto, "fuse"):
            self.proto.fuse()
        return
```
</details>

<br>

### Method `ultralytics.nn.modules.head.YOLOESegment.postprocess` {#ultralytics.nn.modules.head.YOLOESegment.postprocess}

```python
def postprocess(self, preds: torch.Tensor) -> torch.Tensor
```

Post-process YOLO model predictions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `preds` | `torch.Tensor` | Raw predictions with shape (batch_size, num_anchors, 4 + nc + nm) with last dimension<br>    format [x1, y1, x2, y2, class_probs, mask_coefficient]. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Processed predictions with shape (batch_size, min(max_det, num_anchors), 6 + nm) and last |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1317-L1332"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.head.YOLOESegment26` {#ultralytics.nn.modules.head.YOLOESegment26}

```python
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
)
```

**Bases:** `YOLOESegment`

YOLOE-style segmentation head module using Proto26 for mask generation.

This class extends the YOLOESegment functionality to include segmentation capabilities by integrating a Proto26 generation module and convolutional layers to predict mask coefficients.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `nc` | `int` | Number of classes. Defaults to 80. | `80` |
| `nm` | `int` | Number of masks. Defaults to 32. | `32` |
| `npr` | `int` | Number of prototype channels. Defaults to 256. | `256` |
| `embed` | `int` | Embedding dimensionality. Defaults to 512. | `512` |
| `with_bn` | `bool` | Whether to use Batch Normalization. Defaults to False. | `False` |
| `reg_max` | `int` | Maximum number of DFL channels. Defaults to 16. | `16` |
| `end2end` | `bool` | Whether to use end-to-end detection mode. Defaults to False. | `False` |
| `ch` | `tuple[int, ...]` | Input channels for each scale. | `()` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `nm` | `int` | Number of segmentation masks. |
| `npr` | `int` | Number of prototype channels. |
| `proto` | `Proto26` | Prototype generation module for segmentation. |
| `cv5` | `nn.ModuleList` | Convolutional layers for generating mask coefficients from features. |
| `one2one_cv5` | `nn.ModuleList, optional` | Deep copy of cv5 for end-to-end detection branches. |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.nn.modules.head.YOLOESegment26.forward) | Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1344-L1404"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.YOLOESegment26.forward` {#ultralytics.nn.modules.head.YOLOESegment26.forward}

```python
def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor] | dict[str, torch.Tensor]
```

Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1390-L1404"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.head.RTDETRDecoder` {#ultralytics.nn.modules.head.RTDETRDecoder}

```python
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
)
```

**Bases:** `nn.Module`

Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes and class labels for objects in an image. It integrates features from multiple layers and runs through a series of Transformer decoder layers to output the final predictions.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `nc` | `int` | Number of classes. | `80` |
| `ch` | `tuple` | Channels in the backbone feature maps. | `(512, 1024, 2048)` |
| `hd` | `int` | Dimension of hidden layers. | `256` |
| `nq` | `int` | Number of query points. | `300` |
| `ndp` | `int` | Number of decoder points. | `4` |
| `nh` | `int` | Number of heads in multi-head attention. | `8` |
| `ndl` | `int` | Number of decoder layers. | `6` |
| `d_ffn` | `int` | Dimension of the feed-forward networks. | `1024` |
| `dropout` | `float` | Dropout rate. | `0.0` |
| `act` | `nn.Module` | Activation function. | `nn.ReLU()` |
| `eval_idx` | `int` | Evaluation index. | `-1` |
| `nd` | `int` | Number of denoising. | `100` |
| `label_noise_ratio` | `float` | Label noise ratio. | `0.5` |
| `box_noise_scale` | `float` | Box noise scale. | `1.0` |
| `learnt_init_query` | `bool` | Whether to learn initial query embeddings. | `False` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `export` | `bool` | Export mode flag. |
| `hidden_dim` | `int` | Dimension of hidden layers. |
| `nhead` | `int` | Number of heads in multi-head attention. |
| `nl` | `int` | Number of feature levels. |
| `nc` | `int` | Number of classes. |
| `num_queries` | `int` | Number of query points. |
| `num_decoder_layers` | `int` | Number of decoder layers. |
| `input_proj` | `nn.ModuleList` | Input projection layers for backbone features. |
| `decoder` | `DeformableTransformerDecoder` | Transformer decoder module. |
| `denoising_class_embed` | `nn.Embedding` | Class embeddings for denoising. |
| `num_denoising` | `int` | Number of denoising queries. |
| `label_noise_ratio` | `float` | Label noise ratio for training. |
| `box_noise_scale` | `float` | Box noise scale for training. |
| `learnt_init_query` | `bool` | Whether to learn initial query embeddings. |
| `tgt_embed` | `nn.Embedding` | Target embeddings for queries. |
| `query_pos_head` | `MLP` | Query position head. |
| `enc_output` | `nn.Sequential` | Encoder output layers. |
| `enc_score_head` | `nn.Linear` | Encoder score prediction head. |
| `enc_bbox_head` | `MLP` | Encoder bbox prediction head. |
| `dec_score_head` | `nn.ModuleList` | Decoder score prediction heads. |
| `dec_bbox_head` | `nn.ModuleList` | Decoder bbox prediction heads. |

**Methods**

| Name | Description |
| --- | --- |
| [`_generate_anchors`](#ultralytics.nn.modules.head.RTDETRDecoder._generate_anchors) | Generate anchor bounding boxes for given shapes with specific grid size and validate them. |
| [`_get_decoder_input`](#ultralytics.nn.modules.head.RTDETRDecoder._get_decoder_input) | Generate and prepare the input required for the decoder from the provided features and shapes. |
| [`_get_encoder_input`](#ultralytics.nn.modules.head.RTDETRDecoder._get_encoder_input) | Process and return encoder inputs by getting projection features from input and concatenating them. |
| [`_reset_parameters`](#ultralytics.nn.modules.head.RTDETRDecoder._reset_parameters) | Initialize or reset the parameters of the model's various components with predefined weights and biases. |
| [`forward`](#ultralytics.nn.modules.head.RTDETRDecoder.forward) | Run the forward pass of the module, returning bounding box and classification scores for the input. |

**Examples**

```python
Create an RTDETRDecoder
>>> decoder = RTDETRDecoder(nc=80, ch=(512, 1024, 2048), hd=256, nq=300)
>>> x = [torch.randn(1, 512, 64, 64), torch.randn(1, 1024, 32, 32), torch.randn(1, 2048, 16, 16)]
>>> outputs = decoder(x)
```

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1407-L1726"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.RTDETRDecoder._generate_anchors` {#ultralytics.nn.modules.head.RTDETRDecoder.\_generate\_anchors}

```python
def _generate_anchors(
    shapes: list[list[int]],
    grid_size: float = 0.05,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    eps: float = 1e-2,
) -> tuple[torch.Tensor, torch.Tensor]
```

Generate anchor bounding boxes for given shapes with specific grid size and validate them.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `shapes` | `list` | List of feature map shapes. | *required* |
| `grid_size` | `float, optional` | Base size of grid cells. | `0.05` |
| `dtype` | `torch.dtype, optional` | Data type for tensors. | `torch.float32` |
| `device` | `str, optional` | Device to create tensors on. | `"cpu"` |
| `eps` | `float, optional` | Small value for numerical stability. | `1e-2` |

**Returns**

| Type | Description |
| --- | --- |
| `anchors (torch.Tensor)` | Generated anchor boxes. |
| `valid_mask (torch.Tensor)` | Valid mask for anchors. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1581-L1617"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.RTDETRDecoder._get_decoder_input` {#ultralytics.nn.modules.head.RTDETRDecoder.\_get\_decoder\_input}

```python
def _get_decoder_input(
    self,
    feats: torch.Tensor,
    shapes: list[list[int]],
    dn_embed: torch.Tensor | None = None,
    dn_bbox: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

Generate and prepare the input required for the decoder from the provided features and shapes.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `feats` | `torch.Tensor` | Processed features from encoder. | *required* |
| `shapes` | `list` | List of feature map shapes. | *required* |
| `dn_embed` | `torch.Tensor, optional` | Denoising embeddings. | `None` |
| `dn_bbox` | `torch.Tensor, optional` | Denoising bounding boxes. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `embeddings (torch.Tensor)` | Query embeddings for decoder. |
| `refer_bbox (torch.Tensor)` | Reference bounding boxes. |
| `enc_bboxes (torch.Tensor)` | Encoded bounding boxes. |
| `enc_scores (torch.Tensor)` | Encoded scores. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1645-L1702"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.RTDETRDecoder._get_encoder_input` {#ultralytics.nn.modules.head.RTDETRDecoder.\_get\_encoder\_input}

```python
def _get_encoder_input(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, list[list[int]]]
```

Process and return encoder inputs by getting projection features from input and concatenating them.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` | List of feature maps from the backbone. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `feats (torch.Tensor)` | Processed features. |
| `shapes (list)` | List of feature map shapes. |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1619-L1643"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.RTDETRDecoder._reset_parameters` {#ultralytics.nn.modules.head.RTDETRDecoder.\_reset\_parameters}

```python
def _reset_parameters(self)
```

Initialize or reset the parameters of the model's various components with predefined weights and biases.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1704-L1726"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.RTDETRDecoder.forward` {#ultralytics.nn.modules.head.RTDETRDecoder.forward}

```python
def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor
```

Run the forward pass of the module, returning bounding box and classification scores for the input.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `list[torch.Tensor]` | List of feature maps from the backbone. | *required* |
| `batch` | `dict, optional` | Batch information for training. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `outputs (tuple | torch.Tensor)` | During training, returns a tuple of bounding boxes, scores, and other |

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1531-L1578"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.modules.head.v10Detect` {#ultralytics.nn.modules.head.v10Detect}

```python
v10Detect(self, nc: int = 80, ch: tuple = ())
```

**Bases:** `Detect`

v10 Detection head from https://arxiv.org/pdf/2405.14458.

This class implements the YOLOv10 detection head with dual-assignment training and consistent dual predictions for improved efficiency and performance.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `nc` | `int` | Number of classes. | `80` |
| `ch` | `tuple` | Tuple of channel sizes from backbone feature maps. | `()` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `end2end` | `bool` | End-to-end detection mode. |
| `max_det` | `int` | Maximum number of detections. |
| `cv3` | `nn.ModuleList` | Light classification head layers. |
| `one2one_cv3` | `nn.ModuleList` | One-to-one classification head layers. |

**Methods**

| Name | Description |
| --- | --- |
| [`fuse`](#ultralytics.nn.modules.head.v10Detect.fuse) | Remove the one2many head for inference optimization. |

**Examples**

```python
Create a v10Detect head
>>> v10_detect = v10Detect(nc=80, ch=(256, 512, 1024))
>>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
>>> outputs = v10_detect(x)
```

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1729-L1778"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
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
```
</details>

<br>

### Method `ultralytics.nn.modules.head.v10Detect.fuse` {#ultralytics.nn.modules.head.v10Detect.fuse}

```python
def fuse(self)
```

Remove the one2many head for inference optimization.

<details>
<summary>Source code in <code>ultralytics/nn/modules/head.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py#L1776-L1778"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def fuse(self):
    """Remove the one2many head for inference optimization."""
    self.cv2 = self.cv3 = None
```
</details>

<br><br>
