---
title: nn.backends.hailo API Reference
description: Reference for `ultralytics.nn.backends.hailo` in the Ultralytics package.
keywords: Ultralytics, ultralytics.nn.backends.hailo, API reference, YOLO, Python
---

# Reference for `ultralytics/nn/backends/hailo.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/hailo.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/hailo.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`HailoBackend`](#ultralytics.nn.backends.hailo.HailoBackend)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`HailoBackend.load_model`](#ultralytics.nn.backends.hailo.HailoBackend.load_model)
        - [`HailoBackend.__del__`](#ultralytics.nn.backends.hailo.HailoBackend.__del__)
        - [`HailoBackend.forward`](#ultralytics.nn.backends.hailo.HailoBackend.forward)
        - [`HailoBackend._decode_nms`](#ultralytics.nn.backends.hailo.HailoBackend._decode_nms)
        - [`HailoBackend._decode_boxes`](#ultralytics.nn.backends.hailo.HailoBackend._decode_boxes)
        - [`HailoBackend._decode_segment`](#ultralytics.nn.backends.hailo.HailoBackend._decode_segment)
        - [`HailoBackend._decode_pose`](#ultralytics.nn.backends.hailo.HailoBackend._decode_pose)
        - [`HailoBackend._decode_obb`](#ultralytics.nn.backends.hailo.HailoBackend._decode_obb)
        - [`HailoBackend._decode_raw`](#ultralytics.nn.backends.hailo.HailoBackend._decode_raw)
        - [`HailoBackend._decode_depth`](#ultralytics.nn.backends.hailo.HailoBackend._decode_depth)


## Class `ultralytics.nn.backends.hailo.HailoBackend` {#ultralytics.nn.backends.hailo.HailoBackend}

```python
HailoBackend()
```

**Bases:** `BaseBackend`

HailoRT inference backend for Ultralytics Hailo HEF models.

**Methods**

| Name | Description |
| --- | --- |
| [`__del__`](#ultralytics.nn.backends.hailo.HailoBackend.__del__) | Release the Hailo pipeline and device. |
| [`_decode_boxes`](#ultralytics.nn.backends.hailo.HailoBackend._decode_boxes) | Run DFL and box decoding on cached anchors, returning (B, A, 4) xywh boxes (rotated if angle given). |
| [`_decode_depth`](#ultralytics.nn.backends.hailo.HailoBackend._decode_depth) | Decode the raw depth logit into a metric depth map, mirroring ``Depth.forward`` on the host. |
| [`_decode_nms`](#ultralytics.nn.backends.hailo.HailoBackend._decode_nms) | Convert Hailo per-class NMS output from normalized ``yxyx`` to pixel ``xyxy`` coordinates. |
| [`_decode_obb`](#ultralytics.nn.backends.hailo.HailoBackend._decode_obb) | Decode raw OBB tensors (reg, cls, angle per scale) into the dense output the rotated NMS expects. |
| [`_decode_pose`](#ultralytics.nn.backends.hailo.HailoBackend._decode_pose) | Decode raw pose tensors (reg, cls, kpt per scale) into the dense output the predictor's NMS expects. |
| [`_decode_raw`](#ultralytics.nn.backends.hailo.HailoBackend._decode_raw) | Decode branch-first YOLO26 regression and class outputs. |
| [`_decode_segment`](#ultralytics.nn.backends.hailo.HailoBackend._decode_segment) | Decode raw segmentation tensors (reg, cls, coeff per scale + prototypes) for the predictor's NMS. |
| [`forward`](#ultralytics.nn.backends.hailo.HailoBackend.forward) | Run Hailo inference and return decoded detections, or dense outputs and prototypes for segmentation. |
| [`load_model`](#ultralytics.nn.backends.hailo.HailoBackend.load_model) | Load a Hailo export directory and its Ultralytics metadata. |

<details>
<summary>Source code in <code>ultralytics/nn/backends/hailo.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/hailo.py#L17-L209">View on GitHub</a>
```python
class HailoBackend(BaseBackend):
    """HailoRT inference backend for Ultralytics Hailo HEF models."""
```
</details>

<br>

### Method `ultralytics.nn.backends.hailo.HailoBackend.__del__` {#ultralytics.nn.backends.hailo.HailoBackend.\_\_del\_\_}

```python
def __del__(self)
```

Release the Hailo pipeline and device.

<details>
<summary>Source code in <code>ultralytics/nn/backends/hailo.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/hailo.py#L76-L79">View on GitHub</a>
```python
def __del__(self):
    """Release the Hailo pipeline and device."""
    if stack := getattr(self, "_stack", None):
        stack.close()
```
</details>

<br>

### Method `ultralytics.nn.backends.hailo.HailoBackend._decode_boxes` {#ultralytics.nn.backends.hailo.HailoBackend.\_decode\_boxes}

```python
def _decode_boxes(self, reg_maps: list[torch.Tensor], angle: torch.Tensor | None = None) -> torch.Tensor
```

Run DFL and box decoding on cached anchors, returning (B, A, 4) xywh boxes (rotated if angle given).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `reg_maps` | `list[torch.Tensor]` |  | *required* |
| `angle` | `torch.Tensor \| None` |  | `None` |

<details>
<summary>Source code in <code>ultralytics/nn/backends/hailo.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/hailo.py#L127-L138">View on GitHub</a>
```python
def _decode_boxes(self, reg_maps: list[torch.Tensor], angle: torch.Tensor | None = None) -> torch.Tensor:
    """Run DFL and box decoding on cached anchors, returning (B, A, 4) xywh boxes (rotated if angle given)."""
    from ultralytics.utils.tal import dist2bbox, dist2rbox, make_anchors

    if self._anchors is None:
        strides = [self.input_info.shape[0] / m.shape[2] for m in reg_maps]
        self._anchors = make_anchors(reg_maps, strides)
    anchors, stride_tensor = self._anchors
    dist = self._dfl(torch.cat([m.flatten(2) for m in reg_maps], 2)).transpose(1, 2)
    if angle is not None:
        return dist2rbox(dist, angle, anchors) * stride_tensor
    return dist2bbox(dist, anchors, xywh=True) * stride_tensor
```
</details>

<br>

### Method `ultralytics.nn.backends.hailo.HailoBackend._decode_depth` {#ultralytics.nn.backends.hailo.HailoBackend.\_decode\_depth}

```python
def _decode_depth(self, output: np.ndarray) -> torch.Tensor
```

Decode the raw depth logit into a metric depth map, mirroring ``Depth.forward`` on the host.

The HEF is cut at the head's final logit conv, so the clamp/exp and learned log-affine calibration that follow it in the head run here. The map stays at head resolution (H/4, W/4); ``DepthPredictor.postprocess`` resizes it to the image with ``scale_masks``, the same path the PyTorch model takes at inference.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `output` | `np.ndarray` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/backends/hailo.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/hailo.py#L200-L209">View on GitHub</a>
```python
def _decode_depth(self, output: np.ndarray) -> torch.Tensor:
    """Decode the raw depth logit into a metric depth map, mirroring ``Depth.forward`` on the host.

    The HEF is cut at the head's final logit conv, so the clamp/exp and learned log-affine calibration that
    follow it in the head run here. The map stays at head resolution (H/4, W/4); ``DepthPredictor.postprocess``
    resizes it to the image with ``scale_masks``, the same path the PyTorch model takes at inference.
    """
    logit = torch.from_numpy(output).permute(0, 3, 1, 2)  # (B, H/4, W/4, 1) -> (B, 1, H/4, W/4)
    depth = logit.clamp(-4.0, 5.0).exp()
    return depth.pow(self.metadata.get("cal_a", 1.0)) * math.exp(self.metadata.get("cal_b", 0.0))
```
</details>

<br>

### Method `ultralytics.nn.backends.hailo.HailoBackend._decode_nms` {#ultralytics.nn.backends.hailo.HailoBackend.\_decode\_nms}

```python
def _decode_nms(self, output: list) -> np.ndarray
```

Convert Hailo per-class NMS output from normalized ``yxyx`` to pixel ``xyxy`` coordinates.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `output` | `list` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/backends/hailo.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/hailo.py#L106-L125">View on GitHub</a>
```python
def _decode_nms(self, output: list) -> np.ndarray:
    """Convert Hailo per-class NMS output from normalized ``yxyx`` to pixel ``xyxy`` coordinates."""
    height, width = self.input_info.shape[:2]
    scale = np.array([width, height, width, height], dtype=np.float32)
    frames = []
    for detections in output:
        rows = []
        for class_id, class_detections in enumerate(detections):
            if len(class_detections):
                class_detections = np.asarray(class_detections)
                boxes = class_detections[:, [1, 0, 3, 2]] * scale
                classes = np.full((len(boxes), 1), class_id, dtype=np.float32)
                rows.append(np.concatenate((boxes, class_detections[:, 4:5], classes), axis=1))
        frame = np.concatenate(rows) if rows else np.empty((0, 6), dtype=np.float32)
        frames.append(frame[np.argsort(-frame[:, 4])[:300]])
    count = max(map(len, frames), default=0)
    predictions = np.zeros((len(frames), count, 6), dtype=np.float32)
    for i, frame in enumerate(frames):
        predictions[i, : len(frame)] = frame
    return predictions
```
</details>

<br>

### Method `ultralytics.nn.backends.hailo.HailoBackend._decode_obb` {#ultralytics.nn.backends.hailo.HailoBackend.\_decode\_obb}

```python
def _decode_obb(self, outputs: list[np.ndarray]) -> torch.Tensor
```

Decode raw OBB tensors (reg, cls, angle per scale) into the dense output the rotated NMS expects.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `outputs` | `list[np.ndarray]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/backends/hailo.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/hailo.py#L168-L176">View on GitHub</a>
```python
def _decode_obb(self, outputs: list[np.ndarray]) -> torch.Tensor:
    """Decode raw OBB tensors (reg, cls, angle per scale) into the dense output the rotated NMS expects."""
    maps = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in outputs]
    reg_maps, cls_maps, ang_maps = maps[0::3], maps[1::3], maps[2::3]
    angle = torch.cat([m.flatten(2) for m in ang_maps], 2).transpose(1, 2)  # (B, A, 1) raw
    angle = (angle.sigmoid() - 0.25) * math.pi  # OBB head angle squash, applied on the host
    boxes = self._decode_boxes(reg_maps, angle)  # rotated xywh
    cls = torch.cat([m.flatten(2) for m in cls_maps], 2).transpose(1, 2)  # sigmoid baked in at export
    return torch.cat((boxes, cls, angle), 2).transpose(1, 2)
```
</details>

<br>

### Method `ultralytics.nn.backends.hailo.HailoBackend._decode_pose` {#ultralytics.nn.backends.hailo.HailoBackend.\_decode\_pose}

```python
def _decode_pose(self, outputs: list[np.ndarray]) -> torch.Tensor
```

Decode raw pose tensors (reg, cls, kpt per scale) into the dense output the predictor's NMS expects.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `outputs` | `list[np.ndarray]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/backends/hailo.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/hailo.py#L152-L166">View on GitHub</a>
```python
def _decode_pose(self, outputs: list[np.ndarray]) -> torch.Tensor:
    """Decode raw pose tensors (reg, cls, kpt per scale) into the dense output the predictor's NMS expects."""
    maps = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in outputs]
    reg_maps, cls_maps, kpt_maps = maps[0::3], maps[1::3], maps[2::3]
    boxes = self._decode_boxes(reg_maps)
    anchors, stride_tensor = self._anchors
    cls = torch.cat([m.flatten(2) for m in cls_maps], 2).transpose(1, 2)  # sigmoid baked in at export
    kpts = torch.cat([m.flatten(2) for m in kpt_maps], 2).transpose(1, 2)  # (B, A, nk) raw
    n_kpt, ndim = self.kpt_shape
    b, a, _ = kpts.shape
    y = kpts.view(b, a, n_kpt, ndim)
    # Pose.kpts_decode: xy = (raw * 2 + (anchor - 0.5)) * stride; visibility sigmoid is applied on the host
    xy = (y[..., :2] * 2.0 + (anchors.view(a, 1, 2) - 0.5)) * stride_tensor.view(a, 1, 1)
    kpts = torch.cat((xy, y[..., 2:3].sigmoid()), -1) if ndim == 3 else xy
    return torch.cat((boxes, cls, kpts.view(b, a, -1)), 2).transpose(1, 2)
```
</details>

<br>

### Method `ultralytics.nn.backends.hailo.HailoBackend._decode_raw` {#ultralytics.nn.backends.hailo.HailoBackend.\_decode\_raw}

```python
def _decode_raw(self, outputs: list[np.ndarray]) -> np.ndarray
```

Decode branch-first YOLO26 regression and class outputs.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `outputs` | `list[np.ndarray]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/backends/hailo.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/hailo.py#L178-L198">View on GitHub</a>
```python
def _decode_raw(self, outputs: list[np.ndarray]) -> np.ndarray:
    """Decode branch-first YOLO26 regression and class outputs."""
    from ultralytics.utils.tal import dist2bbox, make_anchors

    split = len(outputs) // 2
    box_maps = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in outputs[:split]]
    cls_maps = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in outputs[split:]]
    if self._anchors is None:
        strides = [self.input_info.shape[0] / x.shape[2] for x in box_maps]
        self._anchors = make_anchors(box_maps, strides)
    anchors, stride_tensor = self._anchors
    boxes = torch.cat([x.flatten(2) for x in box_maps], 2).transpose(1, 2)
    boxes = dist2bbox(boxes, anchors, xywh=False) * stride_tensor
    scores = torch.cat([x.flatten(2) for x in cls_maps], 2).transpose(1, 2).sigmoid()
    classes = scores.shape[2]
    anchor_index = scores.amax(-1).topk(min(300, scores.shape[1]), dim=1).indices[..., None]
    boxes = boxes.gather(1, anchor_index.expand(-1, -1, 4))
    scores = scores.gather(1, anchor_index.expand(-1, -1, classes))
    scores, index = scores.flatten(1).topk(min(300, scores.shape[1] * classes), dim=1)
    boxes = boxes.gather(1, (index // classes)[..., None].expand(-1, -1, 4))
    return torch.cat((boxes, scores[..., None], (index % classes)[..., None].float()), 2).numpy()
```
</details>

<br>

### Method `ultralytics.nn.backends.hailo.HailoBackend._decode_segment` {#ultralytics.nn.backends.hailo.HailoBackend.\_decode\_segment}

```python
def _decode_segment(self, outputs: list[np.ndarray]) -> list[torch.Tensor]
```

Decode raw segmentation tensors (reg, cls, coeff per scale + prototypes) for the predictor's NMS.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `outputs` | `list[np.ndarray]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/backends/hailo.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/hailo.py#L140-L150">View on GitHub</a>
```python
def _decode_segment(self, outputs: list[np.ndarray]) -> list[torch.Tensor]:
    """Decode raw segmentation tensors (reg, cls, coeff per scale + prototypes) for the predictor's NMS."""
    proto = torch.from_numpy(outputs[-1]).permute(0, 3, 1, 2)
    k = len(outputs) - 1
    reg_maps = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in outputs[0:k:3]]
    cls_maps = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in outputs[1:k:3]]
    cof_maps = [torch.from_numpy(x).permute(0, 3, 1, 2) for x in outputs[2:k:3]]
    boxes = self._decode_boxes(reg_maps)
    cls = torch.cat([x.flatten(2) for x in cls_maps], 2).transpose(1, 2)  # sigmoid baked in at export
    cof = torch.cat([x.flatten(2) for x in cof_maps], 2).transpose(1, 2)
    return [torch.cat((boxes, cls, cof), 2).transpose(1, 2), proto]
```
</details>

<br>

### Method `ultralytics.nn.backends.hailo.HailoBackend.forward` {#ultralytics.nn.backends.hailo.HailoBackend.forward}

```python
def forward(self, im: torch.Tensor) -> np.ndarray | list[torch.Tensor]
```

Run Hailo inference and return decoded detections, or dense outputs and prototypes for segmentation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `im` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/backends/hailo.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/hailo.py#L81-L104">View on GitHub</a>
```python
def forward(self, im: torch.Tensor) -> np.ndarray | list[torch.Tensor]:
    """Run Hailo inference and return decoded detections, or dense outputs and prototypes for segmentation."""
    im = np.ascontiguousarray(np.clip(im.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8))
    results = self.model.infer({self.input_info.name: im})
    outputs = [results[x.name] for x in self.output_infos]
    if self.task == "segment":
        return self._decode_segment(outputs)
    if self.task == "pose":
        return self._decode_pose(outputs)
    if self.task == "obb":
        return self._decode_obb(outputs)
    if self.task == "classify":
        return torch.from_numpy(outputs[0]).reshape(outputs[0].shape[0], -1)  # on-chip softmax probabilities
    if self.task == "semantic":
        out = torch.from_numpy(outputs[0])
        if self.metadata.get("semantic_baked"):
            # Multi-class Hailo-10/15 baked the upsample and argmax on chip; return the class map.
            return out.reshape(out.shape[0], out.shape[1], out.shape[2])
        # Hailo-8/8L and single-class heads return raw stride-8 logits; hand them to the predictor's existing
        # bilinear upsample, letterbox removal, and class reduction so results match the PyTorch model exactly.
        return out.permute(0, 3, 1, 2)
    if self.task == "depth":
        return self._decode_depth(outputs[0])
    return self._decode_raw(outputs) if not self.metadata.get("nms", False) else self._decode_nms(outputs[0])
```
</details>

<br>

### Method `ultralytics.nn.backends.hailo.HailoBackend.load_model` {#ultralytics.nn.backends.hailo.HailoBackend.load\_model}

```python
def load_model(self, weight: str | Path) -> None
```

Load a Hailo export directory and its Ultralytics metadata.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `weight` | `str \| Path` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/backends/hailo.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/backends/hailo.py#L20-L74">View on GitHub</a>
```python
def load_model(self, weight: str | Path) -> None:
    """Load a Hailo export directory and its Ultralytics metadata."""
    try:
        from hailo_platform import (
            HEF,
            ConfigureParams,
            FormatType,
            HailoStreamInterface,
            InferVStreams,
            InputVStreamParams,
            OutputVStreamParams,
            VDevice,
        )
    except ImportError as e:
        raise ImportError(
            "Hailo inference requires HailoRT. "
            "See https://docs.ultralytics.com/integrations/hailo/#run-hailo-inference"
        ) from e

    w = Path(weight)
    hef_file = next(w.rglob("*.hef"), None)
    if hef_file is None or not hef_file.is_file():
        raise FileNotFoundError(f"No .hef file found in: {w}")

    LOGGER.info(f"Loading {hef_file} for Hailo inference...")
    metadata_file = hef_file.parent / "metadata.yaml"
    if metadata_file.exists():
        from ultralytics.utils import YAML

        self.apply_metadata(YAML.load(metadata_file))
    if self.task and self.task not in {"detect", "segment", "pose", "obb", "classify", "semantic", "depth"}:
        raise ValueError(
            f"Hailo inference only supports detect, segment, pose, obb, classify, semantic and depth tasks, "
            f"not task='{self.task}'."
        )

    self.hef = HEF(str(hef_file))
    self.input_info = self.hef.get_input_vstream_infos()[0]
    self.output_infos = self.hef.get_output_vstream_infos()
    with ExitStack() as stack:
        target = stack.enter_context(VDevice())
        configure_params = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(self.hef, configure_params)[0]
        stack.enter_context(network_group.activate(network_group.create_params()))
        input_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
        output_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
        self.model = stack.enter_context(InferVStreams(network_group, input_params, output_params))
        self._stack = stack.pop_all()
    self._anchors = None
    if self.task in {"segment", "pose", "obb"}:
        from ultralytics.nn.modules import DFL

        self._dfl = DFL()
    # segmentation, pose and OBB return a dense tensor for the predictor's NMS; detect and classify do not
    self.end2end = self.task not in {"segment", "pose", "obb"}
```
</details>

<br><br>
