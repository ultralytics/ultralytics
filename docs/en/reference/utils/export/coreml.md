---
title: utils.export.coreml API Reference
description: CoreML export utilities for converting PyTorch YOLO models to CoreML format for Apple devices. Supports iOS deployment with FP16/INT8 quantization, NMS pipeline integration, and optimized inference on Apple Silicon and mobile devices.
keywords: Ultralytics, CoreML, model export, PyTorch to CoreML, Apple iOS, macOS, Apple Silicon, INT8 quantization, FP16, NMS pipeline, mobile deployment, on-device inference, MLProgram, neural network
---

# Reference for `ultralytics/utils/export/coreml.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/coreml.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/coreml.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`IOSDetectModel`](#ultralytics.utils.export.coreml.IOSDetectModel)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`IOSDetectModel.forward`](#ultralytics.utils.export.coreml.IOSDetectModel.forward)

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`pipeline_coreml`](#ultralytics.utils.export.coreml.pipeline_coreml)
        - [`torch2coreml`](#ultralytics.utils.export.coreml.torch2coreml)


## Class `ultralytics.utils.export.coreml.IOSDetectModel` {#ultralytics.utils.export.coreml.IOSDetectModel}

```python
IOSDetectModel(model: nn.Module, im: torch.Tensor, mlprogram: bool = True)
```

**Bases:** `nn.Module`

Wrap an Ultralytics YOLO model for Apple iOS CoreML export.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `nn.Module` | The YOLO model to wrap. | *required* |
| `im` | `torch.Tensor` | Example input tensor with shape (B, C, H, W). | *required* |
| `mlprogram` | `bool` | Whether exporting to MLProgram format. | `True` |

**Methods**

| Name | Description |
| --- | --- |
| [`forward`](#ultralytics.utils.export.coreml.IOSDetectModel.forward) | Normalize predictions of object detection model with input size-dependent factors. |

<details>
<summary>Source code in <code>ultralytics/utils/export/coreml.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/coreml.py#L15-L45">View on GitHub</a>
```python
class IOSDetectModel(nn.Module):
    """Wrap an Ultralytics YOLO model for Apple iOS CoreML export."""

    def __init__(self, model: nn.Module, im: torch.Tensor, mlprogram: bool = True):
        """Initialize the IOSDetectModel class with a YOLO model and example image.

        Args:
            model (nn.Module): The YOLO model to wrap.
            im (torch.Tensor): Example input tensor with shape (B, C, H, W).
            mlprogram (bool): Whether exporting to MLProgram format.
        """
        super().__init__()
        _, _, h, w = im.shape  # batch, channel, height, width
        self.model = model
        self.nc = len(model.names)  # number of classes
        self.mlprogram = mlprogram
        if w == h:
            self.normalize = 1.0 / w  # scalar
        else:
            self.normalize = torch.tensor(
                [1.0 / w, 1.0 / h, 1.0 / w, 1.0 / h],  # broadcast (slower, smaller)
                device=next(model.parameters()).device,
            )
```
</details>

<br>

### Method `ultralytics.utils.export.coreml.IOSDetectModel.forward` {#ultralytics.utils.export.coreml.IOSDetectModel.forward}

```python
def forward(self, x: torch.Tensor)
```

Normalize predictions of object detection model with input size-dependent factors.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `x` | `torch.Tensor` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/export/coreml.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/coreml.py#L39-L45">View on GitHub</a>
```python
def forward(self, x: torch.Tensor):
    """Normalize predictions of object detection model with input size-dependent factors."""
    xywh, cls = self.model(x)[0].transpose(0, 1).split((4, self.nc), 1)
    if self.mlprogram and self.nc % 80 != 0:  # NMS bug https://github.com/ultralytics/ultralytics/issues/22309
        pad_length = int(((self.nc + 79) // 80) * 80) - self.nc  # pad class length to multiple of 80
        cls = torch.nn.functional.pad(cls, (0, pad_length, 0, 0), "constant", 0)
    return cls, xywh * self.normalize
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.export.coreml.pipeline_coreml` {#ultralytics.utils.export.coreml.pipeline\_coreml}

```python
def pipeline_coreml(
    model: Any,
    output_shape: tuple[int, ...],
    metadata: dict,
    mlmodel: bool = False,
    iou: float = 0.45,
    conf: float = 0.25,
    agnostic_nms: bool = False,
    weights_dir: Path | str | None = None,
    prefix: str = "",
)
```

Create CoreML pipeline with NMS for YOLO detection models.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `Any` | CoreML model. | *required* |
| `output_shape` | `tuple[int, ...]` | Output shape tuple from the exporter. | *required* |
| `metadata` | `dict` | Model metadata. | *required* |
| `mlmodel` | `bool` | Whether the model is an MLModel (vs MLProgram). | `False` |
| `iou` | `float` | IoU threshold for NMS. | `0.45` |
| `conf` | `float` | Confidence threshold for NMS. | `0.25` |
| `agnostic_nms` | `bool` | Whether to use class-agnostic NMS. | `False` |
| `weights_dir` | `Path \| str \| None` | Weights directory for MLProgram models. | `None` |
| `prefix` | `str` | Prefix for log messages. | `""` |

**Returns**

| Type | Description |
| --- | --- |
|  | CoreML pipeline model. |

<details>
<summary>Source code in <code>ultralytics/utils/export/coreml.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/coreml.py#L48-L165">View on GitHub</a>
```python
def pipeline_coreml(
    model: Any,
    output_shape: tuple[int, ...],
    metadata: dict,
    mlmodel: bool = False,
    iou: float = 0.45,
    conf: float = 0.25,
    agnostic_nms: bool = False,
    weights_dir: Path | str | None = None,
    prefix: str = "",
):
    """Create CoreML pipeline with NMS for YOLO detection models.

    Args:
        model: CoreML model.
        output_shape (tuple[int, ...]): Output shape tuple from the exporter.
        metadata (dict): Model metadata.
        mlmodel (bool): Whether the model is an MLModel (vs MLProgram).
        iou (float): IoU threshold for NMS.
        conf (float): Confidence threshold for NMS.
        agnostic_nms (bool): Whether to use class-agnostic NMS.
        weights_dir (Path | str | None): Weights directory for MLProgram models.
        prefix (str): Prefix for log messages.

    Returns:
        CoreML pipeline model.
    """
    import coremltools as ct

    LOGGER.info(f"{prefix} starting pipeline with coremltools {ct.__version__}...")

    spec = model.get_spec()
    outs = list(iter(spec.description.output))
    if mlmodel:  # mlmodel doesn't infer shapes automatically
        outs[0].type.multiArrayType.shape[:] = output_shape[2], output_shape[1] - 4
        outs[1].type.multiArrayType.shape[:] = output_shape[2], 4

    names = metadata["names"]
    nx = spec.description.input[0].type.imageType.width
    ny = spec.description.input[0].type.imageType.height
    nc = outs[0].type.multiArrayType.shape[-1]
    if len(names) != nc:  # Hack fix for MLProgram NMS bug https://github.com/ultralytics/ultralytics/issues/22309
        names = {**names, **{i: str(i) for i in range(len(names), nc)}}

    model = ct.models.MLModel(spec, weights_dir=weights_dir, skip_model_load=True)

    # Create NMS protobuf
    nms_spec = ct.proto.Model_pb2.Model()
    nms_spec.specificationVersion = spec.specificationVersion
    for i in range(len(outs)):
        decoder_output = model._spec.description.output[i].SerializeToString()
        nms_spec.description.input.add()
        nms_spec.description.input[i].ParseFromString(decoder_output)
        nms_spec.description.output.add()
        nms_spec.description.output[i].ParseFromString(decoder_output)

    output_names = ["confidence", "coordinates"]
    for i, name in enumerate(output_names):
        nms_spec.description.output[i].name = name

    for i, out in enumerate(outs):
        ma_type = nms_spec.description.output[i].type.multiArrayType
        ma_type.shapeRange.sizeRanges.add()
        ma_type.shapeRange.sizeRanges[0].lowerBound = 0
        ma_type.shapeRange.sizeRanges[0].upperBound = -1
        ma_type.shapeRange.sizeRanges.add()
        ma_type.shapeRange.sizeRanges[1].lowerBound = out.type.multiArrayType.shape[-1]
        ma_type.shapeRange.sizeRanges[1].upperBound = out.type.multiArrayType.shape[-1]
        del ma_type.shape[:]

    nms = nms_spec.nonMaximumSuppression
    nms.confidenceInputFeatureName = outs[0].name  # 1x507x80
    nms.coordinatesInputFeatureName = outs[1].name  # 1x507x4
    nms.confidenceOutputFeatureName = output_names[0]
    nms.coordinatesOutputFeatureName = output_names[1]
    nms.iouThresholdInputFeatureName = "iouThreshold"
    nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
    nms.iouThreshold = iou
    nms.confidenceThreshold = conf
    nms.pickTop.perClass = not agnostic_nms
    nms.stringClassLabels.vector.extend(names.values())
    nms_model = ct.models.MLModel(nms_spec, skip_model_load=True)

    # Pipeline models together
    pipeline = ct.models.pipeline.Pipeline(
        input_features=[
            ("image", ct.models.datatypes.Array(3, ny, nx)),
            ("iouThreshold", ct.models.datatypes.Double()),
            ("confidenceThreshold", ct.models.datatypes.Double()),
        ],
        output_features=output_names,
    )
    pipeline.add_model(model)
    pipeline.add_model(nms_model)

    # Correct datatypes
    pipeline.spec.description.input[0].ParseFromString(model._spec.description.input[0].SerializeToString())
    pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
    pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

    # Update metadata
    pipeline.spec.specificationVersion = spec.specificationVersion
    pipeline.spec.description.metadata.CopyFrom(spec.description.metadata)
    pipeline.spec.description.metadata.userDefined.update(
        {"IoU threshold": str(nms.iouThreshold), "Confidence threshold": str(nms.confidenceThreshold)}
    )

    # Save the model
    model = ct.models.MLModel(pipeline.spec, weights_dir=weights_dir, skip_model_load=True)
    model.input_description["image"] = "Input image"
    model.input_description["iouThreshold"] = f"(optional) IoU threshold override (default: {nms.iouThreshold})"
    model.input_description["confidenceThreshold"] = (
        f"(optional) Confidence threshold override (default: {nms.confidenceThreshold})"
    )
    model.output_description["confidence"] = 'Boxes × Class confidence (see user-defined metadata "classes")'
    model.output_description["coordinates"] = "Boxes × [x, y, width, height] (relative to image size)"
    LOGGER.info(f"{prefix} pipeline success")
    return model
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.export.coreml.torch2coreml` {#ultralytics.utils.export.coreml.torch2coreml}

```python
def torch2coreml(
    model: nn.Module,
    inputs: list,
    im: torch.Tensor,
    classifier_names: list[str] | None,
    output_file: Path | str | None = None,
    mlmodel: bool = False,
    quantize: int | str | None = None,
    metadata: dict | None = None,
    prefix: str = "",
) -> Any
```

Export a PyTorch model to CoreML ``.mlpackage`` or ``.mlmodel`` format.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | `nn.Module` | The PyTorch model to export. | *required* |
| `inputs` | `list` | CoreML input descriptions for the model. | *required* |
| `im` | `torch.Tensor` | Example input tensor for tracing. | *required* |
| `classifier_names` | `list[str] \| None` | Class names for classifier config, or None if not a classifier. | *required* |
| `output_file` | `Path \| str \| None` | Output file path, or None to skip saving. | `None` |
| `mlmodel` | `bool` | Whether to export as ``.mlmodel`` (neural network) instead of ``.mlpackage`` (ML program). | `False` |
| `quantize` | `int \| str \| None` | Precision scheme, e.g. 16 for FP16 or 8/``"w8a16"`` for INT8 weights. | `None` |
| `metadata` | `dict \| None` | Metadata to embed in the CoreML model. | `None` |
| `prefix` | `str` | Prefix for log messages. | `""` |

**Returns**

| Type | Description |
| --- | --- |
| `ct.models.MLModel` | The converted CoreML model. |

<details>
<summary>Source code in <code>ultralytics/utils/export/coreml.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/export/coreml.py#L168-L255">View on GitHub</a>
```python
def torch2coreml(
    model: nn.Module,
    inputs: list,
    im: torch.Tensor,
    classifier_names: list[str] | None,
    output_file: Path | str | None = None,
    mlmodel: bool = False,
    quantize: int | str | None = None,
    metadata: dict | None = None,
    prefix: str = "",
) -> Any:
    """Export a PyTorch model to CoreML ``.mlpackage`` or ``.mlmodel`` format.

    Args:
        model (nn.Module): The PyTorch model to export.
        inputs (list): CoreML input descriptions for the model.
        im (torch.Tensor): Example input tensor for tracing.
        classifier_names (list[str] | None): Class names for classifier config, or None if not a classifier.
        output_file (Path | str | None): Output file path, or None to skip saving.
        mlmodel (bool): Whether to export as ``.mlmodel`` (neural network) instead of ``.mlpackage`` (ML program).
        quantize (int | str | None): Precision scheme, e.g. 16 for FP16 or 8/``"w8a16"`` for INT8 weights.
        metadata (dict | None): Metadata to embed in the CoreML model.
        prefix (str): Prefix for log messages.

    Returns:
        (ct.models.MLModel): The converted CoreML model.
    """
    import coremltools as ct

    LOGGER.info(f"\n{prefix} starting export with coremltools {ct.__version__}...")
    ts = torch.jit.trace(model.eval(), im, strict=False)  # TorchScript model
    fp16 = quantize == 16
    weight_int8 = quantize in {8, "w8a16"}

    # Based on apple's documentation it is better to leave out the minimum_deployment target and let that get set
    # Internally based on the model conversion and output type.
    # Setting minimum_deployment_target >= iOS16 will require setting compute_precision=ct.precision.FLOAT32.
    # iOS16 adds in better support for FP16, but none of the CoreML NMS specifications handle FP16 as input.
    convert_kwargs = {
        "inputs": inputs,
        "classifier_config": ct.ClassifierConfig(classifier_names) if classifier_names else None,
        "convert_to": "neuralnetwork" if mlmodel else "mlprogram",
        "skip_model_load": True,
    }
    if not mlmodel:
        # ML Program conversion defaults to FP16. Pin FP32 unless FP16/INT8 was requested.
        from ultralytics.nn.modules.head import RTDETRDecoder

        if not (fp16 or weight_int8):
            convert_kwargs["compute_precision"] = ct.precision.FLOAT32
        elif any(isinstance(m, RTDETRDecoder) for m in model.modules()):
            # RT-DETR decoder class logits and deformable-sampling indices drift in fp16; pin those op types to fp32.
            fp32_ops = {"linear", "gather", "gather_nd", "gather_along_axis"}
            convert_kwargs["compute_precision"] = ct.transform.FP16ComputePrecision(
                op_selector=lambda op: op.op_type not in fp32_ops
            )
    ct_model = ct.convert(ts, **convert_kwargs)
    bits, mode = (8, "kmeans") if weight_int8 else (16, "linear") if fp16 else (32, None)
    if bits < 32:
        if "kmeans" in mode:
            check_requirements("scikit-learn")  # scikit-learn package required for k-means quantization
        if mlmodel:
            ct_model = ct.models.neural_network.quantization_utils.quantize_weights(ct_model, bits, mode)
        elif bits == 8:  # mlprogram already quantized to FP16
            import coremltools.optimize.coreml as cto

            op_config = cto.OpPalettizerConfig(mode="kmeans", nbits=bits, weight_threshold=512)
            config = cto.OptimizationConfig(global_config=op_config)
            ct_model = cto.palettize_weights(ct_model, config=config)

    m = dict(metadata or {})  # copy to avoid mutating original
    ct_model.short_description = m.pop("description", "")
    ct_model.author = m.pop("author", "")
    ct_model.license = m.pop("license", "")
    ct_model.version = m.pop("version", "")
    ct_model.user_defined_metadata.update({k: str(v) for k, v in m.items()})

    if output_file is not None:
        try:
            ct_model.save(str(output_file))  # save *.mlpackage
        except Exception as e:
            LOGGER.warning(
                f"{prefix} CoreML export to *.mlpackage failed ({e}), reverting to *.mlmodel export. "
                f"Known coremltools Python 3.11 and Windows bugs https://github.com/apple/coremltools/issues/1928."
            )
            output_file = Path(output_file).with_suffix(".mlmodel")
            ct_model.save(str(output_file))
    return ct_model
```
</details>

<br><br>
