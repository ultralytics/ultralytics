# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements


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

    def forward(self, x: torch.Tensor):
        """Normalize predictions of object detection model with input size-dependent factors."""
        xywh, cls = self.model(x)[0].transpose(0, 1).split((4, self.nc), 1)
        if self.mlprogram and self.nc % 80 != 0:  # NMS bug https://github.com/ultralytics/ultralytics/issues/22309
            pad_length = int(((self.nc + 79) // 80) * 80) - self.nc  # pad class length to multiple of 80
            cls = torch.nn.functional.pad(cls, (0, pad_length, 0, 0), "constant", 0)
        return cls, xywh * self.normalize


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


def torch2coreml(
    model: nn.Module,
    inputs: list,
    im: torch.Tensor,
    classifier_names: list[str] | None,
    output_file: Path | str | None = None,
    mlmodel: bool = False,
    half: bool = False,
    int8: bool = False,
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
        half (bool): Whether to quantize to FP16.
        int8 (bool): Whether to quantize to INT8.
        metadata (dict | None): Metadata to embed in the CoreML model.
        prefix (str): Prefix for log messages.

    Returns:
        (ct.models.MLModel): The converted CoreML model.
    """
    import coremltools as ct

    LOGGER.info(f"\n{prefix} starting export with coremltools {ct.__version__}...")
    ts = torch.jit.trace(model.eval(), im, strict=False)  # TorchScript model

    # Based on apple's documentation it is better to leave out the minimum_deployment target and let that get set
    # Internally based on the model conversion and output type.
    # Setting minimum_deployment_target >= iOS16 will require setting compute_precision=ct.precision.FLOAT32.
    # iOS16 adds in better support for FP16, but none of the CoreML NMS specifications handle FP16 as input.
    ct_model = ct.convert(
        ts,
        inputs=inputs,
        classifier_config=ct.ClassifierConfig(classifier_names) if classifier_names else None,
        convert_to="neuralnetwork" if mlmodel else "mlprogram",
        skip_model_load=True,
    )
    bits, mode = (8, "kmeans") if int8 else (16, "linear") if half else (32, None)
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
