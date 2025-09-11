# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import subprocess
import types
from pathlib import Path

import torch

from ultralytics.nn.modules import Detect, Pose
from ultralytics.utils import LOGGER
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.torch_utils import copy_attr


class FXModel(torch.nn.Module):
    """
    A custom model class for torch.fx compatibility.

    This class extends `torch.nn.Module` and is designed to ensure compatibility with torch.fx for tracing and graph
    manipulation. It copies attributes from an existing model and explicitly sets the model attribute to ensure proper
    copying.

    Attributes:
        model (nn.Module): The original model's layers.
    """

    def __init__(self, model, imgsz=(640, 640)):
        """
        Initialize the FXModel.

        Args:
            model (nn.Module): The original model to wrap for torch.fx compatibility.
            imgsz (tuple[int, int]): The input image size (height, width). Default is (640, 640).
        """
        super().__init__()
        copy_attr(self, model)
        # Explicitly set `model` since `copy_attr` somehow does not copy it.
        self.model = model.model
        self.imgsz = imgsz

    def forward(self, x):
        """
        Forward pass through the model.

        This method performs the forward pass through the model, handling the dependencies between layers and saving
        intermediate outputs.

        Args:
            x (torch.Tensor): The input tensor to the model.

        Returns:
            (torch.Tensor): The output tensor from the model.
        """
        y = []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                # from earlier layers
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if isinstance(m, Detect):
                m._inference = types.MethodType(_inference, m)  # bind method to Detect
                m.anchors, m.strides = (
                    x.transpose(0, 1)
                    for x in make_anchors(
                        torch.cat([s / m.stride.unsqueeze(-1) for s in self.imgsz], dim=1), m.stride, 0.5
                    )
                )
            if type(m) is Pose:
                m.forward = types.MethodType(pose_forward, m)  # bind method to Detect
            x = m(x)  # run
            y.append(x)  # save output
        return x


def _inference(self, x: list[torch.Tensor]) -> tuple[torch.Tensor]:
    """Decode boxes and cls scores for imx object detection."""
    x_cat = torch.cat([xi.view(x[0].shape[0], self.no, -1) for xi in x], 2)
    box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
    return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)


def pose_forward(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass for imx pose estimation, including keypoint decoding."""
    bs = x[0].shape[0]  # batch size
    kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
    x = Detect.forward(self, x)
    pred_kpt = self.kpts_decode(bs, kpt)
    return (*x, pred_kpt.permute(0, 2, 1))


class NMSWrapper(torch.nn.Module):
    """Wrap PyTorch Module with multiclass_nms layer from sony_custom_layers."""

    def __init__(
        self,
        model: torch.nn.Module,
        score_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        max_detections: int = 300,
        task: str = "detect",
    ):
        """
        Initialize NMSWrapper with PyTorch Module and NMS parameters.

        Args:
            model (torch.nn.Module): Model instance.
            score_threshold (float): Score threshold for non-maximum suppression.
            iou_threshold (float): Intersection over union threshold for non-maximum suppression.
            max_detections (int): The number of detections to return.
            task (str): Task type, either 'detect' or 'pose'.
        """
        super().__init__()
        self.model = model
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.task = task

    def forward(self, images):
        """Forward pass with model inference and NMS post-processing."""
        from sony_custom_layers.pytorch import multiclass_nms_with_indices

        # model inference
        outputs = self.model(images)
        boxes, scores = outputs[0], outputs[1]
        nms_outputs = multiclass_nms_with_indices(
            boxes=boxes,
            scores=scores,
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections,
        )
        if self.task == "pose":
            kpts = outputs[2]  # (bs, max_detections, kpts 17*3)
            out_kpts = torch.gather(kpts, 1, nms_outputs.indices.unsqueeze(-1).expand(-1, -1, kpts.size(-1)))
            return nms_outputs.boxes, nms_outputs.scores, nms_outputs.labels, out_kpts
        return nms_outputs.boxes, nms_outputs.scores, nms_outputs.labels, nms_outputs.n_valid


def torch2imx(
    model: torch.nn.Module,
    file: Path | str,
    conf: float,
    iou: float,
    max_det: int,
    metadata: dict | None = None,
    gptq: bool = False,
    dataset=None,
    prefix: str = "",
):
    """
    Export YOLO model to IMX format for deployment on Sony IMX500 devices.

    This function quantizes a YOLO model using Model Compression Toolkit (MCT) and exports it
    to IMX format compatible with Sony IMX500 edge devices. It supports both YOLOv8n and YOLO11n
    models for detection and pose estimation tasks.

    Args:
        model (torch.nn.Module): The YOLO model to export. Must be YOLOv8n or YOLO11n.
        file (Path | str): Output file path for the exported model.
        conf (float): Confidence threshold for NMS post-processing.
        iou (float): IoU threshold for NMS post-processing.
        max_det (int): Maximum number of detections to return.
        metadata (dict | None, optional): Metadata to embed in the ONNX model. Defaults to None.
        gptq (bool, optional): Whether to use Gradient-Based Post Training Quantization.
            If False, uses standard Post Training Quantization. Defaults to False.
        dataset (optional): Representative dataset for quantization calibration. Defaults to None.
        prefix (str, optional): Logging prefix string. Defaults to "".

    Returns:
        f (Path): Path to the exported IMX model directory

    Raises:
        ValueError: If the model is not a supported YOLOv8n or YOLO11n variant.

    Example:
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo11n.pt")
        >>> path, _ = export_imx(model, "model.imx", conf=0.25, iou=0.45, max_det=300)

    Note:
        - Requires model_compression_toolkit, onnx, edgemdt_tpc, and sony_custom_layers packages
        - Only supports YOLOv8n and YOLO11n models (detection and pose tasks)
        - Output includes quantized ONNX model, IMX binary, and labels.txt file
    """
    import model_compression_toolkit as mct
    import onnx
    from edgemdt_tpc import get_target_platform_capabilities

    LOGGER.info(f"\n{prefix} starting export with model_compression_toolkit {mct.__version__}...")

    def representative_dataset_gen(dataloader=dataset):
        for batch in dataloader:
            img = batch["img"]
            img = img / 255.0
            yield [img]

    tpc = get_target_platform_capabilities(tpc_version="4.0", device_type="imx500")

    bit_cfg = mct.core.BitWidthConfig()
    if "C2PSA" in model.__str__():  # YOLO11
        if model.task == "detect":
            layer_names = ["sub", "mul_2", "add_14", "cat_21"]
            weights_memory = 2585350.2439
            n_layers = 238  # 238 layers for fused YOLO11n
        elif model.task == "pose":
            layer_names = ["sub", "mul_2", "add_14", "cat_22", "cat_23", "mul_4", "add_15"]
            weights_memory = 2437771.67
            n_layers = 257  # 257 layers for fused YOLO11n-pose
    else:  # YOLOv8
        if model.task == "detect":
            layer_names = ["sub", "mul", "add_6", "cat_17"]
            weights_memory = 2550540.8
            n_layers = 168  # 168 layers for fused YOLOv8n
        elif model.task == "pose":
            layer_names = ["add_7", "mul_2", "cat_19", "mul", "sub", "add_6", "cat_18"]
            weights_memory = 2482451.85
            n_layers = 187  # 187 layers for fused YOLO11n-pose

    # Check if the model has the expected number of layers
    if len(list(model.modules())) != n_layers:
        raise ValueError("IMX export only supported for YOLOv8n and YOLO11n models.")

    for layer_name in layer_names:
        bit_cfg.set_manual_activation_bit_width([mct.core.common.network_editors.NodeNameFilter(layer_name)], 16)

    config = mct.core.CoreConfig(
        mixed_precision_config=mct.core.MixedPrecisionQuantizationConfig(num_of_images=10),
        quantization_config=mct.core.QuantizationConfig(concat_threshold_update=True),
        bit_width_config=bit_cfg,
    )

    resource_utilization = mct.core.ResourceUtilization(weights_memory=weights_memory)

    quant_model = (
        mct.gptq.pytorch_gradient_post_training_quantization(  # Perform Gradient-Based Post Training Quantization
            model=model,
            representative_data_gen=representative_dataset_gen,
            target_resource_utilization=resource_utilization,
            gptq_config=mct.gptq.get_pytorch_gptq_config(
                n_epochs=1000, use_hessian_based_weights=False, use_hessian_sample_attention=False
            ),
            core_config=config,
            target_platform_capabilities=tpc,
        )[0]
        if gptq
        else mct.ptq.pytorch_post_training_quantization(  # Perform post training quantization
            in_module=model,
            representative_data_gen=representative_dataset_gen,
            target_resource_utilization=resource_utilization,
            core_config=config,
            target_platform_capabilities=tpc,
        )[0]
    )

    quant_model = NMSWrapper(
        model=quant_model,
        score_threshold=conf or 0.001,
        iou_threshold=iou,
        max_detections=max_det,
        task=model.task,
    )

    f = Path(str(file).replace(file.suffix, "_imx_model"))
    f.mkdir(exist_ok=True)
    onnx_model = f / Path(str(file.name).replace(file.suffix, "_imx.onnx"))  # js dir
    mct.exporter.pytorch_export_model(
        model=quant_model, save_model_path=onnx_model, repr_dataset=representative_dataset_gen
    )

    model_onnx = onnx.load(onnx_model)  # load onnx model
    for k, v in metadata.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)

    onnx.save(model_onnx, onnx_model)

    subprocess.run(
        ["imxconv-pt", "-i", str(onnx_model), "-o", str(f), "--no-input-persistency", "--overwrite-output"],
        check=True,
    )

    # Needed for imx models.
    with open(f / "labels.txt", "w", encoding="utf-8") as file:
        file.writelines([f"{name}\n" for _, name in model.names.items()])

    return f
