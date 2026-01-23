# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from ultralytics.nn.modules import Detect, Pose, Pose26
from ultralytics.utils import LOGGER
from ultralytics.utils.tal import make_anchors


def tf_wrapper(model: torch.nn.Module) -> torch.nn.Module:
    """A wrapper for TensorFlow export compatibility (TF-specific handling is now in head modules)."""
    for m in model.modules():
        if not isinstance(m, Detect):
            continue
        import types

        m._get_decode_boxes = types.MethodType(_tf_decode_boxes, m)
        if isinstance(m, Pose):
            m.kpts_decode = types.MethodType(partial(_tf_kpts_decode, is_pose26=type(m) is Pose26), m)
    return model


def _tf_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
    """Decode bounding boxes for TensorFlow export."""
    shape = x["feats"][0].shape  # BCHW
    boxes = x["boxes"]
    if self.format != "imx" and (self.dynamic or self.shape != shape):
        self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
        self.shape = shape
    grid_h, grid_w = shape[2:4]
    grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=boxes.device).reshape(1, 4, 1)
    norm = self.strides / (self.stride[0] * grid_size)
    dbox = self.decode_bboxes(self.dfl(boxes) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
    return dbox


def _tf_kpts_decode(self, kpts: torch.Tensor, is_pose26: bool = False) -> torch.Tensor:
    """Decode keypoints for TensorFlow export."""
    ndim = self.kpt_shape[1]
    bs = kpts.shape[0]
    # Precompute normalization factor to increase numerical stability
    y = kpts.view(bs, *self.kpt_shape, -1)
    grid_h, grid_w = self.shape[2:4]
    grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
    norm = self.strides / (self.stride[0] * grid_size)
    a = ((y[:, :, :2] + self.anchors) if is_pose26 else (y[:, :, :2] * 2.0 + (self.anchors - 0.5))) * norm
    if ndim == 3:
        a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
    return a.view(bs, self.nk, -1)


def torch2tflite(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    output_file: Path,
    half: bool = False,
    int8: bool = False,
    nms: bool = False,
    calibration_loader=None,
    metadata: dict | None = None,
    prefix: str = "",
) -> Path:
    """Convert PyTorch model directly to TFLite using ai-edge-torch.

    Args:
        model (torch.nn.Module): PyTorch model to convert.
        sample_input (torch.Tensor): Sample input tensor for tracing (BCHW format).
        output_file (Path): Output TFLite file path.
        half (bool, optional): Enable FP16 quantization. Defaults to False.
        int8 (bool, optional): Enable INT8 quantization. Defaults to False.
        nms (bool, optional): Whether NMS is embedded in model. Defaults to False.
        calibration_loader (DataLoader, optional): Calibration data for INT8 quantization.
        metadata (dict, optional): Metadata to embed in TFLite file.
        prefix (str, optional): Logging prefix. Defaults to "".

    Returns:
        (Path): Path to the created TFLite file.

    Raises:
        ValueError: If INT8 quantization is requested but no calibration data is provided.
    """
    import ai_edge_torch

    LOGGER.info(f"{prefix} starting TFLite export with ai-edge-torch {ai_edge_torch.__version__}...")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Ensure model is in eval mode and on CPU for export
    model = model.eval().cpu()
    sample_input = sample_input.cpu()

    # Convert sample input from BCHW to BHWC for TFLite
    sample_input_nhwc = sample_input.permute(0, 2, 3, 1).contiguous()

    # Wrap model to handle BCHW -> BHWC conversion internally
    class TFLiteWrapper(torch.nn.Module):
        """Wrapper to convert BHWC input to BCHW for internal model processing."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            # Input x is BHWC from TFLite, convert to BCHW for PyTorch model
            x = x.permute(0, 3, 1, 2).contiguous()
            return self.m(x)

    wrapped_model = TFLiteWrapper(model).eval()

    # Handle INT8 quantization
    if int8:
        if calibration_loader is None:
            raise ValueError(f"{prefix} INT8 quantization requires calibration data (calibration_loader)")

        try:
            from ai_edge_torch.quantize import pt2e_quantizer
            from ai_edge_torch.quantize.pt2e_quantizer import PT2EQuantizer
            from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

            LOGGER.info(f"{prefix} preparing INT8 quantization with calibration data...")

            # Create quantizer for INT8
            quantizer = PT2EQuantizer().set_global(pt2e_quantizer.get_symmetric_quantization_config())

            # Export model for quantization
            exported_model = torch.export.export(wrapped_model, (sample_input_nhwc,))
            prepared_model = prepare_pt2e(exported_model.module(), quantizer)

            # Calibrate with data
            n_calibration = 0
            with torch.no_grad():
                for batch in calibration_loader:
                    img = batch["img"].float() / 255.0  # Normalize
                    # Resize if needed to match sample input size
                    if img.shape[2:4] != sample_input.shape[2:4]:
                        img = torch.nn.functional.interpolate(
                            img, size=sample_input.shape[2:4], mode="bilinear", align_corners=False
                        )
                    # Convert to NHWC
                    img_nhwc = img.permute(0, 2, 3, 1).contiguous()
                    prepared_model(img_nhwc)
                    n_calibration += img_nhwc.shape[0]
                    if n_calibration >= 100:  # Use at least 100 samples
                        break
            LOGGER.info(f"{prefix} calibrated with {n_calibration} images")

            # Convert to quantized model
            quantized_model = convert_pt2e(prepared_model)

            # Export INT8 model
            edge_model = ai_edge_torch.convert(quantized_model, (sample_input_nhwc,))
            edge_model.export(str(output_file))
            LOGGER.info(f"{prefix} INT8 TFLite model exported to {output_file}")
            return output_file

        except Exception as e:
            LOGGER.warning(f"{prefix} INT8 quantization failed: {e}. Falling back to FP32.")

    # Handle FP16 quantization
    if half:
        try:
            from ai_edge_torch.quantize import quant_recipes

            LOGGER.info(f"{prefix} exporting with FP16 quantization...")
            quant_config = quant_recipes.full_fp16_recipe()
            edge_model = ai_edge_torch.convert(wrapped_model, (sample_input_nhwc,), quant_config=quant_config)
            edge_model.export(str(output_file))
            LOGGER.info(f"{prefix} FP16 TFLite model exported to {output_file}")
            return output_file

        except Exception as e:
            LOGGER.warning(f"{prefix} FP16 quantization failed: {e}. Falling back to FP32.")

    # FP32 export (default)
    LOGGER.info(f"{prefix} exporting FP32 TFLite model...")
    edge_model = ai_edge_torch.convert(wrapped_model, (sample_input_nhwc,))
    edge_model.export(str(output_file))
    LOGGER.info(f"{prefix} FP32 TFLite model exported to {output_file}")
    return output_file


def tflite2edgetpu(tflite_file: str | Path, output_dir: str | Path, prefix: str = ""):
    """Convert a TensorFlow Lite model to Edge TPU format using the Edge TPU compiler.

    Args:
        tflite_file (str | Path): Path to the input TensorFlow Lite (.tflite) model file.
        output_dir (str | Path): Output directory path for the compiled Edge TPU model.
        prefix (str, optional): Logging prefix. Defaults to "".

    Notes:
        Requires the Edge TPU compiler to be installed. The function compiles the TFLite model
        for optimal performance on Google's Edge TPU hardware accelerator.

    Raises:
        AssertionError: If EdgeTPU compilation fails (e.g., unsupported ops from ai-edge-torch).
    """
    import subprocess

    cmd = f'edgetpu_compiler -s -d -k 10 --out_dir "{output_dir}" "{tflite_file}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(
            f"{prefix} EdgeTPU compilation failed. This is expected with ai-edge-torch models "
            f"as the EdgeTPU compiler doesn't support all modern TFLite ops.\n"
            f"Error: {result.stderr or result.stdout}"
        )
    LOGGER.info(f"{prefix} Edge TPU model exported to {output_dir}")


def gd_outputs(gd):
    """Retrieve output names from a TensorFlow GraphDef for inference compatibility.

    Args:
        gd (tensorflow.core.framework.graph_pb2.GraphDef): TensorFlow GraphDef object containing
            the model graph structure.

    Returns:
        (list[str]): List of output node names with ':0' suffix, suitable for TensorFlow session
            inference. Output names are prefixed with 'x:0' format required by TensorFlow's
            wrap_frozen_graph function.

    Notes:
        This function is kept for backward compatibility with existing .pb model inference.
        GraphDef (.pb) export is deprecated - use TFLite format for new models.
    """
    name_list, input_list = [], []
    for node in gd.node:
        name_list.append(node.name)
        input_list.extend(node.input)
    return sorted(f"{x}:0" for x in list(set(name_list) - set(input_list)) if not x.startswith("NoOp"))
