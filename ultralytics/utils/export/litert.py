# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import torch

from ultralytics.utils import LOGGER


class _NormalizeCoords(torch.nn.Module):
    """Wrap a model so box (and pose keypoint) coordinates are output normalized to [0, 1].

    LiteRT exports trace the raw PyTorch model, whose detection output concatenates pixel-space box coordinates
    (0-imgsz) with [0, 1] class scores in a single tensor. A single per-tensor INT8 scale cannot represent both ranges,
    so the scores collapse to zero. Normalizing coordinates to [0, 1] keeps the whole tensor in a unit range so
    quantization preserves score resolution; ``LiteRTBackend`` denormalizes by image size at runtime.

    Coordinates are divided by a scalar image size (non-PyTorch exports require square images) rather than a per-channel
    vector: a mixed-magnitude scale vector quantizes to a single per-tensor scale that rounds the small coordinate
    factors to zero, destroying box accuracy.
    """

    def __init__(self, model: torch.nn.Module, imgsz: int, task: str, nc: int, kpt_shape: tuple | None):
        """Initialize with the wrapped model, square image size, task, class count and optional keypoint shape."""
        super().__init__()
        self.model = model
        self.imgsz = imgsz
        self.task = task
        self.nc = nc
        self.kpt_shape = kpt_shape

    def forward(self, x: torch.Tensor):
        """Run the wrapped model and normalize coordinate channels of the detection output to [0, 1]."""
        y = self.model(x)
        det = y[0] if isinstance(y, (tuple, list)) else y  # segment returns (detections, protos)
        parts = [det[:, :4] / self.imgsz]  # box xywh
        if self.task == "pose" and self.kpt_shape:
            parts.append(det[:, 4 : 4 + self.nc])  # class scores
            b, _, a = det.shape
            kpts = det[:, 4 + self.nc :].view(b, self.kpt_shape[0], self.kpt_shape[1], a)
            kpts = torch.cat([kpts[:, :, :2] / self.imgsz, kpts[:, :, 2:]], dim=2)  # normalize x, y; keep conf
            parts.append(kpts.reshape(b, -1, a))
        else:
            parts.append(det[:, 4:])  # class scores (+ mask coefficients / angle)
        det = torch.cat(parts, dim=1)
        return (det, *y[1:]) if isinstance(y, (tuple, list)) else det


def torch2litert(
    model: torch.nn.Module,
    im: torch.Tensor,
    file: Path,
    quantize: int | str | None,
    calibration_dataset: torch.utils.data.DataLoader | None,
    metadata: dict | None,
    prefix: str,
) -> Path:
    """Export a PyTorch model to LiteRT format using litert_torch, with optional INT8 quantization.

    Three INT8 schemes are supported via ``quantize``: ``8`` applies static INT8 (int8 weights + int8 activations) and
    ``'w8a16'`` applies static INT8 weights with int16 activations, both requiring a ``calibration_dataset``;
    ``'w8a32'`` applies dynamic/weight-only INT8 (int8 weights + FP32 activations) and needs no calibration.
    ``None``/``32`` exports FP32. FP16 is not exported as a separate model: LiteRT runs an FP32 model in FP16 at runtime
    via the GPU delegate (FP16 by default) or the XNNPACK ``FORCE_FP16`` flag on ARM.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        im (torch.Tensor): Example input tensor for tracing.
        file (Path | str): Source model file path used to derive output directory.
        quantize (int | str | None): Quantization scheme: ``8`` (static INT8), ``'w8a16'`` (static int8 weights + int16
            activations), ``'w8a32'`` (dynamic INT8), or ``None``/``32`` (FP32).
        calibration_dataset (DataLoader | None): Calibration dataloader for static quantization, as returned by
            ``get_int8_calibration_dataloader``. Required when ``quantize`` is ``8`` or ``'w8a16'``.
        metadata (dict | None): Optional metadata embedded in the ``.tflite`` as a ``metadata.json`` entry.
        prefix (str): Prefix for log messages.

    Returns:
        (Path): Path to the exported ``.tflite`` file with metadata embedded as a ``metadata.json`` entry.
    """
    from ultralytics.utils.checks import check_requirements

    check_requirements(("litert-torch>=0.9.0", "ai-edge-litert>=2.1.4"))
    import litert_torch

    static_int8 = quantize == 8
    static_int16 = quantize == "w8a16"
    dynamic_int8 = quantize == "w8a32"
    LOGGER.info(f"\n{prefix} starting export with litert_torch {litert_torch.__version__}...")
    file = Path(file)
    quant_tag = "_int8" if static_int8 else "_w8a16" if static_int16 else "_dynamic_int8" if dynamic_int8 else ""

    # Normalize coordinate channels to [0, 1] so INT8 quantization preserves scores (denormalized in LiteRTBackend).
    # End-to-end models output post-NMS pixel coordinates in FP32 (no scale collapse), so they are left as-is.
    meta = metadata or {}
    task = meta.get("task")
    if task in {"detect", "segment", "pose", "obb"} and not meta.get("end2end", False):
        model = _NormalizeCoords(model, int(im.shape[3]), task, len(meta.get("names", {})), meta.get("kpt_shape"))

    edge_model = litert_torch.convert(model, (im,))
    tflite_file = file.with_name(f"{file.stem}{quant_tag}.tflite")
    edge_model.export(tflite_file)

    if static_int8 or static_int16 or dynamic_int8:
        check_requirements("ai-edge-quantizer>=0.6.0")
        from ai_edge_quantizer import qtyping, quantizer, recipe

        qt = quantizer.Quantizer(str(tflite_file))
        if static_int8 or static_int16:  # static schemes calibrate over representative images
            act = "int8" if static_int8 else "int16"
            LOGGER.info(f"{prefix} applying static quantization (int8 weights + {act} activations)...")
            calib_samples = []
            for batch in calibration_dataset:
                imgs = batch["img"].cpu().float() / 255.0
                # litert-torch traces a fixed batch; feed whole batches matching im's batch dim (skip a partial tail).
                if imgs.shape[0] == im.shape[0]:
                    calib_samples.append({"args_0": imgs.numpy()})
            qt.load_quantization_recipe(recipe.static_wi8_ai8() if static_int8 else recipe.static_wi8_ai16())
            # Keep FP32 graph input/output (weights/activations stay int8/int16 internally): matches the historical
            # onnx2tf "fp32 in/out" contract that downstream consumers (LiteRT GPU delegate, on-device runtimes) expect,
            # and avoids forcing every consumer to (de)quantize at the boundary. Must run after load_quantization_recipe.
            for op in (qtyping.TFLOperationName.INPUT, qtyping.TFLOperationName.OUTPUT):
                qt.update_quantization_recipe(
                    regex=".*", operation_name=op, algorithm_key=recipe.AlgorithmName.NO_QUANTIZE
                )
            result = qt.calibrate({"serving_default": calib_samples})
            qt.quantize(calibration_result=result).export_model(str(tflite_file), overwrite=True)
        else:  # dynamic / weight-only INT8: int8 weights, FP32 activations, no calibration needed
            LOGGER.info(f"{prefix} applying dynamic INT8 quantization (int8 weights + FP32 activations)...")
            qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
            qt.quantize().export_model(str(tflite_file), overwrite=True)

    # Embed metadata as a JSON entry appended to the .tflite (zip-tolerant flatbuffer), so the model is a single
    # self-contained file that LiteRTBackend reads back at load time.
    with zipfile.ZipFile(tflite_file, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.json", json.dumps(metadata or {}))
    return tflite_file
