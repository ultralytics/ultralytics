# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER, YAML


class _NormalizeCoords(torch.nn.Module):
    """Wrap a model so box (and pose keypoint) coordinates are output normalized to [0, 1].

    LiteRT exports trace the raw PyTorch model, whose detection output concatenates pixel-space box coordinates
    (0-imgsz) with [0, 1] class scores in a single tensor. A single per-tensor INT8 scale cannot represent both
    ranges, so the scores collapse to zero. Normalizing coordinates to [0, 1] keeps the whole tensor in a unit
    range so quantization preserves score resolution; ``LiteRTBackend`` denormalizes by image size at runtime.

    Coordinates are divided by a scalar image size (non-PyTorch exports require square images) rather than a
    per-channel vector: a mixed-magnitude scale vector quantizes to a single per-tensor scale that rounds the
    small coordinate factors to zero, destroying box accuracy.
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
    int8: bool,
    calibration_dataset: torch.utils.data.DataLoader,
    metadata: dict,
    prefix: str,
) -> Path:
    """Export a PyTorch model to LiteRT format using litert_torch, with optional INT8 quantization.

    FP16 is not exported as a separate model: LiteRT runs an FP32 model in FP16 at runtime via the GPU delegate
    (FP16 by default) or the XNNPACK ``FORCE_FP16`` flag on ARM, so a dedicated FP16 file is unnecessary.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        im (torch.Tensor): Example input tensor for tracing.
        file (Path | str): Source model file path used to derive output directory.
        int8 (bool): Whether to apply static INT8 quantization.
        calibration_dataset (DataLoader | None): Calibration dataloader for INT8 quantization, as returned by
            ``get_int8_calibration_dataloader``. Required when ``int8=True``.
        metadata (dict | None): Optional metadata saved as ``metadata.yaml``.
        prefix (str): Prefix for log messages.

    Returns:
        (Path): Path to the exported ``_litert_model`` directory.
    """
    from ultralytics.utils.checks import check_requirements

    check_requirements(("litert-torch>=0.9.0", "ai-edge-litert>=2.1.4"))
    import litert_torch

    LOGGER.info(f"\n{prefix} starting export with litert_torch {litert_torch.__version__}...")
    file = Path(file)
    quant_tag = "_int8" if int8 else ""
    f = Path(str(file).replace(file.suffix, f"{quant_tag}_litert_model"))
    f.mkdir(parents=True, exist_ok=True)

    # Normalize coordinate channels to [0, 1] so INT8 quantization preserves scores (denormalized in LiteRTBackend)
    meta = metadata or {}
    task = meta.get("task")
    if task in {"detect", "segment", "pose", "obb"}:
        model = _NormalizeCoords(model, int(im.shape[3]), task, len(meta.get("names", {})), meta.get("kpt_shape"))

    edge_model = litert_torch.convert(model, (im,))
    suffix = "int8" if int8 else "float32"
    tflite_file = f / f"{file.stem}_{suffix}.tflite"
    edge_model.export(tflite_file)

    if int8:
        check_requirements("ai-edge-quantizer>=0.6.0")
        from ai_edge_quantizer import quantizer, recipe

        LOGGER.info(f"{prefix} applying INT8 static quantization...")
        calib_samples = []
        for batch in calibration_dataset:
            imgs = batch["img"].cpu().float() / 255.0
            for i in range(imgs.shape[0]):
                calib_samples.append({"args_0": imgs[i : i + 1].numpy()})

        qt = quantizer.Quantizer(str(tflite_file))
        qt.load_quantization_recipe(recipe.static_wi8_ai8())
        calibration_result = qt.calibrate({"serving_default": calib_samples})
        qt.quantize(calibration_result=calibration_result).export_model(str(tflite_file), overwrite=True)

    YAML.save(f / "metadata.yaml", metadata or {})
    return f
