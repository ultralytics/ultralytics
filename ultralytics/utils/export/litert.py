# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER, YAML


def torch2litert(
    model: torch.nn.Module,
    im: torch.Tensor,
    file: Path,
    half: bool,
    int8: bool,
    calibration_dataset: torch.utils.data.DataLoader,
    metadata: dict,
    prefix: str,
) -> Path:
    """Export a PyTorch model to LiteRT format using litert_torch, with optional FP16/INT8 quantization.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        im (torch.Tensor): Example input tensor for tracing.
        file (Path | str): Source model file path used to derive output directory.
        half (bool): Whether to apply FP16 weight-only quantization.
        int8 (bool): Whether to apply static INT8 quantization (takes precedence over half).
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
    quant_tag = "_int8" if int8 else "_half" if half else ""
    f = Path(str(file).replace(file.suffix, f"{quant_tag}_litert_model"))
    f.mkdir(parents=True, exist_ok=True)

    edge_model = litert_torch.convert(model, (im,))
    suffix = "int8" if int8 else "float16" if half else "float32"
    tflite_file = f / f"{file.stem}_{suffix}.tflite"
    edge_model.export(tflite_file)

    if int8 or half:
        check_requirements("ai-edge-quantizer>=0.6.0")
        from ai_edge_quantizer import qtyping, quantizer, recipe, recipe_manager
        from ai_edge_quantizer.algorithm_manager import AlgorithmName

        qt = quantizer.Quantizer(str(tflite_file))
        if int8:
            LOGGER.info(f"{prefix} applying INT8 static quantization...")
            qt.load_quantization_recipe(recipe.static_wi8_ai8())
            calib_samples = []
            for batch in calibration_dataset:
                imgs = batch["img"].cpu().float() / 255.0
                for i in range(imgs.shape[0]):
                    calib_samples.append({"args_0": imgs[i : i + 1].numpy()})
            calibration_result = qt.calibrate({"serving_default": calib_samples})
            qt.quantize(calibration_result=calibration_result).export_model(str(tflite_file), overwrite=True)
        else:
            LOGGER.info(f"{prefix} applying FP16 weight-only quantization...")
            rp = recipe_manager.RecipeManager()
            rp.add_weight_only_config(
                regex=".*",
                operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
                num_bits=16,
                algorithm_key=AlgorithmName.FLOAT_CASTING,
            )
            qt.load_quantization_recipe(rp.get_quantization_recipe())
            qt.quantize().export_model(str(tflite_file), overwrite=True)

    YAML.save(f / "metadata.yaml", metadata or {})
    return f
