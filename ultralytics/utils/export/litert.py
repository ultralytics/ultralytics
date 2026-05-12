# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import torch

from ultralytics.utils import LOGGER, YAML


def torch2litert(
    model: torch.nn.Module,
    im: torch.Tensor,
    file: Path | str,
    half: bool = False,
    int8: bool = False,
    metadata: dict | None = None,
    prefix: str = "",
) -> Path:
    """Export a PyTorch model to LiteRT format using litert_torch, with optional FP16/INT8 quantization.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        im (torch.Tensor): Example input tensor for tracing.
        file (Path | str): Source model file path used to derive output directory.
        half (bool): Whether to apply FP16 weight-only quantization.
        int8 (bool): Whether to apply dynamic-range INT8 quantization (takes precedence over half).
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
    f = Path(str(file).replace(file.suffix, "_litert_model"))
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
            LOGGER.info(f"{prefix} applying INT8 dynamic-range quantization...")
            qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
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
