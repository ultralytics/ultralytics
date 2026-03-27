# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ultralytics.utils import LOGGER, YAML


def onnx2axelera(
    onnx_file: str,
    compile_config=None,
    metadata: dict | None = None,
    calibration_dataset: Any | None = None,
    transform_fn: Callable | None = None,
    prefix: str = "",
):
    """Export an ONNX model to Axelera format.

    Args:
        onnx_file (str): Path to the source ONNX file (already exported).
        compile_config (axelera.compiler.CompilerConfig): Compiler configuration object. If None, a default
            ``CompilerConfig`` is created.
        metadata (dict | None): Metadata saved as ``metadata.yaml``.
        calibration_dataset: Dataloader for INT8 calibration.
        transform_fn: Transformation function applied to calibration batches.
        prefix (str): Prefix for log messages.

    Returns:
        (Path): Path to the exported ``_axelera_model`` directory.
    """
    from axelera import compiler

    LOGGER.info(f"\n{prefix} starting export with axelera...")

    model_name = Path(onnx_file).stem
    export_path = Path(f"{model_name}_axelera_model")
    export_path.mkdir(exist_ok=True)

    if compile_config is None:
        from axelera.compiler import CompilerConfig

        compile_config = CompilerConfig(
            tiling_depth=6,
            split_buffer_promotion=True,
            resources_used=0.25,
            aipu_cores_used=1,
            multicore_mode="batch",
            output_axm_format=True,
            model_name=model_name,
        )

    qmodel = compiler.quantize(
        model=onnx_file,
        calibration_dataset=calibration_dataset,
        config=compile_config,
        transform_fn=transform_fn,
    )
    compiler.compile(model=qmodel, config=compile_config, output_dir=export_path)

    axm_name = f"{model_name}.axm"
    axm_src = Path(axm_name)
    axm_dst = export_path / axm_name
    if axm_src.exists():
        axm_src.replace(axm_dst)

    if metadata:
        YAML.save(export_path / "metadata.yaml", metadata)
    return export_path
