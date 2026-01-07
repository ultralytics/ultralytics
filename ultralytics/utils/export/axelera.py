# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license


from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.checks import check_apt_requirements, check_requirements


def onnx2axelera(
    onnx_path: str | Path,
    *,
    model,
    calibration_dataset,
    transform_fn: Callable[[object], np.ndarray],
    metadata: Optional[dict] = None,
    prefix: str = "",
):
    """
    Convert an ONNX model into an Axelera compiled artifact directory.

    Args:
        onnx_path: Path to the ONNX model.
        model: Original PyTorch model (used for YOLO11 vs YOLOv8 detection).
        calibration_dataset: Dataloader used by axelera.compiler.quantize().
        transform_fn: Quantization preprocessing function.
        metadata: Optional metadata dict.
        prefix: Log prefix.

    Returns:
        Path to output directory (e.g. yolo11n_axelera_model/)
    """
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    try:
        from axelera import compiler  # noqa
    except ImportError:
        check_apt_requirements(
            ["libllvm14", "libgirepository1.0-dev", "pkg-config", "libcairo2-dev", "build-essential", "cmake"]
        )
        check_requirements(
            "axelera-voyager-sdk==1.5.2",
            cmds="--extra-index-url https://software.axelera.ai/artifactory/axelera-runtime-pypi "
            "--extra-index-url https://software.axelera.ai/artifactory/axelera-dev-pypi",
        )

    from axelera import compiler
    from axelera.compiler import CompilerConfig

    onnx_path = Path(onnx_path)
    model_name = onnx_path.stem
    export_path = Path(f"{model_name}_axelera_model")
    export_path.mkdir(exist_ok=True)

    if "C2PSA" in model.__str__():  # YOLO11
        config = CompilerConfig(
            quantization_scheme="per_tensor_min_max",
            ignore_weight_buffers=False,
            resources_used=0.25,
            aipu_cores_used=1,
            multicore_mode="batch",
            output_axm_format=True,
            model_name=model_name,
        )
        else:  # YOLOv8
            config = CompilerConfig(
                tiling_depth=6,
                split_buffer_promotion=True,
                resources_used=0.25,
                aipu_cores_used=1,
                multicore_mode="batch",
                output_axm_format=True,
                model_name=model_name,
            )

        qmodel = compiler.quantize(
            model=str(onnx_path),
            calibration_dataset=calibration_dataset,
            config=config,
            transform_fn=transform_fn,
        )

        compiler.compile(model=qmodel, config=config, output_dir=export_path)

        axm_name = f"{model_name}.axm"
        axm_src = Path(axm_name)
        axm_dst = export_path / axm_name

        if axm_src.exists():
            axm_src.replace(axm_dst)

        YAML.save(export_path / "metadata.yaml", metadata)

        return export_path
