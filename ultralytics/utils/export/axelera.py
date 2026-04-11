# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.checks import check_requirements


def torch2axelera(
    model: torch.nn.Module,
    file: str | Path,
    calibration_dataset: torch.utils.data.DataLoader,
    transform_fn: Callable[[Any], np.ndarray],
    metadata: dict | None = None,
    prefix: str = "",
) -> Path:
    """Convert a YOLO model to Axelera format.

    Args:
        model (torch.nn.Module): Source YOLO model for quantization.
        file (str | Path): Source model file path used to derive output names.
        calibration_dataset (torch.utils.data.DataLoader): Calibration dataloader for quantization.
        transform_fn (Callable[[Any], np.ndarray]): Calibration preprocessing transform function.
        metadata (dict | None, optional): Optional metadata to save as YAML. Defaults to None.
        prefix (str, optional): Prefix for log messages. Defaults to "".

    Returns:
        (Path): Path to exported Axelera model directory.
    """
    prev_protobuf = os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION")
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    try:
        from axelera import compiler
    except ImportError:
        check_requirements(
            "axelera-devkit==1.6.0rc3",
            cmds="--extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple --pre",
        )
        from axelera import compiler

    from axelera.compiler import CompilerConfig
    from axelera.compiler.config.model_specific import extract_ultralytics_metadata

    LOGGER.info(f"\n{prefix} starting export with Axelera compiler...")

    file = Path(file)
    model_name = file.stem
    export_path = Path(f"{model_name}_axelera_model")
    export_path.mkdir(exist_ok=True)

    axelera_model_metadata = extract_ultralytics_metadata(model)
    config = CompilerConfig(
        model_metadata=axelera_model_metadata,
        model_name=model_name,
        resources_used=0.25,
        aipu_cores_used=1,
        multicore_mode="batch",
        output_axm_format=True,
    )
    qmodel = compiler.quantize(
        model=model,
        calibration_dataset=calibration_dataset,
        config=config,
        transform_fn=transform_fn,
    )
    compiler.compile(model=qmodel, config=config, output_dir=export_path)

    for artifact in [f"{model_name}.axm", "compiler_config_final.toml"]:
        artifact_path = Path(artifact)
        if artifact_path.exists():
            artifact_path.replace(export_path / artifact_path.name)

    # Remove intermediate compiler artifacts, keeping only the compiled model and config.
    keep_suffixes = {".axm"}
    keep_names = {"compiler_config_final.toml", "metadata.yaml"}
    for f in export_path.iterdir():
        if f.is_file() and f.suffix not in keep_suffixes and f.name not in keep_names:
            f.unlink()

    if metadata is not None:
        YAML.save(export_path / "metadata.yaml", metadata)

    # Restore original PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION value
    if prev_protobuf is None:
        os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
    else:
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = prev_protobuf

    return export_path
