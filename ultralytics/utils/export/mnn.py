# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from pathlib import Path

from ultralytics.utils import LOGGER


def onnx2mnn(
    onnx_file: str,
    output_file: Path | str,
    half: bool = False,
    int8: bool = False,
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Convert an ONNX model to MNN format.

    Args:
        onnx_file (str): Path to the source ONNX file.
        output_file (Path | str): Path to save the exported MNN model.
        half (bool): Whether to enable FP16 conversion.
        int8 (bool): Whether to enable INT8 weight quantization.
        metadata (dict | None): Optional metadata embedded via ``--bizCode``.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported ``.mnn`` file.
    """
    from ultralytics.utils.checks import check_requirements
    from ultralytics.utils.torch_utils import TORCH_1_10

    assert TORCH_1_10, "MNN export requires torch>=1.10.0 to avoid segmentation faults"
    assert Path(onnx_file).exists(), f"failed to export ONNX file: {onnx_file}"

    check_requirements("MNN>=2.9.6")
    import MNN
    from MNN.tools import mnnconvert

    LOGGER.info(f"\n{prefix} starting export with MNN {MNN.version()}...")
    mnn_args = [
        "",
        "-f",
        "ONNX",
        "--modelFile",
        onnx_file,
        "--MNNModel",
        str(output_file),
        "--bizCode",
        json.dumps(metadata or {}),
    ]
    if int8:
        mnn_args.extend(("--weightQuantBits", "8"))
    if half:
        mnn_args.append("--fp16")
    mnnconvert.convert(mnn_args)
    # Remove scratch file created during model convert optimize
    convert_scratch = Path(output_file).parent / ".__convert_external_data.bin"
    if convert_scratch.exists():
        convert_scratch.unlink()
    return str(output_file)
