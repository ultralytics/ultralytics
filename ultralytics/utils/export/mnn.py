# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from ultralytics.utils import LOGGER
from ultralytics.utils.files import clear_execstack


def import_mnn():
    """Import MNN, first clearing the executable-stack flag that glibc>=2.41 refuses in its x86_64 Linux wheels.

    MNN's x86_64 assembly kernels lack a ``.note.GNU-stack`` section, so the linker marks its extension modules as
    requiring an executable stack that they never use (https://github.com/alibaba/MNN/issues/3390).

    Returns:
        (module): The imported ``MNN`` package.
    """
    try:
        import MNN
    except ImportError as e:
        if "cannot enable executable stack" not in str(e):
            raise
        for name in "_mnncengine", "_tools":  # MNN core and MNN.tools extension modules
            file = importlib.util.find_spec(name).origin
            LOGGER.warning(f"Clearing the executable-stack flag that glibc>=2.41 refuses in {file}")
            clear_execstack(file)
        import MNN
    return MNN


def onnx2mnn(
    onnx_file: str,
    output_file: Path | str,
    quantize: int | str | None = None,
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Convert an ONNX model to MNN format.

    Args:
        onnx_file (str): Path to the source ONNX file.
        output_file (Path | str): Path to save the exported MNN model.
        quantize (int | str | None): Precision scheme, e.g. 16 for FP16 or 8 for INT8 weights.
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
    MNN = import_mnn()
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
    if quantize == 8:
        mnn_args.extend(("--weightQuantBits", "8"))
    if quantize == 16:
        mnn_args.append("--fp16")
    mnnconvert.convert(mnn_args)
    # Remove scratch file created during model convert optimize
    convert_scratch = Path(output_file).parent / ".__convert_external_data.bin"
    if convert_scratch.exists():
        convert_scratch.unlink()
    return str(output_file)
