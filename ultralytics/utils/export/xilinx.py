# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import os
import re
import sysconfig
from pathlib import Path

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements


def onnx2xilinx(
    onnx_file: str | Path,
    output_dir: str | Path,
    dataset,
    transform_fn,
    name: str = "ve2-xc2ve3858",
    prefix: str = "",
) -> str:
    """Quantize an ONNX model with AMD Quark for AMD Xilinx Versal AI Edge Series Gen 2 NPUs.

    The model is quantized to the Vitis AI `VINT8` configuration (symmetric INT8 with power-of-two scales) using the
    options AMD requires for NPU compilation. The model head stays in floating point, which the Vitis AI compiler runs
    in BF16 on the NPU, because INT8 there costs most of the quantization accuracy loss. The output directory holds the
    quantized ONNX model, which keeps the Ultralytics metadata, and a `vitisai_config.json` for the target device. The
    ONNX Runtime Vitis AI Execution Provider compiles them to a `.rai` model in the same directory on first load.

    Args:
        onnx_file (str | Path): Path to the source FP32 ONNX model, deleted after quantization.
        output_dir (str | Path): Directory to save the exported AMD Xilinx model.
        dataset (DataLoader): Calibration dataloader (from `Exporter.get_int8_calibration_dataloader`).
        transform_fn (Callable): Preprocessing transform (`Exporter._transform_fn`) converting a calibration batch to a
            normalized `float32` NCHW array.
        name (str): Vitis AI compiler device, e.g. `ve2-xc2ve3858` for the VEK385 evaluation kit.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported AMD Xilinx model directory.
    """
    check_requirements("amd-quark>=0.13.0,<0.14.0")
    # Quark JIT-loads its ONNX custom ops with the ninja it installs beside this interpreter, which is missing from PATH
    # when the interpreter is launched by absolute path instead of through an activated environment
    scripts = sysconfig.get_path("scripts")
    if scripts not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = os.pathsep.join(filter(None, (scripts, os.environ.get("PATH"))))
    import onnx
    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config import Config, get_default_config

    from ultralytics.utils.export.onnx import onnx_calibration_reader

    onnx_file, output_dir = Path(onnx_file), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = onnx.load(onnx_file).graph
    head = max(int(m[1]) for n in graph.node if (m := re.match(r"/model\.(\d+)/", n.name)))
    exclude = [n.name for n in graph.node if n.name.startswith(f"/model.{head}/")]
    del graph

    config = get_default_config("VINT8")
    config.extra_options.update(Int32Bias=False, DedicatedQDQPair=True, QuantizeAllOpTypes=True)
    config.enable_npu_cnn = True
    config.nodes_to_exclude = exclude
    f = output_dir / onnx_file.name
    LOGGER.info(f"{prefix} quantizing VINT8 with AMD Quark, keeping {len(exclude)} head nodes float...")
    ModelQuantizer(Config(global_quant_config=config)).quantize_model(
        str(onnx_file), str(f), onnx_calibration_reader(dataset, transform_fn)
    )
    onnx_file.unlink()

    vaiml = {"device": name, "keep_outputs": True, "optimize_level": 2, "threshold_gops_percent": 20}
    vitis_config = {
        "passes": [
            {"name": "init", "plugin": "vaip-pass_init"},
            {"name": "vaiml_partition", "plugin": "vaip-pass_vaiml_partition", "vaiml_config": vaiml},
        ],
        "target": "VAIML",
        "targets": [{"name": "VAIML", "pass": ["init", "vaiml_partition"]}],
    }
    (output_dir / "vitisai_config.json").write_text(json.dumps(vitis_config, indent=2))
    return str(output_dir)
