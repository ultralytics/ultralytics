# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import shutil
import subprocess
import types
from itertools import chain, islice
from pathlib import Path

from ultralytics.utils import LOGGER, YAML

CALIBRATION_IMAGES = 20  # hb_mapper per-tensor activation ranges converge well within this many images


def _rdk_forward(self, x: list) -> list:
    """Return undecoded per-level cls/box head outputs in NHWC, the layout the RDK board-side decoder consumes."""
    heads = self.one2one if self.end2end else self.one2many
    return [
        heads[k][i](x[i]).permute(0, 2, 3, 1).contiguous() for i in range(self.nl) for k in ("cls_head", "box_head")
    ]


def rdk_wrapper(model):
    """Bind the RDK detect head forward so the exported graph emits undecoded cls/box tensors."""
    head = model.model[-1]
    head.forward = types.MethodType(_rdk_forward, head)
    return model


def _check_hb_mapper() -> None:
    """Raise if the D-Robotics hb_mapper compiler is not on PATH."""
    if not shutil.which("hb_mapper"):
        raise FileNotFoundError(
            "RDK export requires the D-Robotics 'hb_mapper' compiler, which was not found on PATH. Install the "
            "toolchain, i.e. `pip install rdkx5-yolo-mapper` for RDK X5, and ensure `hb_mapper` is on your PATH. "
            "See https://docs.ultralytics.com/integrations/drobotics-rdk"
        )


def onnx2rdk(
    onnx_file: str | Path,
    output_dir: str | Path,
    dataset,
    name: str = "bayes-e",
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Compile an ONNX model to a D-Robotics RDK INT8 .bin model with the hb_mapper compiler.

    Args:
        onnx_file (str | Path): Path to the source ONNX file exported with undecoded RDK detect head outputs.
        output_dir (str | Path): Directory to write the compiled RDK model into.
        dataset (DataLoader): Calibration dataloader (from `Exporter.get_int8_calibration_dataloader`) supplying the
            letterboxed uint8 RGB NCHW images hb_mapper derives its INT8 activation ranges from.
        name (str): Target BPU microarchitecture passed to hb_mapper as ``march``, e.g. ``"bayes-e"`` for RDK X5.
        metadata (dict | None): Optional metadata to save as YAML.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported RDK model directory.
    """
    onnx_file = Path(onnx_file).resolve()
    output_dir = Path(output_dir).resolve()
    work_dir = output_dir / "hb_mapper"  # calibration blobs and compiler intermediates, removed on success
    shutil.rmtree(work_dir, ignore_errors=True)  # drop leftovers from a previous failed export
    cal_dir = work_dir / "calibration_data"
    compile_dir = work_dir / "compiler_output"
    cal_dir.mkdir(parents=True)
    compile_dir.mkdir(parents=True)

    # hb_mapper reads raw float32 NCHW RGB blobs in 0-255 and applies scale_value itself
    for i, im in enumerate(islice(chain.from_iterable(batch["img"] for batch in dataset), CALIBRATION_IMAGES)):
        im.float().numpy().tofile(cal_dir / f"{i}.rgbchw")

    config_file = work_dir / "hb_mapper_config.yaml"
    YAML.save(
        config_file,
        {
            "model_parameters": {
                "onnx_model": str(onnx_file),
                "march": name,
                "layer_out_dump": False,
                "working_dir": str(compile_dir),
                "output_model_file_prefix": onnx_file.stem,
            },
            "input_parameters": {
                "input_name": "",
                "input_type_rt": "nv12",  # RDK cameras deliver NV12 frames
                "input_type_train": "rgb",
                "input_layout_train": "NCHW",
                "norm_type": "data_scale",
                "scale_value": 1 / 255,
            },
            "calibration_parameters": {
                "cal_data_dir": str(cal_dir),
                "cal_data_type": "float32",
                "calibration_type": "default",
            },
            "compiler_parameters": {"jobs": 16, "compile_mode": "latency", "debug": True, "optimize_level": "O3"},
        },
    )

    LOGGER.info(f"\n{prefix} starting export with hb_mapper for {name}...")
    subprocess.run(
        ["hb_mapper", "makertbin", "--config", str(config_file), "--model-type", "onnx"], check=True, cwd=work_dir
    )

    compiled = compile_dir / f"{onnx_file.stem}.bin"
    if not compiled.is_file():
        raise FileNotFoundError(f"{prefix} hb_mapper completed but {compiled} was not produced.")
    shutil.move(str(compiled), output_dir / compiled.name)
    if metadata is not None:
        YAML.save(output_dir / "metadata.yaml", metadata)
    shutil.rmtree(work_dir)
    return str(output_dir)
