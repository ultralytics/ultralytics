# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from ultralytics.utils import LOGGER, YAML


def _check_hb_mapper() -> None:
    """Raise if the D-Robotics hb_mapper compiler is not on PATH."""
    if not shutil.which("hb_mapper"):
        raise FileNotFoundError(
            "RDK export requires the D-Robotics 'hb_mapper' compiler, which was not found on PATH. Install "
            "'rdkx5-yolo-mapper' in a separate Python 3.10 environment on x86_64 Linux and add its bin directory to "
            "PATH. See https://docs.ultralytics.com/integrations/rdk"
        )


def onnx2rdk(
    onnx_file: str | Path,
    output_dir: str | Path,
    dataset,
    head: str | None = None,
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Compile an ONNX model to an INT8 D-Robotics RDK X5 BPU model (.bin) with the hb_mapper compiler.

    Args:
        onnx_file (str | Path): Input ONNX model path (opset 11).
        output_dir (str | Path): Directory to write the compiled .bin model into.
        dataset (Iterable): INT8 calibration dataloader yielding batch dicts of uint8 RGB ``img`` tensors, batch 1.
        head (str | None, optional): ONNX node-name scope of the model head, e.g. ``"/model.23/"``. The BPU graph ends
            at the head's per-level output convolutions and the remaining box decode is saved as ``tail.onnx`` to run on
            the host in floating point.
        metadata (dict | None, optional): Optional metadata to save as YAML. Defaults to None.
        prefix (str, optional): Logging prefix. Defaults to "".

    Returns:
        (str): Path to the exported RDK model directory.
    """
    import onnx

    from ultralytics.utils.export.onnx import head_outputs

    onnx_file = Path(onnx_file)
    output_dir = Path(output_dir).resolve()
    model = onnx.load(onnx_file)
    graph = model.graph
    outputs = head_outputs(graph, head) if head else []
    names = {x: f"out{i}" for i, x in enumerate(outputs) if "/" in x}
    for n in graph.node:  # the X5 runtime rejects "/" in output names
        n.input[:] = [names.get(x, x) for x in n.input]
        n.output[:] = [names.get(x, x) for x in n.output]
    outputs = [names.get(x, x) for x in outputs]
    LOGGER.info(f"\n{prefix} starting export with hb_mapper...")
    with tempfile.TemporaryDirectory() as scratch:  # hb_mapper drops logs and intermediate models in its CWD
        (Path(scratch) / "cal").mkdir()  # hb_mapper reads every file in the calibration directory
        for i, batch in enumerate(dataset):
            batch["img"].float().numpy().tofile(f"{scratch}/cal/{i}.rgbchw")
        onnx.save(model, f"{scratch}/model.onnx")
        if outputs:  # the compiled body keeps this output order, which the tail inputs share
            onnx.utils.extract_model(f"{scratch}/model.onnx", f"{scratch}/body.onnx", [graph.input[0].name], outputs)
        YAML.save(
            f"{scratch}/config.yaml",
            {
                "model_parameters": {
                    "onnx_model": f"{scratch}/{'body' if outputs else 'model'}.onnx",
                    "march": "bayes-e",
                    "working_dir": scratch,
                    "output_model_file_prefix": onnx_file.stem,
                },
                "input_parameters": {
                    "input_type_rt": "nv12",  # the X5 camera pipeline delivers NV12 frames
                    "input_type_train": "rgb",
                    "input_layout_train": "NCHW",
                    "norm_type": "data_scale",
                    "scale_value": 1 / 255,
                },
                "calibration_parameters": {
                    "cal_data_dir": f"{scratch}/cal",
                    "cal_data_type": "float32",
                    # The default method search is 3x slower and picks KL on some runs, 3 mAP lower on YOLO26n
                    "calibration_type": "max",
                    "max_percentile": 0.99995,
                },
                "compiler_parameters": {"optimize_level": "O3"},
            },
        )
        subprocess.run(
            ["hb_mapper", "makertbin", "--config", "config.yaml", "--model-type", "onnx"], check=True, cwd=scratch
        )
        shutil.rmtree(output_dir, ignore_errors=True)  # a re-export would keep a stale tail
        output_dir.mkdir(parents=True)
        shutil.copy(f"{scratch}/{onnx_file.stem}.bin", output_dir)
        if outputs:
            onnx.utils.extract_model(
                f"{scratch}/model.onnx", str(output_dir / "tail.onnx"), outputs, [x.name for x in graph.output]
            )
    if metadata is not None:
        YAML.save(output_dir / "metadata.yaml", metadata)

    return str(output_dir)
