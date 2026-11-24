# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from ultralytics.utils import LOGGER, YAML


def onnx2axera(
    onnx_file: str | Path,
    output_dir: str | Path,
    name: str,
    quantize: int,
    images: list[str],
    head: str | None = None,
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Compile an ONNX model to an AXERA NPU model (.axmodel) with the Pulsar2 compiler.

    Args:
        onnx_file (str | Path): Input ONNX model path.
        output_dir (str | Path): Directory to write the compiled .axmodel into.
        name (str): Target chip passed to Pulsar2 as ``--target_hardware``, e.g. ``"AX650"``.
        quantize (int): ``8`` for INT8, or ``16`` for BF16 on chips that support it.
        images (list[str]): Calibration image files.
        head (str | None, optional): ONNX node-name scope of the model head. INT8 graphs end at its per-level output
            convs and the decode is saved as ``tail.onnx`` to run on the host in FP32. Defaults to None.
        metadata (dict | None, optional): Optional metadata to save as YAML. Defaults to None.
        prefix (str, optional): Logging prefix. Defaults to "".

    Returns:
        (str): Path to the exported AXERA model directory.
    """
    if not shutil.which("pulsar2"):
        raise FileNotFoundError(
            "AXERA export requires the Pulsar2 'pulsar2' compiler, which was not found on PATH. Install the 'axnn' "
            "wheel from https://huggingface.co/AXERA-TECH/Pulsar2. See https://docs.ultralytics.com/integrations/axera"
        )
    import onnx

    from ultralytics.utils.export.onnx import head_outputs

    output_dir = Path(output_dir).resolve()
    shutil.rmtree(output_dir, ignore_errors=True)  # a re-export would keep another chip's model and INT8 tail
    output_dir.mkdir(parents=True)
    graph = onnx.load(onnx_file).graph
    outputs = head_outputs(graph, head) if head and quantize == 8 else []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        model = str(onnx_file)
        if outputs:
            model = str(tmp / "body.onnx")
            onnx.utils.extract_model(str(onnx_file), model, [x.name for x in graph.input], outputs)
            onnx.utils.extract_model(
                str(onnx_file), str(output_dir / "tail.onnx"), outputs, [x.name for x in graph.output]
            )
        with zipfile.ZipFile(tmp / "calib.zip", "w") as z:
            for i, f in enumerate(images):
                z.write(f, f"{i}{Path(f).suffix}")  # unique names without spaces
        config = {
            "input": model,
            "output_dir": str(tmp),
            "target_hardware": name,
            "quant": {
                "input_configs": [
                    {
                        "tensor_name": "DEFAULT",
                        "calibration_dataset": str(tmp / "calib.zip"),
                        "calibration_size": -1,  # every image
                        "calibration_mean": [0, 0, 0],
                        "calibration_std": [255, 255, 255],
                    }
                ],
                "layer_configs": [
                    {"start_tensor_names": ["DEFAULT"], "end_tensor_names": ["DEFAULT"], "data_type": "BF16"}
                ]
                if quantize == 16
                else [],
            },
            # uint8 RGB NHWC at runtime; tensor_format also makes Pulsar2 read the calibration images as RGB
            "input_processors": [
                {"tensor_name": "DEFAULT", "tensor_format": "RGB", "src_dtype": "U8", "src_layout": "NHWC"}
            ],
        }
        (tmp / "config.json").write_text(json.dumps(config))
        LOGGER.info(f"\n{prefix} starting export with Pulsar2 for {name}...")
        subprocess.run(["pulsar2", "build", "--config", str(tmp / "config.json")], check=True)
        shutil.copy(tmp / "compiled.axmodel", output_dir / f"{Path(onnx_file).stem}.axmodel")

    if metadata is not None:
        YAML.save(output_dir / "metadata.yaml", metadata)

    return str(output_dir)
