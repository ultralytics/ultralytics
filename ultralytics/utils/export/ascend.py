# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from ultralytics.utils import LOGGER, YAML


def _check_atc() -> None:
    """Raise if the CANN ATC compiler is not on PATH."""
    if not shutil.which("atc"):
        raise FileNotFoundError(
            "Ascend export requires the CANN toolkit 'atc' compiler, which was not found on PATH. Install CANN and "
            "source its environment, e.g. `source /usr/local/Ascend/ascend-toolkit/set_env.sh`. "
            "See https://docs.ultralytics.com/integrations/ascend/"
        )


def onnx2ascend(
    onnx_file: str | Path,
    output_dir: str | Path,
    name: str,
    imgsz: tuple[int, int],
    batch: int = 1,
    channels: int = 3,
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Convert an ONNX model to a Huawei Ascend offline model (.om) with the CANN ATC compiler.

    Args:
        onnx_file (str | Path): Input ONNX model path.
        output_dir (str | Path): Directory to write the compiled .om model into.
        name (str): Target Ascend SoC passed to ATC as ``--soc_version``, e.g. ``"Ascend310B4"``.
        imgsz (tuple[int, int]): Export image size as ``(height, width)``.
        batch (int, optional): Static batch size baked into the offline model. Defaults to 1.
        channels (int, optional): Input channel count, matching the traced ONNX graph. Defaults to 3.
        metadata (dict | None, optional): Optional metadata to save as YAML. Defaults to None.
        prefix (str, optional): Logging prefix. Defaults to "".

    Returns:
        (str): Path to the exported Ascend model directory.
    """
    _check_atc()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # CANN dereferences the optional score-threshold input when it is omitted from ONNX NonMaxSuppression nodes.
    # NMSModel already filters scores by a positive confidence threshold, so an explicit zero preserves its semantics.
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(onnx_file)
    modified = False
    for i, node in enumerate(model.graph.node):
        if node.op_type == "NonMaxSuppression" and len(node.input) == 4:
            threshold_name = f"ascend_nms_score_threshold_{i}"
            node.input.append(threshold_name)
            model.graph.initializer.append(helper.make_tensor(threshold_name, TensorProto.FLOAT, [1], [0.0]))
            modified = True

    with tempfile.TemporaryDirectory() as scratch:  # ATC drops kernel_meta/ and fusion_result.json in its CWD
        model_file = Path(onnx_file).resolve()
        if modified:
            model_file = Path(scratch) / model_file.name
            onnx.save(model, model_file)
        cmd = [
            "atc",
            f"--model={model_file}",
            "--framework=5",  # 5 = ONNX
            f"--output={output_dir / f'{Path(onnx_file).stem}_{name}'}",  # ATC appends the .om suffix
            "--input_format=NCHW",
            f"--input_shape=images:{batch},{channels},{imgsz[0]},{imgsz[1]}",
            f"--soc_version={name}",
            "--precision_mode=force_fp16",  # Ascend AI Core convolutions reject FP32 inputs
        ]  # argv list avoids shell metacharacter issues in onnx_file/output_dir paths
        LOGGER.info(f"\n{prefix} starting export with ATC for {name}...")
        LOGGER.info(f"{prefix} running '{shlex.join(cmd)}'")
        subprocess.run(cmd, check=True, cwd=scratch)

    if metadata is not None:
        YAML.save(output_dir / "metadata.yaml", metadata)

    return str(output_dir)
