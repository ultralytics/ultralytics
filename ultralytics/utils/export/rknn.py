# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
import re
from pathlib import Path

from ultralytics.utils import IS_COLAB, LOGGER, YAML


def _check(ret, name: str):
    """Raise a RuntimeError if an RKNN API call returned a non-zero (error) code."""
    if ret not in {0, None}:
        raise RuntimeError(f"RKNN {name} failed with return code {ret}.")


def onnx2rknn(
    onnx_file: str,
    output_dir: Path | str,
    name: str = "rk3588",
    int8: bool = False,
    dataset: Path | str | None = None,
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Export an ONNX model to RKNN format for Rockchip NPUs with optional INT8 quantization.

    INT8 export uses RKNN Toolkit's two-step hybrid quantization, keeping the output tensors in float16 while the rest
    of the network is INT8. The output concat packs box coordinates (~0-640) and class scores (0-1) into one tensor, so
    a shared INT8 scale would crush the scores to ~0 and return zero detections on device; float16 outputs avoid that.

    Args:
        onnx_file (str): Path to the source ONNX file (already exported, opset <=19).
        output_dir (Path | str): Directory to save the exported RKNN model.
        name (str): Target platform name (e.g. ``"rk3588"``).
        int8 (bool): Enable INT8 quantization. When False, builds a float model for FP16-capable targets.
        dataset (Path | str | None): Path to the calibration image-list file, required when ``int8=True``.
        metadata (dict | None): Metadata saved as ``metadata.yaml``.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported ``_rknn_model`` directory.
    """
    if name in {"rv1103", "rv1106", "rv1103b", "rv1106b"} and not int8:
        raise ValueError(
            f"Rockchip target '{name}' requires int8=True. Use a target that supports floating-point builds "
            f"(e.g. rk2118, rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b) or export with int8=True."
        )
    if int8:
        dataset = Path(dataset).resolve() if dataset else None
        if not (dataset and dataset.is_file()):
            raise ValueError(f"RKNN INT8 export requires a calibration image-list file, got: {dataset}")

    from ultralytics.utils.checks import check_requirements

    LOGGER.info(f"\n{prefix} starting export with rknn-toolkit2...")
    check_requirements("rknn-toolkit2>=2.3.2")
    check_requirements("onnx<1.19.0")  # fix AttributeError: module 'onnx' has no attribute 'mapping'

    if IS_COLAB:
        # Prevent 'exit' from closing the notebook https://github.com/airockchip/rknn-toolkit2/issues/259
        import builtins

        builtins.exit = lambda: None

    from rknn.api import RKNN

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rknn_file = str(output_dir / f"{Path(onnx_file).stem}-{name}.rknn")
    config = {"mean_values": [[0, 0, 0]], "std_values": [[255, 255, 255]], "target_platform": name}

    if int8:
        _hybrid_quantize(onnx_file, output_dir, config, str(dataset), rknn_file, prefix)
    else:
        rknn = RKNN(verbose=False)
        _check(rknn.config(**config), "config")
        _check(rknn.load_onnx(model=onnx_file), "load_onnx")
        _check(rknn.build(do_quantization=False), "build")
        _check(rknn.export_rknn(rknn_file), "export_rknn")
        rknn.release()

    if metadata:
        YAML.save(output_dir / "metadata.yaml", metadata)
    return str(output_dir)


def _hybrid_quantize(onnx_file, output_dir, config, dataset, rknn_file, prefix=""):
    """Run RKNN two-step hybrid quantization, forcing the graph output tensors to float16 (see :func:`onnx2rknn`).

    Step 1 emits ``<stem>.model``/``.data``/``.quantization.cfg``; the cfg's (empty) ``custom_quantize_layers`` is then
    set to keep every ``output*`` tensor in float16, and step 2 rebuilds and exports the model from it.
    """
    from rknn.api import RKNN

    onnx_file = str(Path(onnx_file).resolve())
    stem = Path(onnx_file).stem
    cwd = os.getcwd()
    os.chdir(output_dir)  # step 1 writes its intermediate files to the current directory
    try:
        rknn = RKNN(verbose=False)
        _check(rknn.config(**config), "config")
        _check(rknn.load_onnx(model=onnx_file), "load_onnx")
        _check(rknn.hybrid_quantization_step1(dataset=dataset, proposal=False), "step1")
        rknn.release()

        # Edit the generated cfg to keep the output tensors in float16, leaving 'quantize_parameters' untouched
        cfg = Path(f"{stem}.quantization.cfg")
        text = cfg.read_text()
        outputs = re.findall(r"^ {4}(output[^:\s]*):", text, flags=re.M)
        block = "custom_quantize_layers:\n" + "".join(f"    {n}: float16\n" for n in outputs)
        cfg.write_text(re.sub(r"^custom_quantize_layers:.*?(?=^quantize_parameters:)", block, text, flags=re.M | re.S))
        LOGGER.info(f"{prefix} hybrid quantization: keeping output layers {outputs} in float16, INT8 elsewhere")

        rknn = RKNN(verbose=False)
        _check(rknn.hybrid_quantization_step2(f"{stem}.model", f"{stem}.data", str(cfg)), "step2")
        _check(rknn.export_rknn(rknn_file), "export_rknn")
        rknn.release()

        for f in cfg, Path(f"{stem}.model"), Path(f"{stem}.data"):
            f.unlink(missing_ok=True)  # keep only the exported .rknn
    finally:
        os.chdir(cwd)
