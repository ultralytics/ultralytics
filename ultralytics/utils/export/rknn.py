# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
import re
from pathlib import Path

from ultralytics.utils import IS_COLAB, LOGGER, YAML


def _check_rknn_return(ret, name: str):
    """Raise a RuntimeError if an RKNN API call failed."""
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

    INT8 export uses RKNN Toolkit's two-step *hybrid* quantization: the whole graph is quantized to INT8 except the
    output tensors, which are kept in float16. The detection output concatenates box coordinates (range ~0-640) and
    class scores (range 0-1) into a single tensor, so a shared per-tensor INT8 scale collapses the score channels to ~0
    and the model returns zero detections on device. Keeping the ``output*`` tensors in float16 fixes that while
    retaining the INT8 speed-up on the rest of the network.

    Args:
        onnx_file (str): Path to the source ONNX file (already exported, opset <=19).
        output_dir (Path | str): Directory to save the exported RKNN model.
        name (str): Target platform name (e.g. ``"rk3588"``).
        int8 (bool): Whether to enable INT8 quantization. When False, RKNN Toolkit builds a floating-point model for
            FP16-capable targets.
        dataset (Path | str | None): Path to the generated RKNN Toolkit calibration image-list file, required when
            ``int8=True``. Users should pass YOLO dataset YAMLs to ``export(data=...)``; ``export_rknn()`` converts them
            to this internal image-path list.
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
        if not dataset:
            raise ValueError("RKNN INT8 export requires a generated calibration image-list file.")
        dataset = Path(dataset).resolve()
        if not dataset.is_file():
            raise ValueError(f"Generated RKNN INT8 calibration image-list file not found: {dataset}")

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
        _rknn_hybrid_quantization(onnx_file, output_dir, config, str(dataset), rknn_file, prefix)
    else:
        rknn = RKNN(verbose=False)
        _check_rknn_return(rknn.config(**config), "config")
        _check_rknn_return(rknn.load_onnx(model=onnx_file), "load_onnx")
        _check_rknn_return(rknn.build(do_quantization=False), "build")
        _check_rknn_return(rknn.export_rknn(rknn_file), "export_rknn")
        rknn.release()

    if metadata:
        YAML.save(output_dir / "metadata.yaml", metadata)
    return str(output_dir)


def _rknn_hybrid_quantization(onnx_file, output_dir, config, dataset, rknn_file, prefix=""):
    """Run RKNN's two-step hybrid quantization, keeping the detection-head decode layers in float16.

    Step 1 analyzes the graph against the calibration ``dataset`` and emits ``.model``, ``.data`` and
    ``.quantization.cfg`` files. The ``.quantization.cfg`` is then edited to mark the graph output layers as
    ``float16`` (see :func:`onnx2rknn` for why), and step 2 rebuilds the model from the edited config and exports it.

    Args:
        onnx_file (str): Path to the source ONNX file.
        output_dir (Path): Directory the RKNN intermediate files and final model are written to.
        config (dict): Arguments forwarded to ``rknn.config()`` (mean/std values and target platform).
        dataset (str): Path to the calibration image-list file.
        rknn_file (str): Absolute path of the ``.rknn`` file to export.
        prefix (str): Prefix for log messages.
    """
    from rknn.api import RKNN

    onnx_file = str(Path(onnx_file).resolve())
    cwd = os.getcwd()
    os.chdir(output_dir)  # hybrid_quantization_step1 writes its intermediate files to the current directory
    try:
        # Step 1: analyze the graph and emit the .model / .data / .quantization.cfg intermediate files
        rknn = RKNN(verbose=False)
        _check_rknn_return(rknn.config(**config), "config")
        _check_rknn_return(rknn.load_onnx(model=onnx_file), "load_onnx")
        _check_rknn_return(rknn.hybrid_quantization_step1(dataset=dataset, proposal=False), "hybrid_quantization_step1")
        rknn.release()

        # Step 1 emits '{stem}.model/.data/.quantization.cfg'; find them in the CWD or next to the ONNX file
        stem = Path(onnx_file).stem
        cfg_name = f"{stem}.quantization.cfg"
        cfg_file = next((d / cfg_name for d in (Path("."), Path(onnx_file).parent) if (d / cfg_name).is_file()), None)
        if cfg_file is None:
            raise RuntimeError(f"Hybrid quantization step 1 did not produce '{cfg_name}'.")
        model_file, data_file = cfg_file.with_name(f"{stem}.model"), cfg_file.with_name(f"{stem}.data")

        layers = _force_float16_output_layers(cfg_file)
        LOGGER.info(f"{prefix} hybrid quantization: keeping output layers {layers} in float16, INT8 elsewhere")

        # Step 2: rebuild from the edited config and export the final RKNN model
        rknn = RKNN(verbose=False)
        _check_rknn_return(
            rknn.hybrid_quantization_step2(
                model_input=str(model_file), data_input=str(data_file), model_quantization_cfg=str(cfg_file)
            ),
            "hybrid_quantization_step2",
        )
        _check_rknn_return(rknn.export_rknn(rknn_file), "export_rknn")
        rknn.release()

        for f in (model_file, data_file, cfg_file):  # remove intermediate files, keep only the .rknn model
            f.unlink(missing_ok=True)
    finally:
        os.chdir(cwd)


def _force_float16_output_layers(cfg_file: Path) -> list[str]:
    """Edit an RKNN ``.quantization.cfg`` in place, marking the graph output layers as ``float16``.

    The config keys are RKNN tensor names. A detection model's output tensor concatenates box coordinates (range
    ~0-640) and class scores (range 0-1), so a shared per-tensor INT8 scale collapses the score channels to ~0 and the
    model returns zero detections on device. Keeping the ``output*`` tensors (e.g. ``output0`` and its reshape variant
    ``output0-rs``) in float16 preserves them while the rest of the graph stays INT8. Only the
    ``custom_quantize_layers`` block is rewritten; the ``quantize_parameters`` block is left byte-for-byte unchanged.

    Args:
        cfg_file (Path): Path to the ``.quantization.cfg`` file generated by hybrid quantization step 1.

    Returns:
        (list[str]): Tensor names that were set to ``float16``.
    """
    lines = cfg_file.read_text().splitlines(keepends=True)

    # Collect tensor names from the 'quantize_parameters' section (4-space-indented keys ending in ':')
    qp_start = next((i for i, line in enumerate(lines) if line.startswith("quantize_parameters:")), None)
    if qp_start is None:
        raise RuntimeError(f"Malformed RKNN quantization config (no 'quantize_parameters' section): {cfg_file}")
    names = []
    for line in lines[qp_start + 1 :]:
        if line.strip() and not line[0].isspace():
            break  # reached the next top-level section
        if m := re.match(r"^ {4}(\S[^:]*):\s*$", line):
            names.append(m.group(1).strip("'\""))

    selected = [n for n in names if n.startswith("output")]
    if not selected:
        raise RuntimeError(f"No 'output*' layers found in RKNN quantization config to keep in float16: {cfg_file}")

    block = "custom_quantize_layers:\n" + "".join(f"    {n}: float16\n" for n in selected)
    start = next(i for i, line in enumerate(lines) if line.startswith("custom_quantize_layers:"))
    end = start + 1
    while end < len(lines) and (lines[end][0].isspace() or not lines[end].strip()):
        end += 1  # skip the existing (possibly empty) block body
    cfg_file.write_text("".join(lines[:start]) + block + "".join(lines[end:]))
    return selected
