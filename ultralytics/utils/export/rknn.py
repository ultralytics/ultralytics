# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import shutil
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
    quantize: int | str | None = None,
    dataset: Path | str | None = None,
    metadata: dict | None = None,
    prefix: str = "",
    batch: int = 1,
    head: str | None = None,
) -> str:
    """Export an ONNX model to RKNN format for Rockchip NPUs with optional INT8 quantization.

    Args:
        onnx_file (str): Path to the source ONNX file (already exported, opset <=19).
        output_dir (Path | str): Directory to save the exported RKNN model.
        name (str): Target platform name (e.g. ``"rk3588"``).
        quantize (int | str | None): Precision scheme, 8 for INT8 or 16/None for floating-point builds.
        dataset (Path | str | None): Path to the generated RKNN Toolkit calibration image-list file, required when
            ``quantize=8``. Users should pass YOLO dataset YAMLs to ``export(data=...)``; ``export_rknn()`` converts
            them to this internal image-path list.
        metadata (dict | None): Metadata saved as ``metadata.yaml``.
        prefix (str): Prefix for log messages.
        batch (int): Inference batch size applied by RKNN Toolkit after loading the batch-1 ONNX model.
        head (str | None): ONNX node-name scope of the model head, e.g. ``"/model.23/"``. INT8 builds end the NPU graph
            at the head's per-level output convolutions and save the remaining box decode as ``tail.onnx`` to run on the
            host in floating point. INT8 builds for FP16-capable targets also use RKNN Toolkit ``auto_hybrid`` to keep
            high-error layers in floating point.

    Returns:
        (str): Path to the exported ``_rknn_model`` directory.

    Raises:
        ValueError: If an INT8-only target is exported without ``quantize=8``, or the INT8 calibration list is missing.
        RuntimeError: If an RKNN Toolkit API call fails.
    """
    use_int8 = quantize == 8
    int8_only = name in {"rv1103", "rv1106", "rv1103b", "rv1106b"}
    if int8_only and not use_int8:
        raise ValueError(
            f"Rockchip target '{name}' requires quantize=8. Use a target that supports floating-point builds "
            f"(e.g. rk2118, rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b) or export with quantize=8."
        )
    if use_int8:
        if not dataset:
            raise ValueError("RKNN INT8 export requires a generated calibration image-list file.")
        dataset = Path(dataset)
        if not dataset.is_file():
            raise ValueError(f"Generated RKNN INT8 calibration image-list file not found: {dataset}")

    from ultralytics.utils.checks import check_requirements

    LOGGER.info(f"\n{prefix} starting export with rknn-toolkit2...")
    # setuptools<82 retains pkg_resources, which rknn-toolkit2 depends on and setuptools 82 removed
    # <2.5: 2.5.0 fails YOLO11 and YOLO26 builds https://github.com/airockchip/rknn-toolkit2/issues/581
    check_requirements(["rknn-toolkit2>=2.3.2,<2.5", "setuptools<82"])

    if IS_COLAB:
        # Prevent 'exit' from closing the notebook https://github.com/airockchip/rknn-toolkit2/issues/259
        import builtins

        builtins.exit = lambda: None

    from rknn.api import RKNN

    rknn = RKNN(verbose=False)
    config = {"mean_values": [[0, 0, 0]], "std_values": [[255, 255, 255]], "target_platform": name}
    outputs = None
    if use_int8:
        if not int8_only:
            config["auto_hybrid_cos_thresh"] = 0.999
        if head:
            import onnx

            graph = onnx.load(onnx_file).graph
            ops = {"Reshape", "Concat"}  # head convs feed only these per-level ops, e.g. Concat in WorldDetect
            ends = {x for n in graph.node if n.op_type in ops for x in n.input}
            ends -= {x for n in graph.node if n.op_type not in ops for x in n.input}
            # Per-tensor INT8 cannot hold the decode's pixel-range DFL, anchor and stride math, so stop at the head convs
            outputs = [
                n.output[0]
                for n in graph.node
                if n.op_type == "Conv" and head in n.name and "/dfl/" not in n.name and n.output[0] in ends
            ]
            if outputs:
                outputs += [x.name for x in graph.output if x.name != "output0"]  # e.g. segment prototypes
    _check_rknn_return(rknn.config(**config), "config")
    _check_rknn_return(rknn.load_onnx(model=onnx_file, outputs=outputs or None), "load_onnx")
    build_kwargs = {"do_quantization": use_int8}
    if use_int8:
        build_kwargs.update(dataset=str(dataset), auto_hybrid=not int8_only)  # INT8-only NPUs have no FP16 layers
    _check_rknn_return(rknn.build(**build_kwargs, rknn_batch_size=batch), "build")
    output_dir = Path(output_dir).resolve()  # rmtree refuses a symlinked directory
    shutil.rmtree(output_dir, ignore_errors=True)  # a re-export would keep another chip's model and INT8 tail
    output_dir.mkdir(parents=True)
    _check_rknn_return(rknn.export_rknn(str(output_dir / f"{Path(onnx_file).stem}-{name}.rknn")), "export_rknn")
    if outputs:
        onnx.utils.extract_model(onnx_file, str(output_dir / "tail.onnx"), outputs, [x.name for x in graph.output])
    if metadata:
        YAML.save(output_dir / "metadata.yaml", metadata)
    return str(output_dir)
