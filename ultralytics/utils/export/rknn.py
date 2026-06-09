# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

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
        dataset = Path(dataset)
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

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=False)
    config = {"mean_values": [[0, 0, 0]], "std_values": [[255, 255, 255]], "target_platform": name}
    _check_rknn_return(rknn.config(**config), "config")
    _check_rknn_return(rknn.load_onnx(model=onnx_file), "load_onnx")
    build_kwargs = {"do_quantization": int8}
    if int8:
        build_kwargs["dataset"] = str(dataset)
    _check_rknn_return(rknn.build(**build_kwargs), "build")
    _check_rknn_return(rknn.export_rknn(str(output_dir / f"{Path(onnx_file).stem}-{name}.rknn")), "export_rknn")
    if metadata:
        YAML.save(output_dir / "metadata.yaml", metadata)
    return str(output_dir)
