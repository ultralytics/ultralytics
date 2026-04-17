# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

from ultralytics.utils import IS_COLAB, LOGGER, YAML


def onnx2rknn(
    onnx_file: str,
    output_dir: Path | str,
    name: str = "rk3588",
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Export an ONNX model to RKNN format for Rockchip NPUs.

    Args:
        onnx_file (str): Path to the source ONNX file (already exported, opset <=19).
        output_dir (Path | str): Directory to save the exported RKNN model.
        name (str): Target platform name (e.g. ``"rk3588"``).
        metadata (dict | None): Metadata saved as ``metadata.yaml``.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported ``_rknn_model`` directory.
    """
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
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=name)
    rknn.load_onnx(model=onnx_file)
    rknn.build(do_quantization=False)  # TODO: Add quantization support
    rknn.export_rknn(str(output_dir / f"{Path(onnx_file).stem}-{name}.rknn"))
    if metadata:
        YAML.save(output_dir / "metadata.yaml", metadata)
    return str(output_dir)
