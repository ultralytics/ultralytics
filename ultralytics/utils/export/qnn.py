# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from pathlib import Path

from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.checks import check_requirements


def onnx2qnn(
    onnx_file: str | Path,
    output_dir: Path | str,
    imgsz: tuple[int, int],
    batch: int = 1,
    name: str | None = None,
    runtime: str = "qnn_dlc",
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Convert an ONNX model to Qualcomm QNN format using Qualcomm AI Hub.

    The conversion is performed by `Qualcomm AI Hub <https://aihub.qualcomm.com/>`_, which compiles the ONNX graph into
    a Qualcomm AI Engine Direct (QNN) artifact for the requested Snapdragon target. A free API token is required and
    must be configured once with ``qai-hub configure --api_token <TOKEN>`` (get a token at
    https://app.aihub.qualcomm.com/).

    Args:
        onnx_file (str | Path): Path to the source ONNX file (already exported).
        output_dir (Path | str): Directory to save the exported QNN model.
        imgsz (tuple[int, int]): Export image size as ``(height, width)``.
        batch (int): Batch size of the exported ONNX model, used to build the compile input spec.
        name (str | None): Qualcomm AI Hub target device name, e.g. ``"Snapdragon 8 Elite QRD"``. If ``None``, the first
            device available to the account is selected (preferring a Snapdragon 8 Elite reference device). Run
            ``qai_hub.get_devices()`` to list every available device.
        runtime (str): Target runtime, either ``"qnn_dlc"`` (portable QNN Deep Learning Container) or
            ``"qnn_context_binary"`` (device-specific precompiled context binary).
        metadata (dict | None): Metadata saved as ``metadata.yaml``.
        prefix (str): Prefix for log messages.

    Returns:
        (str): Path to the exported ``_qnn_model`` directory.
    """
    assert runtime in {"qnn_dlc", "qnn_context_binary"}, (
        f"Invalid QNN runtime '{runtime}', use 'qnn_dlc' or 'qnn_context_binary'."
    )
    check_requirements("qai-hub")
    import qai_hub as hub

    # Mirror the qai_hub config resolution (QAIHUB_CLIENT_INI env var, else ~/.qai_hub/client.ini)
    config_path = Path(os.environ.get("QAIHUB_CLIENT_INI", Path.home() / ".qai_hub" / "client.ini"))
    if not config_path.exists():
        raise FileNotFoundError(
            "Qualcomm AI Hub API token not configured. Create a free token at https://app.aihub.qualcomm.com/ "
            "and run 'qai-hub configure --api_token <TOKEN>' once before exporting to QNN."
        )

    if not name:  # resolve a device from the account instead of assuming a specific one exists
        devices = hub.get_devices()
        if not devices:
            raise RuntimeError("No Qualcomm AI Hub devices available for this account.")
        name = next((d.name for d in devices if "Snapdragon 8 Elite" in d.name), devices[0].name)
        LOGGER.info(f"{prefix} no 'name' provided, auto-selecting Qualcomm AI Hub device '{name}'.")

    LOGGER.info(f"\n{prefix} starting export with Qualcomm AI Hub targeting '{name}'...")

    onnx_file = Path(onnx_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compile_job = hub.submit_compile_job(
        model=str(onnx_file),
        device=hub.Device(name),
        options=f"--target_runtime {runtime}",
        input_specs={"images": (batch, 3, imgsz[0], imgsz[1])},
    )
    target_model = compile_job.get_target_model()  # blocks until the cloud compile job completes
    if target_model is None:
        raise RuntimeError(f"Qualcomm AI Hub compile job failed, see {compile_job.url} for details.")

    suffix = ".bin" if runtime == "qnn_context_binary" else ".dlc"
    target_model.download(str(output_dir / f"{onnx_file.stem}{suffix}"))
    if metadata:
        YAML.save(output_dir / "metadata.yaml", metadata)
    return str(output_dir)
