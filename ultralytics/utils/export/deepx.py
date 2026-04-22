# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

from ultralytics.utils import LINUX, LOGGER, VERBOSE, YAML
from ultralytics.utils.checks import check_requirements


def _ensure_dxtron(prefix: str = "") -> None:
    """Install dxtron visualizer if not already present (amd64 Linux only)."""
    if shutil.which("dxtron"):
        return
    if not (LINUX and platform.machine() in ("x86_64", "AMD64")):
        return

    download_url = "https://sdk.deepx.ai/release/dxtron/v2.0.1/dxtron_2.0.1_amd64.deb"
    _devnull = {} if VERBOSE else {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    try:
        LOGGER.info(f"{prefix} Installing dxtron visualizer from {download_url}...")
        with tempfile.TemporaryDirectory() as tmpdir:
            deb_path = Path(tmpdir) / "dxtron_2.0.1_amd64.deb"
            subprocess.run(["wget", "-q", "-O", str(deb_path), download_url], check=True, timeout=120, **_devnull)
            subprocess.run(["sudo", "dpkg", "-i", str(deb_path)], check=True, **_devnull)
        LOGGER.info(f"{prefix} dxtron installed successfully ✅")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        LOGGER.warning(f"{prefix} dxtron installation failed (non-fatal): {e}")


def onnx2deepx(
    onnx_file: str | Path,
    imgsz: tuple[int, int],
    dataset,
    metadata: dict | None = None,
    opt_level: int = 0,
    prefix: str = "",
) -> Path:
    """Convert an ONNX model to DeepX format using the DeepX CLI tools.

    Args:
        onnx_file (str | Path): Input ONNX model path.
        imgsz (tuple[int, int]): Export image size as ``(height, width)``.
        dataset (DataLoader): Calibration dataloader used to build the DeepX config.
        metadata (dict | None, optional): Optional metadata to save as YAML. Defaults to None.
        opt_level (int, optional): Optimization level for the DeepX compiler (0-1). Higher levels reduce inference
            latency but increase compilation time. Defaults to 0.
        prefix (str, optional): Logging prefix. Defaults to "".

    Returns:
        (Path): Path to the exported DeepX model directory.
    """
    try:
        import dx_com
    except ImportError:
        check_requirements("dx_com", cmds="-f https://sdk.deepx.ai/release/dxcom/v2.3.0/index.html")
        import dx_com

    _ensure_dxtron(prefix=prefix)

    LOGGER.info(f"\n{prefix} starting export with DeepX...")

    onnx_file = Path(onnx_file)
    export_path = Path(f"{onnx_file.stem}_deepx_model")
    export_path.mkdir(exist_ok=True)
    config_path = export_path / "config.json"

    config = {
        "inputs": {"images": [1, 3, imgsz[0], imgsz[1]]},
        "calibration_num": 100,
        "calibration_method": "ema",
        "train_batchsize": 32,
        "num_samples": 1024,
        "default_loader": {
            "dataset_path": str(dataset.dataset.img_path),
            "file_extensions": ["jpeg", "jpg", "png", "JPEG"],
            "preprocessings": [
                {"resize": {"mode": "pad", "size": imgsz[0], "pad_location": "edge", "pad_value": [114, 114, 114]}},
                {"div": {"x": 255.0}},
                {"convertColor": {"form": "BGR2RGB"}},
                {"transpose": {"axis": [2, 0, 1]}},
                {"expandDim": {"axis": 0}},
            ],
        },
    }

    with open(config_path, "w") as file:
        json.dump(config, file)

    dx_com.compile(
        model=str(onnx_file),
        output_dir=str(export_path),
        config=str(config_path),
        opt_level=opt_level,
    )

    if metadata is not None:
        YAML.save(export_path / "metadata.yaml", metadata)

    return export_path
