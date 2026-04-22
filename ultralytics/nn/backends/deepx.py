# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import ARM64, IS_DEBIAN_TRIXIE, LOGGER, VERBOSE
from ultralytics.utils.checks import check_apt_requirements, is_sudo_available

from .base import BaseBackend


class DeepXBackend(BaseBackend):
    """DeepX NPU inference backend for DeepX hardware accelerators.

    Loads compiled DeepX models (.dxnn files) and runs inference using the DeepX dx_engine runtime. Requires the
    dx_engine package to be installed.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load a DeepX model from a directory containing a .dxnn file.

        Args:
            weight (str | Path): Path to the DeepX model directory containing the .dxnn binary.

        Raises:
            FileNotFoundError: If no .dxnn file is found in the given directory.
        """
        cmd = ["dxrt-cli", "--version"]
        help_url_sixfab = "https://github.com/sixfab/sixfab_dx/"
        download_url_driver = "https://github.com/DEEPX-AI/dx_rt_npu_linux_driver/raw/main/release/2.4.0/dxrt-driver-dkms_2.4.0-2_all.deb"
        download_url_runtime = "https://github.com/DEEPX-AI/dx_rt/raw/main/release/3.3.0/libdxrt_3.3.0_all.deb"
        whl_dir = Path("/usr/share/libdxrt/src/python_package")
        dxrt_available = False
        _devnull = {} if VERBOSE else {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            dxrt_available = True
        except (FileNotFoundError, subprocess.CalledProcessError):
            if IS_DEBIAN_TRIXIE and ARM64:
                LOGGER.info(f"\nDeepX inference requires the DeepX runtime. Attempting install from {help_url_sixfab}")
                sudo = "sudo " if is_sudo_available() else ""
                for c in (
                    f"wget -qO - https://sixfab.github.io/sixfab_dx/public.gpg | {sudo}gpg --dearmor -o /usr/share/keyrings/sixfab-dx.gpg",
                    f'echo "deb [signed-by=/usr/share/keyrings/sixfab-dx.gpg] https://sixfab.github.io/sixfab_dx trixie main" | {sudo}tee /etc/apt/sources.list.d/sixfab-dx.list',
                ):
                    subprocess.run(c, shell=True, check=True, stdout=subprocess.DEVNULL)
                check_apt_requirements(["sixfab-dx"])
                dxrt_available = True
            else:
                LOGGER.warning(
                    "DeepX runtime (dxrt-cli) not found. "
                    "Will attempt to install automatically if needed."
                )

        try:
            from dx_engine import InferenceEngine
        except ImportError:
            import sys

            installed = False
            pip_cmd = [sys.executable, "-m", "pip", "install"]

            if IS_DEBIAN_TRIXIE and ARM64:
                wheels = sorted(Path("/opt/sixfab-dx/wheels").glob("dx_engine-*.whl"))
                if wheels:
                    LOGGER.info(f"Attempting to install dx_engine from {wheels[-1]}")
                    subprocess.run([*pip_cmd, str(wheels[-1])], check=True)
                    installed = True

            if not installed:
                # Try pre-built wheel from libdxrt package
                wheels = sorted(whl_dir.glob("dx_engine-*.whl")) if whl_dir.exists() else []
                if not wheels and not (IS_DEBIAN_TRIXIE and ARM64):
                    # Wheel not found — attempt to download and install libdxrt to get it
                    LOGGER.info("dx_engine wheel not found. Attempting to install libdxrt package...")
                    import tempfile

                    sudo = "sudo " if is_sudo_available() else ""
                    with tempfile.TemporaryDirectory() as tmpdir:
                        # Install driver if not already present
                        if not dxrt_available:
                            try:
                                driver_deb = Path(tmpdir) / "dxrt-driver-dkms_2.4.0-2_all.deb"
                                LOGGER.info(f"Downloading NPU driver from {download_url_driver}...")
                                subprocess.run(
                                    ["wget", "-q", "-O", str(driver_deb), download_url_driver],
                                    check=True,
                                    timeout=120,
                                    **_devnull,
                                )
                                LOGGER.info("Installing NPU driver...")
                                subprocess.run(f"{sudo}dpkg -i {driver_deb}".split(), check=True, **_devnull)
                            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
                                LOGGER.warning(f"NPU driver installation failed (non-fatal): {e}")

                        # Install runtime (libdxrt)
                        try:
                            runtime_deb = Path(tmpdir) / "libdxrt_3.3.0_all.deb"
                            LOGGER.info(f"Downloading runtime from {download_url_runtime}...")
                            subprocess.run(
                                ["wget", "-q", "-O", str(runtime_deb), download_url_runtime],
                                check=True,
                                timeout=120,
                                **_devnull,
                            )
                            LOGGER.info("Installing runtime (libdxrt)...")
                            # postinst may fail due to PEP 668, but files are still extracted
                            subprocess.run(f"{sudo}dpkg -i {runtime_deb}".split(), **_devnull)
                        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
                            LOGGER.warning(f"Runtime installation failed: {e}")

                    # Re-check for wheels after installation
                    wheels = sorted(whl_dir.glob("dx_engine-*.whl")) if whl_dir.exists() else []

                if wheels:
                    LOGGER.info(f"Attempting to install dx_engine from {wheels[-1]}")
                    try:
                        subprocess.run([*pip_cmd, str(wheels[-1])], check=True, **_devnull)
                        installed = True
                    except subprocess.CalledProcessError as e:
                        LOGGER.warning(f"Failed to install dx_engine wheel: {e}")
                else:
                    LOGGER.warning(
                        f"No dx_engine wheel found in {whl_dir}. "
                        "The libdxrt package may not be installed."
                    )

            if not installed:
                libdxrt_installed = whl_dir.exists()
                msg = "dx_engine is not installed and automatic installation failed.\n"
                if not dxrt_available or not libdxrt_installed:
                    msg += (
                        "\nThe DeepX runtime is not properly installed.\n"
                        "For manual installation, follow these steps:\n"
                        "  1. Download and install the NPU driver:\n"
                        f"     wget {download_url_driver}\n"
                        "     sudo dpkg -i dxrt-driver-dkms_2.4.0-2_all.deb\n"
                        "  2. Download and install the runtime:\n"
                        f"     wget {download_url_runtime}\n"
                        "     sudo dpkg -i libdxrt_3.3.0_all.deb\n"
                        "  3. Install the dx_engine Python package:\n"
                        "     pip install /usr/share/libdxrt/src/python_package/dx_engine-*.whl\n"
                    )
                else:
                    msg += (
                        "\nThe DeepX runtime is installed but the dx_engine Python package is missing.\n"
                        "Install it with:\n"
                        "  pip install /usr/share/libdxrt/src/python_package/dx_engine-*.whl\n"
                    )
                msg += (
                    "\nIf you are using Python 3.12+ and encountering PEP 668 errors,\n"
                    "please create a virtual environment and try again:\n"
                    f"  python3 -m venv /path/to/venv\n"
                    f"  source /path/to/venv/bin/activate\n"
                    f"  pip install /usr/share/libdxrt/src/python_package/dx_engine-*.whl\n"
                    f"\n"
                    f"Current Python: {sys.executable} (v{sys.version.split()[0]})"
                )
                raise OSError(msg)
            from dx_engine import InferenceEngine

        if dxrt_available:
            ver = (
                subprocess.run(cmd, capture_output=True, check=True)
                .stdout.decode()
                .splitlines()[0]
                .split()[-1]
                .lstrip("v")
            )
            LOGGER.info(f"Loading {weight} for DeepX inference... (runtime v{ver})")
        else:
            LOGGER.info(f"Loading {weight} for DeepX inference...")

        w = Path(weight)
        found = next(w.rglob("*.dxnn"), None)
        if found is None:
            raise FileNotFoundError(f"No .dxnn file found in: {w}")

        self.model = InferenceEngine(str(found))

        # Load metadata
        metadata_file = found.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

    def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray]:
        """Run inference on the DeepX NPU.

        Converts each image from BCHW float [0, 1] to HWC uint8 [0, 255] per the DeepX runtime contract,
        runs the engine per image, then stacks outputs along the batch dimension.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (np.ndarray | list[np.ndarray]): Model predictions as a single array or list of arrays.
        """
        outputs = []
        for sample in im.cpu().numpy():
            sample = np.ascontiguousarray(np.clip(np.transpose(sample, (1, 2, 0)) * 255, 0, 255).astype(np.uint8))
            for i, out in enumerate(map(np.asarray, self.model.run([sample]))):
                if i == len(outputs):
                    outputs.append([])
                outputs[i].append(out if out.ndim and out.shape[0] == 1 else out[None])
        y = [np.concatenate(x, axis=0) for x in outputs]
        return y[0] if len(y) == 1 else y
