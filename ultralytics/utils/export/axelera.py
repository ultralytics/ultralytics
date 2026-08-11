# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Axelera Voyager SDK export, and the SDK version policy shared with AxeleraBackend.

A version bump edits the two constants below and nothing else:

- AXELERA_SDK is installed when the environment has no SDK, and is the version the docs describe. An SDK
  the user installed is used as-is, whatever its version, so a validated deployment is never moved.
- AXELERA_SDK_MIN is the oldest SDK this integration supports. Older ones still run, with a warning that
  they are outside the supported window.

Which SDK releases change the compiled .axm format is deliberately not tracked here. Each export records
the SDK that built the model, and the runtime enforces the format it accepts, so a failed load can name
both without this code carrying a per-release table that would go stale on the next Axelera release.
"""

from __future__ import annotations

import os
import shutil
import sysconfig
import tempfile
import threading
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version  # not `metadata`: shadowed by an argument below
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.utils import LOGGER, TORCH_VERSION, YAML
from ultralytics.utils.checks import check_requirements, check_version

AXELERA_SDK = "1.8.0"  # installed when the environment has none
AXELERA_SDK_MIN = "1.6.0"  # oldest SDK this integration supports

# Axelera exports mutate process-global state (the PROTOCOL_BUFFERS env var below, plus any working-directory
# files the compiler emits), so a module-level lock serializes concurrent in-process exports. Cross-process
# Platform workers each hold their own lock and never contend.
_AXELERA_EXPORT_LOCK = threading.Lock()


def sdk_version(package: str) -> str | None:
    """Return an installed Axelera package version, or None when the package is not installed.

    Args:
        package (str): Package to look up, e.g. "axelera-compiler".

    Returns:
        (str | None): Installed version, or None if the package has no distribution metadata.
    """
    try:
        return installed_version(package)
    except PackageNotFoundError:
        return None  # SDK components install without their metapackage, so absence is not an error


def check_sdk_version(package: str) -> None:
    """Report an installed Axelera SDK this release does not target, and a mismatched devkit/runtime pair.

    An SDK below AXELERA_SDK_MIN is outside the supported window and says so; one within it is used
    without comment beyond naming the targeted version, since each SDK is validated against one
    metis-dkms driver and ships the card firmware it expects.

    Args:
        package (str): Installed SDK package to report on, "axelera-devkit" or "axelera-rt".
    """
    installed = sdk_version(package)
    if installed is None:
        return  # the SDK imports but its metapackage is absent, so there is no version to compare
    move = (
        f"pip install {package}=={AXELERA_SDK} --extra-index-url "
        "https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple"
    )
    if not check_version(installed, f">={AXELERA_SDK_MIN}"):
        LOGGER.warning(
            f"{package} {installed} is older than {AXELERA_SDK_MIN}, the oldest Voyager SDK this Ultralytics "
            f"release supports, and is used as-is. To move: {move}"
        )
    elif installed != AXELERA_SDK:
        LOGGER.warning(
            f"{package} {installed} is installed and is used as-is, but this Ultralytics release targets Axelera "
            f"SDK {AXELERA_SDK}. An SDK older than the installed metis-dkms driver can fail while programming the "
            f"device. To move: {move}. Release notes: https://docs.axelera.ai/sdk/release-notes/"
        )
    other = "axelera-rt" if package == "axelera-devkit" else "axelera-devkit"
    paired = sdk_version(other)  # None on a build host with no runtime, or a device host with no devkit
    if paired is not None and paired != installed:
        LOGGER.warning(
            f"{package} {installed} and {other} {paired} are different Voyager SDK releases. The compiled .axm "
            "format is versioned, so a model exported by one is rejected when the other loads it."
        )


def torch2axelera(
    model: torch.nn.Module,
    output_dir: Path | str,
    calibration_dataset: torch.utils.data.DataLoader,
    transform_fn: Callable[[Any], np.ndarray],
    model_name: str = "model",
    metadata: dict | None = None,
    prefix: str = "",
) -> str:
    """Convert a YOLO model to Axelera format.

    Args:
        model (torch.nn.Module): Source YOLO model for quantization.
        output_dir (Path | str): Directory to save the exported Axelera model.
        calibration_dataset (torch.utils.data.DataLoader): Calibration dataloader for quantization.
        transform_fn (Callable[[Any], np.ndarray]): Calibration preprocessing transform function.
        model_name (str, optional): Name for the compiled model. Defaults to "model".
        metadata (dict | None, optional): Optional metadata to save as YAML. Defaults to None.
        prefix (str, optional): Prefix for log messages. Defaults to "".

    Returns:
        (str): Path to exported Axelera model directory.
    """
    # Serialize within the process: the steps below mutate process-global state (the protobuf and PATH env
    # vars, plus any working-directory files the compiler writes), so concurrent in-process exports must
    # not overlap.
    with _AXELERA_EXPORT_LOCK:
        prev_protobuf = os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION")
        prev_path = os.environ.get("PATH")
        try:
            os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
            # The compiler runs `axkernelcc` from PATH, which is missing when the interpreter is launched
            # by absolute path instead of through an activated environment.
            scripts_dir = sysconfig.get_path("scripts")  # not sys.executable's parent: they differ system-wide
            if scripts_dir not in (prev_path or "").split(os.pathsep):
                # Prepended: reached only when the interpreter's own directory is missing, so any
                # axkernelcc already on PATH belongs to another environment and mismatches the devkit.
                os.environ["PATH"] = os.pathsep.join(filter(None, (scripts_dir, prev_path)))
            devkit = f"axelera-devkit=={AXELERA_SDK}"
            try:
                from axelera import compiler
            except ImportError:
                check_requirements(
                    devkit,
                    cmds="--extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple",
                )
                if installed_version("torch").split("+")[0] != TORCH_VERSION.split("+")[0]:
                    # The devkit requires torch<2.13, so installing it downgrades a newer torch. Its
                    # quantizer extension links libtorch and would load against the replaced version.
                    raise RuntimeError(f"{devkit} replaced torch {TORCH_VERSION} during export. Rerun.") from None

                from axelera import compiler
            else:
                check_sdk_version("axelera-devkit")
            check_requirements("omnimalloc==0.5.0")
            from axelera.compiler import CompilerConfig
            from axelera.compiler.config.model_specific import extract_ultralytics_metadata

            LOGGER.info(f"\n{prefix} starting export with Axelera compiler...")

            # Resolve to an absolute path so the relative compile dir below can never alias it.
            output_dir = Path(output_dir).resolve()
            if output_dir.exists():
                shutil.rmtree(output_dir)

            axelera_model_metadata = extract_ultralytics_metadata(model)
            config = CompilerConfig(
                model_metadata=axelera_model_metadata,
                model_name=model_name,
                resources_used=0.25,
                aipu_cores_used=1,
                multicore_mode="batch",
                output_axm_format=True,
            )
            qmodel = compiler.quantize(
                model=model,
                calibration_dataset=calibration_dataset,
                config=config,
                transform_fn=transform_fn,
            )

            # The Axelera compiler emits invalid artifacts for absolute output paths, so compile into a local
            # relative directory. TemporaryDirectory gives it a unique name (so back-to-back exports of identically
            # named models never collide) in the current working directory, and removes it on exit even if
            # compilation raises; passing its relative basename keeps it from aliasing the absolute output_dir, so
            # cleanup can never delete the result.
            with tempfile.TemporaryDirectory(prefix="axelera_compile_", dir=".") as compile_root:
                compile_dir = Path(Path(compile_root).name)
                compiler.compile(model=qmodel, config=config, output_dir=compile_dir)

                output_dir.mkdir(parents=True, exist_ok=True)
                for artifact in [f"{model_name}.axm", "compiler_config_final.toml"]:
                    for artifact_path in [compile_dir / artifact, Path(artifact)]:
                        if artifact_path.exists():
                            artifact_path.replace(output_dir / artifact_path.name)
                            break

                # Remove intermediate compiler artifacts, keeping only the compiled model and config.
                keep_suffixes = {".axm"}
                keep_names = {"compiler_config_final.toml", "metadata.yaml"}
                for f in output_dir.iterdir():
                    if f.is_file() and f.suffix not in keep_suffixes and f.name not in keep_names:
                        f.unlink()

                if metadata is not None:
                    # Recorded so a load failure can name the SDK that built the model. The compiler emits
                    # the .axm, so its version is the one that matters, not the metapackage's.
                    YAML.save(
                        output_dir / "metadata.yaml", {**metadata, "axelera_sdk": sdk_version("axelera-compiler")}
                    )

            return str(output_dir)
        finally:
            # Restore original PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION and PATH values
            if prev_protobuf is None:
                os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
            else:
                os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = prev_protobuf
            if prev_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = prev_path
