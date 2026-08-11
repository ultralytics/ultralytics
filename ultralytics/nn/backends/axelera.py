# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class AxeleraBackend(BaseBackend):
    """Axelera AI inference backend for Axelera Metis AI accelerators.

    Loads compiled Axelera models (.axm files) and runs inference using the Axelera AI runtime SDK.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load an Axelera model from a directory containing a .axm file.

        The SDK is installed only when its API cannot be imported, so a version the user installed is
        used as-is and `check_sdk_version` only reports the gap; `AXELERA_SDK` is what an environment
        without an SDK receives. The kernel driver warning is best effort, since a manual install never
        reaches that branch, and a driver mismatch surfaces when the model loads.

        Args:
            weight (str | Path): Path to the Axelera model directory containing the .axm binary.
        """
        from ultralytics.utils.export.axelera import AXELERA_SDK, check_sdk_version, sdk_version

        try:
            from axelera.runtime import op
        except ImportError:
            LOGGER.warning(
                f"Axelera SDK {AXELERA_SDK} requires metis-dkms 1.6.2 or newer. An older kernel driver "
                "leaves the device unopenable. See https://docs.ultralytics.com/integrations/axelera/"
            )
            check_requirements(
                f"axelera-rt=={AXELERA_SDK}",
                cmds="--extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple",
            )
            from axelera.runtime import op
        else:
            check_sdk_version("axelera-rt")

        w = Path(weight)
        found = next(w.rglob("*.axm"), None)
        if found is None:
            raise FileNotFoundError(f"No .axm file found in: {w}")

        from ultralytics.utils import YAML

        metadata_file = found.parent / "metadata.yaml"
        metadata = YAML.load(metadata_file) if metadata_file.exists() else {}
        built_by = metadata.pop("axelera_sdk", "an unrecorded SDK")  # apply_metadata() sets the rest as attributes

        try:
            self.model = op.load(str(found)).optimized()
        except Exception as e:
            # Either the runtime rejects the compiled format, or it cannot program the device with the
            # installed driver and firmware. It prints which, so name the action for both.
            raise RuntimeError(
                f"{e}\nThe model was built with Axelera SDK {built_by} and axelera-runtime "
                f"{sdk_version('axelera-runtime')} is installed. If those differ, re-export the model with "
                "yolo export model=your-model.pt format=axelera. Otherwise check the metis-dkms driver and card "
                "firmware against your SDK: https://docs.ultralytics.com/integrations/axelera/"
            ) from e

        if metadata:
            self.apply_metadata(metadata)

    def forward(self, im: torch.Tensor) -> np.ndarray | list[np.ndarray]:
        """Run inference on the Axelera hardware accelerator.

        A compiled model accepts a single image, so batches go through the Axelera scheduler, which
        overlaps host-side quantization with execution on the device. `batch()` takes the whole tensor
        as one argument and expands its leading dimension: its `*inputs` varargs mean one entry per
        model input, not per image, and passing one image per argument is rejected. It returns one
        result per image in input order, each keeping its singleton batch dimension, hence the
        concatenate below rather than a stack.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (np.ndarray | list[np.ndarray]): Model predictions, one array per model output.
        """
        im = im.cpu()
        if im.shape[0] == 1:
            return self.model(im)

        outputs = self.model.batch(im)
        if isinstance(outputs[0], (list, tuple)):  # multi-output head, e.g. detections plus masks
            return [np.concatenate(o, axis=0) for o in zip(*outputs)]
        return np.concatenate(outputs, axis=0)
