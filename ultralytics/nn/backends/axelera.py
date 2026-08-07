# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import sys
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

        Args:
            weight (str | Path): Path to the Axelera model directory containing the .axm binary.
        """
        # Evaluated even when the import would succeed, so an older installed SDK is upgraded rather
        # than silently satisfying the import. A floor rather than an exact pin, so a newer runtime is
        # left alone instead of being downgraded on every load.
        runtime = "axelera-rt>=1.8.0"
        if not check_requirements(runtime, install=False):
            # Best effort: users who install the SDK by hand already satisfy the pin and never see this,
            # which is why the docs carry it too. The driver mismatch itself surfaces at model load.
            LOGGER.warning(
                "Axelera SDK 1.8 requires metis-dkms 1.6.2 or newer. An older kernel driver leaves the "
                "device unopenable. See https://docs.ultralytics.com/integrations/axelera/"
            )
            check_requirements(
                runtime,
                cmds="--extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple",
            )
            if "axelera.runtime" in sys.modules:
                # The upgraded runtime cannot replace the already-imported one, so inference would
                # silently run on the old version.
                raise RuntimeError(f"{runtime} was installed over an already-imported older version. Rerun.")

        from axelera.runtime import op

        w = Path(weight)
        found = next(w.rglob("*.axm"), None)
        if found is None:
            raise FileNotFoundError(f"No .axm file found in: {w}")

        self.model = op.load(str(found)).optimized()

        # Load metadata
        metadata_file = found.parent / "metadata.yaml"
        if metadata_file.exists():
            from ultralytics.utils import YAML

            self.apply_metadata(YAML.load(metadata_file))

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
