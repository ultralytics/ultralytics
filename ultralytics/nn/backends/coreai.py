# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import asyncio
from pathlib import Path

import torch

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements

from .base import BaseBackend


class CoreAIBackend(BaseBackend):
    """Apple Core AI inference backend for macOS 26+ on Apple silicon.

    Loads a `.aimodel` asset and runs it through the Core AI runtime. That runtime's Python API is async, so this
    backend owns a private event loop and drives every call through it.
    """

    def load_model(self, weight: str | Path) -> None:
        """Load a Core AI `.aimodel` asset.

        Args:
            weight (str | Path): Path to the `.aimodel` asset directory.
        """
        LOGGER.info(f"Loading {weight} for Apple Core AI inference...")
        check_requirements("coreai-torch>=0.4.2")
        from coreai.runtime import AIModel

        w = Path(weight)
        self._loop = asyncio.new_event_loop()
        model = self._loop.run_until_complete(_await(AIModel.load(w)))
        self._function = self._loop.run_until_complete(_await(model.load_function("main")))
        descriptor = self._function.desc
        self._input_name = descriptor.input_names[0]
        self._output_names = list(descriptor.output_names)
        self.fp16 = "float16" in str(descriptor.input_descriptor(self._input_name).dtype)
        self.apply_metadata(self.read_metadata(w))

    def forward(self, im: torch.Tensor) -> list:
        """Run inference through the Core AI runtime.

        Args:
            im (torch.Tensor): Input image tensor in BCHW format, normalized to [0, 1].

        Returns:
            (list): Model outputs as torch tensors, in declared output order.
        """
        from coreai.runtime import NDArray

        inputs = {self._input_name: NDArray(im.cpu().numpy())}
        out = self._loop.run_until_complete(_await(self._function(inputs)))
        values = [out[n] for n in self._output_names] if isinstance(out, dict) else list(out)
        return [torch.from_numpy(v.numpy() if hasattr(v, "numpy") else v).float() for v in values]


async def _await(value):
    """Await `value` when the runtime returned a coroutine, and pass it straight through when it did not."""
    import inspect

    return await value if inspect.isawaitable(value) else value
