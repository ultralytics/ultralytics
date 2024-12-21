from pathlib import Path
from typing import Union

from ultralytics.engine.model import Model


class NeuronModel(Model):
    def __init__(
        self,
        model: Union[str, Path] = "yolov8n.pt",
        task: str = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(model, task, verbose)

    def export(
        self,
        **kwargs,
    ) -> str:
        """
        Exports the model to a different format suitable for deployment.

        This method facilitates the export of the model to various formats (e.g., ONNX, TorchScript) for deployment
        purposes. It uses the 'Exporter' class for the export process, combining model-specific overrides, method
        defaults, and any additional arguments provided. The combined arguments are used to configure export settings.

        The method supports a wide range of arguments to customize the export process. For a comprehensive list of all
        possible arguments, refer to the 'configuration' section in the documentation.

        Args:
            **kwargs (any): Arbitrary keyword arguments to customize the export process. These are combined with the
                model's overrides and method defaults.

        Returns:
            (str): The exported model filename in the specified format, or an object related to the export process.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        from .neuron_exporter import NeuronExporter

        custom = {
            "imgsz": self.model.args["imgsz"],
            "batch": 1,
            "data": None,
            "verbose": False,
        }  # method defaults
        args = {
            **self.overrides,
            **custom,
            **kwargs,
            "mode": "export",
        }  # highest priority args on the right
        return NeuronExporter(overrides=args, _callbacks=self.callbacks)(model=self.model)
