from ultralytics.engine.model import Model
from ultralytics.nn.tasks import DFINEDetectModel
from ..rtdetr import RTDETRPredictor, RTDETRValidator
from .train import DEIMModelTrainer


class DEIM(Model):
    """
    Interface for the DEIM model, a Vision Transformer-based real-time object detector.

    This model provides real-time performance with high accuracy. It supports efficient hybrid encoding, IoU-aware query
    selection, and adaptable inference speed.

    Attributes:
        model (str): Path to the pre-trained DEIM model.

    Examples:
        >>> from ultralytics import DEIM
        >>> model = DEIM("deim-l.pt")
        >>> results = model("image.jpg")
    """

    def __init__(self, model: str = "") -> None:
        """
        Initialize the DEIM model with the given pre-trained model file.

        Args:
            model (str): Path to the pre-trained DEIM model. Supports .pt, .yaml, and .yml formats.

        Raises:
            NotImplementedError: If the model file extension is not 'pt', 'yaml', or 'yml'.
        """
        super().__init__(model=model, task="detect")

    @property
    def task_map(self) -> dict:
        """
        Returns a task map for DEIM, associating tasks with corresponding Ultralytics classes.

        Returns:
            (dict): A dictionary mapping task names to Ultralytics task classes for the DEIM model.
        """
        return {
            "detect": {
                "predictor": RTDETRPredictor,
                "validator": RTDETRValidator,
                "trainer": DEIMModelTrainer,
                "model": DFINEDetectModel,
            }
        }
