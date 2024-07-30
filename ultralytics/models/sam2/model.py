# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
SAM2 model interface.

This module provides an interface to the Segment Anything Model (SAM2) from Ultralytics, designed for real-time image
segmentation tasks. The SAM2 model allows for promptable segmentation with unparalleled versatility in image analysis,
and has been trained on the SA-1B dataset. It features zero-shot performance capabilities, enabling it to adapt to new
image distributions and tasks without prior knowledge.

Key Features:
    - Promptable segmentation
    - Real-time performance
    - Zero-shot transfer capabilities
    - Trained on SA-1B dataset
"""

from ultralytics.models.sam import SAM

from .build import build_sam2
from .predict import SAM2Predictor


class SAM2(SAM):
    """
    SAM2 (Segment Anything Model) interface class.

    SAM2 is designed for promptable real-time image segmentation. It can be used with a variety of prompts such as
    bounding boxes, points, or labels. The model has capabilities for zero-shot performance and is trained on the SA-1B
    dataset.
    """

    def __init__(self, model="sam2_b.pt") -> None:
        """
        Initializes the SAM2 model with a pre-trained model file.

        Args:
            model (str): Path to the pre-trained SAM2 model file. File should have a .pt or .pth extension.

        Raises:
            NotImplementedError: If the model file extension is not .pt or .pth.
        """
        super().__init__(model=model)

    def _load(self, weights: str, task=None):
        """
        Loads the specified weights into the SAM2 model.

        Args:
            weights (str): Path to the weights file.
            task (str, optional): Task name. Defaults to None.
        """
        self.model = build_sam2(weights)

    @property
    def task_map(self):
        """
        Provides a mapping from the 'segment' task to its corresponding 'Predictor'.

        Returns:
            (dict): A dictionary mapping the 'segment' task to its corresponding 'Predictor'.
        """
        return {"segment": {"predictor": SAM2Predictor}}
