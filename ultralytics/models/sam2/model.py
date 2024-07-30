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
    SAM2 class for real-time image segmentation using the Segment Anything Model (SAM2).

    This class extends the SAM base class, providing an interface to the SAM2 model for promptable segmentation
    tasks. It supports loading pre-trained weights and offers zero-shot performance capabilities.

    Attributes:
        model (torch.nn.Module): The loaded SAM2 model.
        task_map (Dict[str, Type[SAM2Predictor]]): Mapping of 'segment' task to SAM2Predictor.

    Methods:
        __init__: Initializes the SAM2 model with pre-trained weights.
        _load: Loads specified weights into the SAM2 model.

    Examples:
        >>> sam2 = SAM2("sam2_b.pt")
        >>> sam2._load('path/to/sam2_weights.pt')
        >>> task_map = sam2.task_map
        >>> print(task_map)
        {'segment': SAM2Predictor}

    Notes:
        - Supports .pt and .pth file extensions for model weights.
        - Offers zero-shot transfer capabilities for new image distributions and tasks.
    """

    def __init__(self, model="sam2_b.pt") -> None:
        """
        Initializes the SAM2 model with a pre-trained model file.

        Args:
            model (str): Path to the pre-trained SAM2 model file. File should have a .pt or .pth extension.

        Raises:
            NotImplementedError: If the model file extension is not .pt or .pth.

        Examples:
            >>> sam2 = SAM2("sam2_b.pt")
        """
        super().__init__(model=model)

    def _load(self, weights: str, task=None):
        """
        Loads the specified weights into the SAM2 model.

        This method is responsible for loading pre-trained weights into the SAM2 model. It supports loading
        weights from files with .pt or .pth extensions.

        Args:
            weights (str): Path to the weights file. Should be a file with .pt or .pth extension.
            task (str | None): Task name. If provided, it may be used to configure model-specific settings.

        Examples:
            >>> sam2_model = SAM2()
            >>> sam2_model._load('path/to/sam2_weights.pt')
        """
        self.model = build_sam2(weights)

    @property
    def task_map(self):
        """
        Provides a mapping from the 'segment' task to its corresponding 'Predictor'.

        Returns:
            (Dict[str, Type[SAM2Predictor]]): A dictionary mapping the 'segment' task to its corresponding
                SAM2Predictor class.

        Examples:
            >>> sam2 = SAM2()
            >>> task_map = sam2.task_map
            >>> print(task_map)
            {'segment': SAM2Predictor}
        """
        return {"segment": {"predictor": SAM2Predictor}}
