# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
SAM model interface.

This module provides an interface to the Segment Anything Model (SAM) from Ultralytics, designed for real-time image
segmentation tasks. The SAM model allows for promptable segmentation with unparalleled versatility in image analysis,
and has been trained on the SA-1B dataset. It features zero-shot performance capabilities, enabling it to adapt to new
image distributions and tasks without prior knowledge.

Key Features:
    - Promptable segmentation
    - Real-time performance
    - Zero-shot transfer capabilities
    - Trained on SA-1B dataset
"""

from pathlib import Path
from typing import Dict, Type

from ultralytics.engine.model import Model
from ultralytics.utils.torch_utils import model_info

from .predict import Predictor, SAM2Predictor


class SAM(Model):
    """
    SAM (Segment Anything Model) interface class for real-time image segmentation tasks.

    This class provides an interface to the Segment Anything Model (SAM) from Ultralytics, designed for
    promptable segmentation with versatility in image analysis. It supports various prompts such as bounding
    boxes, points, or labels, and features zero-shot performance capabilities.

    Attributes:
        model (torch.nn.Module): The loaded SAM model.
        is_sam2 (bool): Indicates whether the model is SAM2 variant.
        task (str): The task type, set to "segment" for SAM models.

    Methods:
        predict: Perform segmentation prediction on the given image or video source.
        info: Log information about the SAM model.

    Examples:
        >>> sam = SAM("sam_b.pt")
        >>> results = sam.predict("image.jpg", points=[[500, 375]])
        >>> for r in results:
        >>>     print(f"Detected {len(r.masks)} masks")
    """

    def __init__(self, model: str = "sam_b.pt") -> None:
        """
        Initialize the SAM (Segment Anything Model) instance.

        Args:
            model (str): Path to the pre-trained SAM model file. File should have a .pt or .pth extension.

        Raises:
            NotImplementedError: If the model file extension is not .pt or .pth.

        Examples:
            >>> sam = SAM("sam_b.pt")
            >>> print(sam.is_sam2)
        """
        if model and Path(model).suffix not in {".pt", ".pth"}:
            raise NotImplementedError("SAM prediction requires pre-trained *.pt or *.pth model.")
        self.is_sam2 = "sam2" in Path(model).stem
        super().__init__(model=model, task="segment")

    def _load(self, weights: str, task=None):
        """
        Load the specified weights into the SAM model.

        Args:
            weights (str): Path to the weights file. Should be a .pt or .pth file containing the model parameters.
            task (str | None): Task name. If provided, it specifies the particular task the model is being loaded for.

        Examples:
            >>> sam = SAM("sam_b.pt")
            >>> sam._load("path/to/custom_weights.pt")
        """
        from .build import build_sam  # slow import

        self.model = build_sam(weights)

    def predict(self, source, stream: bool = False, bboxes=None, points=None, labels=None, **kwargs):
        """
        Perform segmentation prediction on the given image or video source.

        Args:
            source (str | PIL.Image | numpy.ndarray): Path to the image or video file, or a PIL.Image object, or
                a numpy.ndarray object.
            stream (bool): If True, enables real-time streaming.
            bboxes (List[List[float]] | None): List of bounding box coordinates for prompted segmentation.
            points (List[List[float]] | None): List of points for prompted segmentation.
            labels (List[int] | None): List of labels for prompted segmentation.
            **kwargs (Any): Additional keyword arguments for prediction.

        Returns:
            (list): The model predictions.

        Examples:
            >>> sam = SAM("sam_b.pt")
            >>> results = sam.predict("image.jpg", points=[[500, 375]])
            >>> for r in results:
            ...     print(f"Detected {len(r.masks)} masks")
        """
        overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024)
        kwargs = {**overrides, **kwargs}
        prompts = dict(bboxes=bboxes, points=points, labels=labels)
        return super().predict(source, stream, prompts=prompts, **kwargs)

    def __call__(self, source=None, stream: bool = False, bboxes=None, points=None, labels=None, **kwargs):
        """
        Perform segmentation prediction on the given image or video source.

        This method is an alias for the 'predict' method, providing a convenient way to call the SAM model
        for segmentation tasks.

        Args:
            source (str | PIL.Image | numpy.ndarray | None): Path to the image or video file, or a PIL.Image
                object, or a numpy.ndarray object.
            stream (bool): If True, enables real-time streaming.
            bboxes (List[List[float]] | None): List of bounding box coordinates for prompted segmentation.
            points (List[List[float]] | None): List of points for prompted segmentation.
            labels (List[int] | None): List of labels for prompted segmentation.
            **kwargs (Any): Additional keyword arguments to be passed to the predict method.

        Returns:
            (list): The model predictions, typically containing segmentation masks and other relevant information.

        Examples:
            >>> sam = SAM("sam_b.pt")
            >>> results = sam("image.jpg", points=[[500, 375]])
            >>> print(f"Detected {len(results[0].masks)} masks")
        """
        return self.predict(source, stream, bboxes, points, labels, **kwargs)

    def info(self, detailed: bool = False, verbose: bool = True):
        """
        Log information about the SAM model.

        Args:
            detailed (bool): If True, displays detailed information about the model layers and operations.
            verbose (bool): If True, prints the information to the console.

        Returns:
            (tuple): A tuple containing the model's information (string representations of the model).

        Examples:
            >>> sam = SAM("sam_b.pt")
            >>> info = sam.info()
            >>> print(info[0])  # Print summary information
        """
        return model_info(self.model, detailed=detailed, verbose=verbose)

    @property
    def task_map(self) -> Dict[str, Dict[str, Type[Predictor]]]:
        """
        Provide a mapping from the 'segment' task to its corresponding 'Predictor'.

        Returns:
            (Dict[str, Dict[str, Type[Predictor]]]): A dictionary mapping the 'segment' task to its corresponding
                Predictor class. For SAM2 models, it maps to SAM2Predictor, otherwise to the standard Predictor.

        Examples:
            >>> sam = SAM("sam_b.pt")
            >>> task_map = sam.task_map
            >>> print(task_map)
            {'segment': {'predictor': <class 'ultralytics.models.sam.predict.Predictor'>}}
        """
        return {"segment": {"predictor": SAM2Predictor if self.is_sam2 else Predictor}}
