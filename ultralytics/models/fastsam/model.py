# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path
from typing import Any, Dict, List, Optional

from ultralytics.engine.model import Model

from .predict import FastSAMPredictor
from .val import FastSAMValidator


class FastSAM(Model):
    """
    FastSAM model interface for segment anything tasks.

    This class extends the base Model class to provide specific functionality for the FastSAM (Fast Segment Anything
    Model) implementation, allowing for efficient and accurate image segmentation with optional prompting support.

    Attributes:
        model (str): Path to the pre-trained FastSAM model file.
        task (str): The task type, set to "segment" for FastSAM models.

    Methods:
        predict: Perform segmentation prediction on image or video source with optional prompts.
        task_map: Returns mapping of segment task to predictor and validator classes.

    Examples:
        Initialize FastSAM model and run prediction
        >>> from ultralytics import FastSAM
        >>> model = FastSAM("FastSAM-x.pt")
        >>> results = model.predict("ultralytics/assets/bus.jpg")

        Run prediction with bounding box prompts
        >>> results = model.predict("image.jpg", bboxes=[[100, 100, 200, 200]])
    """

    def __init__(self, model: str = "FastSAM-x.pt"):
        """Initialize the FastSAM model with the specified pre-trained weights."""
        if str(model) == "FastSAM.pt":
            model = "FastSAM-x.pt"
        assert Path(model).suffix not in {".yaml", ".yml"}, "FastSAM models only support pre-trained models."
        super().__init__(model=model, task="segment")

    def predict(
        self,
        source,
        stream: bool = False,
        bboxes: Optional[List] = None,
        points: Optional[List] = None,
        labels: Optional[List] = None,
        texts: Optional[List] = None,
        **kwargs: Any,
    ):
        """
        Perform segmentation prediction on image or video source.

        Supports prompted segmentation with bounding boxes, points, labels, and texts. The method packages these
        prompts and passes them to the parent class predict method for processing.

        Args:
            source (str | PIL.Image | numpy.ndarray): Input source for prediction, can be a file path, URL, PIL image,
                or numpy array.
            stream (bool): Whether to enable real-time streaming mode for video inputs.
            bboxes (List, optional): Bounding box coordinates for prompted segmentation in format [[x1, y1, x2, y2]].
            points (List, optional): Point coordinates for prompted segmentation in format [[x, y]].
            labels (List, optional): Class labels for prompted segmentation.
            texts (List, optional): Text prompts for segmentation guidance.
            **kwargs (Any): Additional keyword arguments passed to the predictor.

        Returns:
            (List): List of Results objects containing the prediction results.
        """
        prompts = dict(bboxes=bboxes, points=points, labels=labels, texts=texts)
        return super().predict(source, stream, prompts=prompts, **kwargs)

    @property
    def task_map(self) -> Dict[str, Dict[str, Any]]:
        """Returns a dictionary mapping segment task to corresponding predictor and validator classes."""
        return {"segment": {"predictor": FastSAMPredictor, "validator": FastSAMValidator}}
