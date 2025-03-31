# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

from ultralytics.engine.model import Model

from .predict import FastSAMPredictor
from .val import FastSAMValidator


class FastSAM(Model):
    """
    FastSAM model interface for segment anything tasks.

    This class extends the base Model class to provide specific functionality for the FastSAM (Fast Segment Anything Model)
    implementation, allowing for efficient and accurate image segmentation.

    Attributes:
        model (str): Path to the pre-trained FastSAM model file.
        task (str): The task type, set to "segment" for FastSAM models.

    Examples:
        >>> from ultralytics import FastSAM
        >>> model = FastSAM("last.pt")
        >>> results = model.predict("ultralytics/assets/bus.jpg")
    """

    def __init__(self, model="FastSAM-x.pt"):
        """Initialize the FastSAM model with the specified pre-trained weights."""
        if str(model) == "FastSAM.pt":
            model = "FastSAM-x.pt"
        assert Path(model).suffix not in {".yaml", ".yml"}, "FastSAM models only support pre-trained models."
        super().__init__(model=model, task="segment")

    def predict(self, source, stream=False, bboxes=None, points=None, labels=None, texts=None, **kwargs):
        """
        Perform segmentation prediction on image or video source.

        Supports prompted segmentation with bounding boxes, points, labels, and texts. The method packages these
        prompts and passes them to the parent class predict method.

        Args:
            source (str | PIL.Image | numpy.ndarray): Input source for prediction, can be a file path, URL, PIL image,
                or numpy array.
            stream (bool): Whether to enable real-time streaming mode for video inputs.
            bboxes (list): Bounding box coordinates for prompted segmentation in format [[x1, y1, x2, y2], ...].
            points (list): Point coordinates for prompted segmentation in format [[x, y], ...].
            labels (list): Class labels for prompted segmentation.
            texts (list): Text prompts for segmentation guidance.
            **kwargs (Any): Additional keyword arguments passed to the predictor.

        Returns:
            (list): List of Results objects containing the prediction results.
        """
        prompts = dict(bboxes=bboxes, points=points, labels=labels, texts=texts)
        return super().predict(source, stream, prompts=prompts, **kwargs)

    @property
    def task_map(self):
        """Returns a dictionary mapping segment task to corresponding predictor and validator classes."""
        return {"segment": {"predictor": FastSAMPredictor, "validator": FastSAMValidator}}
