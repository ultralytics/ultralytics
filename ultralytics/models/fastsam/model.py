# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

from ultralytics.engine.model import Model

from .predict import FastSAMPredictor
from .val import FastSAMValidator


class FastSAM(Model):
    """
    FastSAM model interface.

    Example:
        ```python
        from ultralytics import FastSAM

        model = FastSAM('last.pt')
        results = model.predict('ultralytics/assets/bus.jpg')
        ```
    """

    def __init__(self, model="FastSAM-x.pt"):
        """Call the __init__ method of the parent class (YOLO) with the updated default model."""
        if str(model) == "FastSAM.pt":
            model = "FastSAM-x.pt"
        assert Path(model).suffix not in {".yaml", ".yml"}, "FastSAM models only support pre-trained models."
        super().__init__(model=model, task="segment")

    def predict(self, source, stream=False, bboxes=None, points=None, labels=None, texts=None, **kwargs):
        """
        Performs segmentation prediction on the given image or video source.

        Args:
            source (str): Path to the image or video file, or a PIL.Image object, or a numpy.ndarray object.
            stream (bool, optional): If True, enables real-time streaming. Defaults to False.
            bboxes (list, optional): List of bounding box coordinates for prompted segmentation. Defaults to None.
            points (list, optional): List of points for prompted segmentation. Defaults to None.
            labels (list, optional): List of labels for prompted segmentation. Defaults to None.
            texts (list, optional): List of texts for prompted segmentation. Defaults to None.

        Returns:
            (list): The model predictions.
        """
        prompts = dict(bboxes=bboxes, points=points, labels=labels, texts=texts)
        return super().predict(source, stream, prompts=prompts, **kwargs)

    @property
    def task_map(self):
        """Returns a dictionary mapping segment task to corresponding predictor and validator classes."""
        return {"segment": {"predictor": FastSAMPredictor, "validator": FastSAMValidator}}
