# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

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

        model = FastSAM("last.pt")
        results = model.predict("ultralytics/assets/bus.jpg")
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
        Perform segmentation prediction on image or video source.

        Supports prompted segmentation with bounding boxes, points, labels, and texts.

        Args:
            source (str | PIL.Image | numpy.ndarray): Input source.
            stream (bool): Enable real-time streaming.
            bboxes (list): Bounding box coordinates for prompted segmentation.
            points (list): Points for prompted segmentation.
            labels (list): Labels for prompted segmentation.
            texts (list): Texts for prompted segmentation.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            (list): Model predictions.
        """
        prompts = dict(bboxes=bboxes, points=points, labels=labels, texts=texts)
        return super().predict(source, stream, prompts=prompts, **kwargs)

    @property
    def task_map(self):
        """Returns a dictionary mapping segment task to corresponding predictor and validator classes."""
        return {"segment": {"predictor": FastSAMPredictor, "validator": FastSAMValidator}}
