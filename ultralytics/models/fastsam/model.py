# Ultralytics YOLO 🚀, AGPL-3.0 license
from ultralytics.engine.model import Model
from pathlib import Path

from .predict import FastSAMPredictor
from .val import FastSAMValidator


class FastSAM(Model):
    """
    FastSAM model interface.

    Usage - Predict:
        from ultralytics import FastSAM

        model = FastSAM('last.pt')
        results = model.predict('ultralytics/assets/bus.jpg')
    """
    def __init__(self, model="FastSAM-x.pt"):
        """Call the __init__ method of the parent class (YOLO) with the updated default model"""
        if model == "FastSAM.pt":
            model = "FastSAM-x.pt"
        assert Path(model).suffix != ".yaml", "FastSAM models only support pre-trained models."
        super().__init__(model=model, task="segment")
        # any additional initialization code for FastSAM

    def train(self, **kwargs):
        """Function trains models but raises an error as FastSAM models do not support training."""
        raise NotImplementedError("FastSAM models don't support training")

    @property
    def task_map(self):
        return {"segment": {"predictor": FastSAMPredictor, "validator": FastSAMValidator}}
