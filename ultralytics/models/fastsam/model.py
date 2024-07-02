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
<<<<<<< HEAD
        assert Path(model).suffix not in {".yaml", ".yml"}, "FastSAM models only support pre-trained models."
=======
        assert Path(model).suffix not in (".yaml", ".yml"), "FastSAM models only support pre-trained models."
>>>>>>> 2d87fb01604a79af96d1d3778626415fb4b54ac9
        super().__init__(model=model, task="segment")

    @property
    def task_map(self):
        """Returns a dictionary mapping segment task to corresponding predictor and validator classes."""
        return {"segment": {"predictor": FastSAMPredictor, "validator": FastSAMValidator}}
