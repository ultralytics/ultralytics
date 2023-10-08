# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
YOLO-NAS model interface.

Example:
    ```python
    from ultralytics import NAS

    model = NAS('yolo_nas_s')
    results = model.predict('ultralytics/assets/bus.jpg')
    ```
"""

from pathlib import Path

import torch

from ultralytics.engine.model import Model
from ultralytics.utils.torch_utils import model_info, smart_inference_mode

from .predict import NASPredictor
from .val import NASValidator


class NAS(Model):

    def __init__(self, model='yolo_nas_s.pt') -> None:
        """Initializes the NAS model with the provided or default 'yolo_nas_s.pt' model."""
        assert Path(model).suffix not in ('.yaml', '.yml'), 'YOLO-NAS models only support pre-trained models.'
        super().__init__(model, task='detect')

    @smart_inference_mode()
    def _load(self, weights: str, task: str):
        """Loads an existing NAS model weights or creates a new NAS model with pretrained weights if not provided."""
        import super_gradients
        suffix = Path(weights).suffix
        if suffix == '.pt':
            self.model = torch.load(weights)
        elif suffix == '':
            self.model = super_gradients.training.models.get(weights, pretrained_weights='coco')
        # Standardize model
        self.model.fuse = lambda verbose=True: self.model
        self.model.stride = torch.tensor([32])
        self.model.names = dict(enumerate(self.model._class_names))
        self.model.is_fused = lambda: False  # for info()
        self.model.yaml = {}  # for info()
        self.model.pt_path = weights  # for export()
        self.model.task = 'detect'  # for export()

    def info(self, detailed=False, verbose=True):
        """
        Logs model info.

        Args:
            detailed (bool): Show detailed information about model.
            verbose (bool): Controls verbosity.
        """
        return model_info(self.model, detailed=detailed, verbose=verbose, imgsz=640)

    @property
    def task_map(self):
        """Returns a dictionary mapping tasks to respective predictor and validator classes."""
        return {'detect': {'predictor': NASPredictor, 'validator': NASValidator}}
