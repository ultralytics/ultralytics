# Ultralytics YOLO 🚀, AGPL-3.0 license
"""RT-DETR model interface."""
from ultralytics.engine.model import Model
from ultralytics.nn.tasks import RTDETRDetectionModel

from .predict import RTDETRPredictor
from .train import RTDETRTrainer
from .val import RTDETRValidator


class RTDETR(Model):
    """RTDETR model interface."""

    def __init__(self, model='rtdetr-l.pt') -> None:
        """Initializes the RTDETR model with the given model file, defaulting to 'rtdetr-l.pt'."""
        if model and model.split('.')[-1] not in ('pt', 'yaml', 'yml'):
            raise NotImplementedError('RT-DETR only supports creating from *.pt file or *.yaml file.')
        super().__init__(model=model, task='detect')

    @property
    def task_map(self):
        """Returns a dictionary mapping task names to corresponding Ultralytics task classes for RTDETR model."""
        return {
            'detect': {
                'predictor': RTDETRPredictor,
                'validator': RTDETRValidator,
                'trainer': RTDETRTrainer,
                'model': RTDETRDetectionModel}}
