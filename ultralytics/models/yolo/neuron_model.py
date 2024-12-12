from ultralytics.engine.neuron_model import NeuronModel
from ultralytics.models import yolo


class NeuronYOLO(NeuronModel):
    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        super().__init__(model, task, verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "predictor": yolo.detect.NeuronDetectionPredictor,
            },
        }
