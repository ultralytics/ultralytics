from ultralytics.engine.predictor import BasePredictor
from ultralytics.nn.neuron_autobackend import NeuronAutoBackend
from ultralytics.utils.torch_utils import select_device


class NeuronPredictor(BasePredictor):
    def __init__(self, cfg=..., overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)

    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        self.model = NeuronAutoBackend(
            weights=model or self.args.model,
            device=select_device(self.args.device, verbose=verbose),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            batch=self.args.batch,
            fuse=True,
            verbose=verbose,
        )

        self.device = self.model.device  # update device
        self.args.half = self.model.fp16  # update half
        self.model.eval()
