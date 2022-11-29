"""
Top-level YOLO model interface. First principle usage example - https://github.com/ultralytics/ultralytics/issues/13
"""
import yaml

from ultralytics.yolo.utils import LOGGER
from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.yolo.utils.modeling import get_model
from ultralytics.yolo.utils.modeling.tasks import ClassificationModel, DetectionModel, SegmentationModel

# map head: [model, trainer]
MODEL_MAP = {
    "classify": [ClassificationModel, 'yolo.VERSION.classify.train.ClassificationTrainer'],
    "detect": [ClassificationModel, 'yolo.VERSION.classify.train.ClassificationTrainer'],  # temp
    "segment": []}


class YOLO:

    def __init__(self, task=None, version=8) -> None:
        self.version = version
        self.ModelClass = None
        self.TrainerClass = None
        self.model = None
        self.pretrained_weights = None
        if task:
            if task.lower() not in MODEL_MAP:
                raise Exception(f"Unsupported task {task}. The supported tasks are: \n {MODEL_MAP.keys()}")
            self.ModelClass, self.TrainerClass = MODEL_MAP[task]
            self.TrainerClass = eval(self.trainer.replace("VERSION", f"v{self.version}"))

    def new(self, cfg: str):
        cfg = check_yaml(cfg)  # check YAML
        if self.model:
            self.model = self.model(cfg)
        else:
            with open(cfg, encoding='ascii', errors='ignore') as f:
                cfg = yaml.safe_load(f)  # model dict
            self.ModelClass, self.TrainerClass = self._get_model_and_trainer(cfg["head"])
            self.model = self.ModelClass(cfg)  # initialize

    def load(self, weights, autodownload=True):
        if not isinstance(self.pretrained_weights, type(None)):
            LOGGER.info("Overwriting weights")
        # TODO: weights = smart_file_loader(weights)
        if self.model:
            self.model.load(weights)
            LOGGER.info("Checkpoint loaded successfully")
        else:
            self.model = get_model(weights)
            self.ModelClass, self.TrainerClass = self._guess_model_and_trainer(list(self.model.named_children()))
        self.pretrained_weights = weights

    def reset(self):
        for m in self.model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True

    def train(self, **kwargs):
        if 'data' not in kwargs:
            raise Exception("data is required to train")
        if not self.model:
            raise Exception("model not initialized. Use .new() or .load()")
        # kwargs["model"] = self.model
        trainer = self.TrainerClass(overrides=kwargs)
        trainer.model = self.model
        trainer.train()

    def _guess_model_and_trainer(self, cfg):
        # TODO: warn
        head = cfg[-1][-2]
        if head.lower() in ["classify", "classifier", "cls", "fc"]:
            task = "classify"
        if head.lower() in ["detect"]:
            task = "detect"
        if head.lower() in ["segment"]:
            task = "segment"
        model_class, trainer_class = MODEL_MAP[task]
        # warning: eval is unsafe. Use with caution
        trainer_class = eval(trainer_class.replace("VERSION", f"v{self.version}"))

        return model_class, trainer_class


if __name__ == "__main__":
    model = YOLO()
    # model.new("assets/dummy_model.yaml")
    model.load("yolov5n-cls.pt")
    model.train(data="imagenette160", epochs=1, lr0=0.01)
