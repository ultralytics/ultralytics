"""
Top-level YOLO model interface. First principle usage example - https://github.com/ultralytics/ultralytics/issues/13
"""
import torch
import yaml

from ultralytics.yolo.utils import LOGGER
from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.yolo.utils.modeling import attempt_load_weights
from ultralytics.yolo.utils.modeling.tasks import ClassificationModel, DetectionModel, SegmentationModel

# map head: [model, trainer]
MODEL_MAP = {
    "classify": [ClassificationModel, 'yolo.VERSION.classify.ClassificationTrainer'],
    "detect": [DetectionModel, 'yolo.VERSION.detect.DetectionTrainer'],
    "segment": [SegmentationModel, 'yolo.VERSION.segment.SegmentationTrainer']}


class YOLO:

    def __init__(self, version=8) -> None:
        self.version = version
        self.ModelClass = None
        self.TrainerClass = None
        self.model = None
        self.trainer = None
        self.task = None
        self.ckpt = None

    def new(self, cfg: str):
        cfg = check_yaml(cfg)  # check YAML
        with open(cfg, encoding='ascii', errors='ignore') as f:
            cfg = yaml.safe_load(f)  # model dict
        self.ModelClass, self.TrainerClass, self.task = self._guess_model_trainer_and_task(cfg["head"][-1][-2])
        self.model = self.ModelClass(cfg)  # initialize

    def load(self, weights):
        self.ckpt = torch.load(weights, map_location="cpu")
        self.task = self.ckpt["train_args"]["task"]
        _, trainer_class_literal = MODEL_MAP[self.task]
        self.TrainerClass = eval(trainer_class_literal.replace("VERSION", f"v{self.version}"))
        self.model = attempt_load_weights(weights)

    def reset(self):
        for m in self.model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True

    def train(self, **kwargs):
        if 'data' not in kwargs:
            raise Exception("data is required to train")
        if not self.model and not self.ckpt:
            raise Exception("model not initialized. Use .new() or .load()")

        kwargs["task"] = self.task
        kwargs["mode"] = "train"
        self.trainer = self.TrainerClass(overrides=kwargs)
        # load pre-trained weights if found, else use the loaded model
        self.trainer.model = self.trainer.load_model(weights=self.ckpt) if self.ckpt else self.model
        self.trainer.train()

    def resume(self, task=None, model=None):
        if not task:
            raise Exception(
                "pass the task type and/or model(optional) from which you want to resume: `model.resume(task="
                ")`")
        if task.lower() not in MODEL_MAP:
            raise Exception(f"unrecognised task - {task}. Supported tasks are {MODEL_MAP.keys()}")
        _, trainer_class_literal = MODEL_MAP[task.lower()]
        self.TrainerClass = eval(trainer_class_literal.replace("VERSION", f"v{self.version}"))
        self.trainer = self.TrainerClass(overrides={"task": task.lower(), "resume": model if model else True})
        self.trainer.train()

    def _guess_model_trainer_and_task(self, head):
        # TODO: warn
        task = None
        if head.lower() in ["classify", "classifier", "cls", "fc"]:
            task = "classify"
        if head.lower() in ["detect"]:
            task = "detect"
        if head.lower() in ["segment"]:
            task = "segment"
        model_class, trainer_class = MODEL_MAP[task]
        # warning: eval is unsafe. Use with caution
        trainer_class = eval(trainer_class.replace("VERSION", f"v{self.version}"))

        return model_class, trainer_class, task

    def __call__(self, imgs):
        if not self.model:
            LOGGER.info("model not initialized!")
        return self.model(imgs)
