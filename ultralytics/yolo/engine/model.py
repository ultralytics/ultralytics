import torch
import yaml

from ultralytics import yolo
from ultralytics.yolo.utils import LOGGER
from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.yolo.utils.files import yaml_load
from ultralytics.yolo.utils.modeling import attempt_load_weights
from ultralytics.yolo.utils.modeling.tasks import ClassificationModel, DetectionModel, SegmentationModel

# map head: [model, trainer]
MODEL_MAP = {
    "classify": [ClassificationModel, 'yolo.TYPE.classify.ClassificationTrainer'],
    "detect": [DetectionModel, 'yolo.TYPE.detect.DetectionTrainer'],
    "segment": [SegmentationModel, 'yolo.TYPE.segment.SegmentationTrainer']}


class YOLO:
    """
    Python interface which emulates a model-like behaviour by wrapping trainers.
    """

    def __init__(self, type="v8") -> None:
        """
        Args:
            type (str): Type/version of models to use
        """
        self.type = type
        self.ModelClass = None
        self.TrainerClass = None
        self.model = None
        self.trainer = None
        self.task = None
        self.ckpt = None

    def new(self, cfg: str):
        """
        Initializes a new model and infers the task type from the model definitions

        Args:
            cfg (str): model configuration file
        """
        cfg = check_yaml(cfg)  # check YAML
        with open(cfg, encoding='ascii', errors='ignore') as f:
            cfg = yaml.safe_load(f)  # model dict
        self.ModelClass, self.TrainerClass, self.task = self._guess_model_trainer_and_task(cfg["head"][-1][-2])
        self.model = self.ModelClass(cfg)  # initialize

    def load(self, weights: str):
        """
        Initializes a new model and infers the task type from the model head

        Args:
            weights (str): model checkpoint to be loaded

        """
        self.ckpt = torch.load(weights, map_location="cpu")
        self.task = self.ckpt["train_args"]["task"]
        _, trainer_class_literal = MODEL_MAP[self.task]
        self.TrainerClass = eval(trainer_class_literal.replace("TYPE", f"v{self.type}"))
        self.model = attempt_load_weights(weights)

    def reset(self):
        """
        Resets the model modules .
        """
        for m in self.model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True

    def train(self, **kwargs):
        """
        Trains the model on given dataset.

        Args:
            **kwargs (Any): Any number of arguments representing the training configuration. List of all args can be found in 'config' section.
                            You can pass all arguments as a yaml file in `cfg`. Other args are ignored if `cfg` file is passed
        """
        if not self.model and not self.ckpt:
            raise Exception("model not initialized. Use .new() or .load()")

        overrides = kwargs
        if kwargs.get("cfg"):
            LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
            overrides = yaml_load(check_yaml(kwargs["cfg"]))
        overrides["task"] = self.task
        overrides["mode"] = "train"
        if not overrides.get("data"):
            raise Exception("dataset not provided! Please check if you have defined `data` in you configs")

        self.trainer = self.TrainerClass(overrides=overrides)
        # load pre-trained weights if found, else use the loaded model
        self.trainer.model = self.trainer.load_model(weights=self.ckpt) if self.ckpt else self.model
        self.trainer.train()

    def resume(self, task, model=None):
        """
        Resume a training task.

        Args:
            task (str): The task type you want to resume. Automatically finds the last run to resume if `model` is not specified.
            model (str): [Optional] The model checkpoint to resume from. If not found, the last run of the given task type is resumed.
        """
        if task.lower() not in MODEL_MAP:
            raise Exception(f"unrecognised task - {task}. Supported tasks are {MODEL_MAP.keys()}")
        _, trainer_class_literal = MODEL_MAP[task.lower()]
        self.TrainerClass = eval(trainer_class_literal.replace("TYPE", f"v{self.type}"))
        self.trainer = self.TrainerClass(overrides={"task": task.lower(), "resume": model if model else True})
        self.trainer.train()

    def _guess_model_trainer_and_task(self, head):
        task = None
        if head.lower() in ["classify", "classifier", "cls", "fc"]:
            task = "classify"
        if head.lower() in ["detect"]:
            task = "detect"
        if head.lower() in ["segment"]:
            task = "segment"
        model_class, trainer_class = MODEL_MAP[task]
        # warning: eval is unsafe. Use with caution
        trainer_class = eval(trainer_class.replace("TYPE", f"{self.type}"))

        return model_class, trainer_class, task

    def __call__(self, imgs):
        if not self.model:
            LOGGER.info("model not initialized!")
        return self.model(imgs)
