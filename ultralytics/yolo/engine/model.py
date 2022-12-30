from pathlib import Path

import torch

from ultralytics import yolo  # noqa required for python usage
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, SegmentationModel, attempt_load_weights
from ultralytics.yolo.configs import get_config
from ultralytics.yolo.engine.exporter import Exporter
from ultralytics.yolo.utils import DEFAULT_CONFIG, HELP_MSG, LOGGER
from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.yolo.utils.files import yaml_load
from ultralytics.yolo.utils.torch_utils import smart_inference_mode

# map head: [model, trainer, validator, predictor]
MODEL_MAP = {
    "classify": [
        ClassificationModel, 'yolo.TYPE.classify.ClassificationTrainer', 'yolo.TYPE.classify.ClassificationValidator',
        'yolo.TYPE.classify.ClassificationPredictor'],
    "detect": [
        DetectionModel, 'yolo.TYPE.detect.DetectionTrainer', 'yolo.TYPE.detect.DetectionValidator',
        'yolo.TYPE.detect.DetectionPredictor'],
    "segment": [
        SegmentationModel, 'yolo.TYPE.segment.SegmentationTrainer', 'yolo.TYPE.segment.SegmentationValidator',
        'yolo.TYPE.segment.SegmentationPredictor']}


class YOLO:
    """
    Python interface which emulates a model-like behaviour by wrapping trainers.
    """
    __init_key = object()

    def __init__(self, init_key=None, type="v8") -> None:
        """
        Args:
            type (str): Type/version of models to use
        """
        if init_key != YOLO.__init_key:
            raise SyntaxError(HELP_MSG)

        self.type = type
        self.ModelClass = None
        self.TrainerClass = None
        self.ValidatorClass = None
        self.PredictorClass = None
        self.model = None
        self.trainer = None
        self.task = None
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.overrides = {}
        self.init_disabled = False

    @classmethod
    def new(cls, cfg: str, verbose=True):
        """
        Initializes a new model and infers the task type from the model definitions

        Args:
            cfg (str): model configuration file
            verbsoe (bool): display model info on load
        """
        cfg = check_yaml(cfg)  # check YAML
        cfg_dict = yaml_load(cfg)  # model dict
        obj = cls(init_key=cls.__init_key)
        obj.task = obj._guess_task_from_head(cfg_dict["head"][-1][-2])
        obj.ModelClass, obj.TrainerClass, obj.ValidatorClass, obj.PredictorClass = obj._guess_ops_from_task(obj.task)
        obj.model = obj.ModelClass(cfg_dict, verbose=verbose)  # initialize
        obj.cfg = cfg

        return obj

    @classmethod
    def load(cls, weights: str):
        """
        Initializes a new model and infers the task type from the model head

        Args:
            weights (str): model checkpoint to be loaded

        """
        obj = cls(init_key=cls.__init_key)
        obj.ckpt = torch.load(weights, map_location="cpu")
        obj.task = obj.ckpt["train_args"]["task"]
        obj.overrides = dict(obj.ckpt["train_args"])
        obj.overrides["device"] = ''  # reset device
        LOGGER.info("Device has been reset to ''")

        obj.ModelClass, obj.TrainerClass, obj.ValidatorClass, obj.PredictorClass = obj._guess_ops_from_task(
            task=obj.task)
        obj.model = attempt_load_weights(weights)

        return obj

    def reset(self):
        """
        Resets the model modules .
        """
        for m in self.model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True

    def info(self, verbose=False):
        """
        Logs model info

        Args:
        verbose (bool): Controls verbosity.
        """
        if not self.model:
            LOGGER.info("model not initialized!")
        self.model.info(verbose=verbose)

    def fuse(self):
        if not self.model:
            LOGGER.info("model not initialized!")
        self.model.fuse()

    @smart_inference_mode()
    def predict(self, source, **kwargs):
        """
        Visualize prediction.

        Args:
        source (str): Accepts all source types accepted by yolo
        **kwargs : Any other args accepted by the predictors. To see all args check 'configuration' section in the docs
        """
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        overrides["mode"] = "predict"
        predictor = self.PredictorClass(overrides=overrides)

        # check size type
        sz = predictor.args.imgsz
        if type(sz) != int:  # received listConfig
            predictor.args.imgsz = [sz[0], sz[0]] if len(sz) == 1 else [sz[0], sz[1]]  # expand
        else:
            predictor.args.imgsz = [sz, sz]

        predictor.setup(model=self.model, source=source)
        predictor()

    @smart_inference_mode()
    def val(self, data=None, **kwargs):
        """
        Validate a model on a given dataset

        Args:
        data (str): The dataset to validate on. Accepts all formats accepted by yolo
        kwargs: Any other args accepted by the validators. To see all args check 'configuration' section in the docs
        """
        if not self.model:
            raise ModuleNotFoundError("model not initialized!")

        overrides = self.overrides.copy()
        overrides.update(kwargs)
        overrides["mode"] = "val"
        args = get_config(config=DEFAULT_CONFIG, overrides=overrides)
        args.data = data or args.data
        args.task = self.task

        validator = self.ValidatorClass(args=args)
        validator(model=self.model)

    @smart_inference_mode()
    def export(self, **kwargs):
        """
        Export model.

        Args:
        format (str): Export format
        **kwargs : Any other args accepted by the predictors. To see all args check 'configuration' section in the docs
        """

        overrides = self.overrides.copy()
        overrides.update(kwargs)
        args = get_config(config=DEFAULT_CONFIG, overrides=overrides)
        args.task = self.task

        exporter = Exporter(overrides=overrides)
        exporter(model=self.model)

    def train(self, **kwargs):
        """
        Trains the model on given dataset.

        Args:
            **kwargs (Any): Any number of arguments representing the training configuration. List of all args can be found in 'config' section.
                            You can pass all arguments as a yaml file in `cfg`. Other args are ignored if `cfg` file is passed
        """
        if not self.model:
            raise AttributeError("model not initialized. Use .new() or .load()")

        overrides = kwargs
        if kwargs.get("cfg"):
            LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
            overrides = yaml_load(check_yaml(kwargs["cfg"]))
        overrides["task"] = self.task
        overrides["mode"] = "train"
        if not overrides.get("data"):
            raise AttributeError("dataset not provided! Please define `data` in config.yaml or pass as an argument.")

        self.trainer = self.TrainerClass(overrides=overrides)
        self.trainer.model = self.trainer.load_model(weights=self.ckpt,
                                                     model_cfg=self.model.yaml if self.task != "classify" else None)
        self.model = self.trainer.model  # override here to save memory

        self.trainer.train()

    def resume(self, task=None, model=None):
        """
        Resume a training task. Requires either `task` or `model`. `model` takes the higher precedence.
        Args:
            task (str): The task type you want to resume. Automatically finds the last run to resume if `model` is not specified.
            model (str): The model checkpoint to resume from. If not found, the last run of the given task type is resumed.
                         If `model` is specified
        """
        if task:
            if task.lower() not in MODEL_MAP:
                raise SyntaxError(f"unrecognised task - {task}. Supported tasks are {MODEL_MAP.keys()}")
        else:
            ckpt = torch.load(model, map_location="cpu")
            task = ckpt["train_args"]["task"]
            del ckpt
        self.ModelClass, self.TrainerClass, self.ValidatorClass, self.PredictorClass = self._guess_ops_from_task(
            task=task.lower())
        self.trainer = self.TrainerClass(overrides={"task": task.lower(), "resume": model or True})

        self.trainer.train()

    @staticmethod
    def _guess_task_from_head(head):
        task = None
        if head.lower() in ["classify", "classifier", "cls", "fc"]:
            task = "classify"
        if head.lower() in ["detect"]:
            task = "detect"
        if head.lower() in ["segment"]:
            task = "segment"

        if not task:
            raise SyntaxError("task or model not recognized! Please refer the docs at : ")  # TODO: add docs links

        return task

    def to(self, device):
        self.model.to(device)

    def _guess_ops_from_task(self, task):
        model_class, train_lit, val_lit, pred_lit = MODEL_MAP[task]
        # warning: eval is unsafe. Use with caution
        trainer_class = eval(train_lit.replace("TYPE", f"{self.type}"))
        validator_class = eval(val_lit.replace("TYPE", f"{self.type}"))
        predictor_class = eval(pred_lit.replace("TYPE", f"{self.type}"))

        return model_class, trainer_class, validator_class, predictor_class

    @smart_inference_mode()
    def __call__(self, imgs):
        if not self.model:
            LOGGER.info("model not initialized!")
        return self.model(imgs)

    def forward(self, imgs):
        return self.__call__(imgs)
