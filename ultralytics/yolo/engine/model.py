from pathlib import Path

import torch

from ultralytics import yolo  # noqa
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, SegmentationModel, attempt_load_weights
from ultralytics.yolo.configs import get_config
from ultralytics.yolo.engine.exporter import Exporter
from ultralytics.yolo.utils import DEFAULT_CONFIG, HELP_MSG, LOGGER, yaml_load
from ultralytics.yolo.utils.checks import check_imgsz, check_yaml
from ultralytics.yolo.utils.torch_utils import guess_task_from_head, smart_inference_mode

# Map head to model, trainer, validator, and predictor classes
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
    YOLO

    A python interface which emulates a model-like behaviour by wrapping trainers.
    """

    def __init__(self, model='yolov8n.yaml', type="v8") -> None:
        """
        Initializes the YOLO object.

        Args:
            model (str, Path): model to load or create
            type (str): Type/version of models to use. Defaults to "v8".
        """
        self.type = type
        self.ModelClass = None  # model class
        self.TrainerClass = None  # trainer class
        self.ValidatorClass = None  # validator class
        self.PredictorClass = None  # predictor class
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.task = None  # task type
        self.ckpt = None  # if loaded from *.pt
        self.ckpt_path = None
        self.cfg = None  # if loaded from *.yaml
        self.overrides = {}  # overrides for trainer object
        self.init_disabled = False  # disable model initialization

        # Load or create new YOLO model
        {'.pt': self._load, '.yaml': self._new}[Path(model).suffix](model)

    def _new(self, cfg: str, verbose=True):
        """
        Initializes a new model and infers the task type from the model definitions.

        Args:
            cfg (str): model configuration file
            verbose (bool): display model info on load
        """
        cfg = check_yaml(cfg)  # check YAML
        cfg_dict = yaml_load(cfg)  # model dict
        self.task = guess_task_from_head(cfg_dict["head"][-1][-2])
        self.ModelClass, self.TrainerClass, self.ValidatorClass, self.PredictorClass = \
            self._guess_ops_from_task(self.task)
        self.model = self.ModelClass(cfg_dict, verbose=verbose)  # initialize
        self.cfg = cfg

    def _load(self, weights: str):
        """
        Initializes a new model and infers the task type from the model head

        Args:
            weights (str): model checkpoint to be loaded
        """
        self.model = attempt_load_weights(weights)
        self.ckpt_path = weights
        self.task = self.model.args["task"]
        self.overrides = self.model.args
        self._reset_ckpt_args(self.overrides)
        self.ModelClass, self.TrainerClass, self.ValidatorClass, self.PredictorClass = \
            self._guess_ops_from_task(self.task)

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
            **kwargs : Any other args accepted by the predictors. To see all args check 'configuration' section in docs
        """
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        overrides["mode"] = "predict"
        predictor = self.PredictorClass(overrides=overrides)

        predictor.args.imgsz = check_imgsz(predictor.args.imgsz, min_dim=2)  # check image size
        predictor.setup(model=self.model, source=source)
        predictor()

    @smart_inference_mode()
    def val(self, data=None, **kwargs):
        """
        Validate a model on a given dataset

        Args:
            data (str): The dataset to validate on. Accepts all formats accepted by yolo
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
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
            **kwargs : Any other args accepted by the predictors. To see all args check 'configuration' section in docs
        """

        overrides = self.overrides.copy()
        overrides.update(kwargs)
        args = get_config(config=DEFAULT_CONFIG, overrides=overrides)
        args.task = self.task

        exporter = Exporter(overrides=args)
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
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        if kwargs.get("cfg"):
            LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
            overrides = yaml_load(check_yaml(kwargs["cfg"]))
        overrides["task"] = self.task
        overrides["mode"] = "train"
        if not overrides.get("data"):
            raise AttributeError("dataset not provided! Please define `data` in config.yaml or pass as an argument.")

        if overrides.get("resume"):
            overrides["resume"] = self.ckpt_path
        self.trainer = self.TrainerClass(overrides=overrides)
        if not overrides.get("resume"):
            self.trainer.model = self.trainer.load_model(weights=self.model,
                                                         model_cfg=self.model.yaml if self.task != "classify" else None)
            self.model = self.trainer.model  # override here to save memory

        self.trainer.train()

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
        device = next(self.model.parameters()).device  # get model device
        return self.model(imgs.to(device))

    def forward(self, imgs):
        return self.__call__(imgs)

    @staticmethod
    def _reset_ckpt_args(args):
        args.pop("device", None)
        args.pop("project", None)
        args.pop("name", None)
        args.pop("batch", None)
        args.pop("epochs", None)
