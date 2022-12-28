import torch
import yaml

from ultralytics import yolo  # noqa required for python usage
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, SegmentationModel, attempt_load_weights
# from ultralytics.yolo.data.utils import check_dataset, check_dataset_yaml
from ultralytics.yolo.engine.trainer import DEFAULT_CONFIG
from ultralytics.yolo.utils import LOGGER
from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.yolo.utils.configs import get_config
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

    def __init__(self, type="v8") -> None:
        """
        Args:
            type (str): Type/version of models to use
        """
        self.type = type
        self.ModelClass = None
        self.TrainerClass = None
        self.ValidatorClass = None
        self.PredictorClass = None
        self.model = None
        self.trainer = None
        self.task = None
        self.ckpt = None
        self.overrides = {}

    def new(self, cfg: str):
        """
        Initializes a new model and infers the task type from the model definitions

        Args:
            cfg (str): model configuration file
        """
        cfg = check_yaml(cfg)  # check YAML
        with open(cfg, encoding='ascii', errors='ignore') as f:
            cfg = yaml.safe_load(f)  # model dict
        self.task = self._guess_task_from_head(cfg["head"][-1][-2])
        self.ModelClass, self.TrainerClass, self.ValidatorClass, self.PredictorClass = self._guess_ops_from_task(
            self.task)
        self.model = self.ModelClass(cfg)  # initialize

    def load(self, weights: str):
        """
        Initializes a new model and infers the task type from the model head

        Args:
            weights (str): model checkpoint to be loaded

        """
        self.ckpt = torch.load(weights, map_location="cpu")
        self.task = self.ckpt["train_args"]["task"]
        self.overrides = dict(self.ckpt["train_args"])
        self.overrides["device"] = ''  # reset device
        LOGGER.info("Device has been reset to ''")

        self.ModelClass, self.TrainerClass, self.ValidatorClass, self.PredictorClass = self._guess_ops_from_task(
            task=self.task)
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

    def predict(self, source, **kwargs):
        """
        Visualize prection.

        Args:
        source (str): Accepts all source types accepted by yolo
        **kwargs : Any other args accepted by the predictors. Too see all args check 'configuration' section in the docs
        """
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        predictor = self.PredictorClass(overrides=overrides)

        # check size type
        sz = predictor.args.imgsz
        if type(sz) != int:  # recieved listConfig
            predictor.args.imgsz = [sz[0], sz[0]] if len(sz) == 1 else [sz[0], sz[1]]  # expand
        else:
            predictor.args.imgsz = [sz, sz]

        predictor.setup(model=self.model, source=source)
        predictor()

    def val(self, data=None, **kwargs):
        """
        Validate a model on a given dataset

        Args:
        data (str): The dataset to validate on. Accepts all formats accepted by yolo
        kwargs: Any other args accepted by the validators. Too see all args check 'configuration' section in the docs
        """
        if not self.model:
            raise Exception("model not initialized!")

        overrides = self.overrides.copy()
        overrides.update(kwargs)
        args = get_config(config=DEFAULT_CONFIG, overrides=overrides)
        args.data = data or args.data
        args.task = self.task

        validator = self.ValidatorClass(args=args)
        validator(model=self.model)

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
            raise AttributeError("dataset not provided! Please check if you have defined `data` in you configs")

        self.trainer = self.TrainerClass(overrides=overrides)
        self.trainer.model = self.trainer.load_model(weights=self.ckpt,
                                                     model_cfg=self.model.yaml if self.task != "classify" else None)
        self.model = self.trainer.model  # override here to save memory

        self.trainer.train()

    def resume(self, task=None, model=None):
        """
        Resume a training task. Requires either `task` or `model`. `model` takes the higher precederence.
        Args:
            task (str): The task type you want to resume. Automatically finds the last run to resume if `model` is not specified.
            model (str): The model checkpoint to resume from. If not found, the last run of the given task type is resumed.
                         If `model` is speficied
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
