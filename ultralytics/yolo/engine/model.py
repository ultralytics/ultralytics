# Ultralytics YOLO 🚀, GPL-3.0 license

from pathlib import Path

from ultralytics import yolo  # noqa
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, SegmentationModel, attempt_load_one_weight
from ultralytics.yolo.configs import get_config
from ultralytics.yolo.engine.exporter import Exporter
from ultralytics.yolo.utils import DEFAULT_CFG_PATH, LOGGER, yaml_load
from ultralytics.yolo.utils.checks import check_yaml
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
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.task = None  # task type
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object

        # Load or create new YOLO model
        {'.pt': self._load, '.yaml': self._new}[Path(model).suffix](model)

    def __call__(self, source=None, stream=False, verbose=False, **kwargs):
        return self.predict(source, stream, verbose, **kwargs)

    def _new(self, cfg: str, verbose=True):
        """
        Initializes a new model and infers the task type from the model definitions.

        Args:
            cfg (str): model configuration file
            verbose (bool): display model info on load
        """
        cfg = check_yaml(cfg)  # check YAML
        cfg_dict = yaml_load(cfg, append_filename=True)  # model dict
        self.task = guess_task_from_head(cfg_dict["head"][-1][-2])
        self.ModelClass, self.TrainerClass, self.ValidatorClass, self.PredictorClass = \
            self._guess_ops_from_task(self.task)
        self.model = self.ModelClass(cfg_dict, verbose=verbose)  # initialize
        self.cfg = cfg

    def _load(self, weights: str):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
        """
        self.model, self.ckpt = attempt_load_one_weight(weights)
        self.ckpt_path = weights
        self.task = self.model.args["task"]
        self.overrides = self.model.args
        self._reset_ckpt_args(self.overrides)
        self.ModelClass, self.TrainerClass, self.ValidatorClass, self.PredictorClass = \
            self._guess_ops_from_task(self.task)

    def reset(self):
        """
        Resets the model modules.
        """
        for m in self.model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True

    def info(self, verbose=False):
        """
        Logs model info.

        Args:
            verbose (bool): Controls verbosity.
        """
        self.model.info(verbose=verbose)

    def fuse(self):
        self.model.fuse()

    @smart_inference_mode()
    def predict(self, source=None, stream=False, verbose=False, **kwargs):
        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                          Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            verbose (bool): Whether to print verbose information or not. Defaults to False.
            **kwargs : Additional keyword arguments passed to the predictor.
                       Check the 'configuration' section in the documentation for all available options.

        Returns:
            (dict): The prediction results.
        """
        overrides = self.overrides.copy()
        overrides["conf"] = 0.25
        overrides.update(kwargs)
        overrides["mode"] = "predict"
        overrides["save"] = kwargs.get("save", False)  # not save files by default
        if not self.predictor:
            self.predictor = self.PredictorClass(overrides=overrides)
            self.predictor.setup_model(model=self.model)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_config(self.predictor.args, overrides)
        return self.predictor(source=source, stream=stream, verbose=verbose)

    @smart_inference_mode()
    def val(self, data=None, **kwargs):
        """
        Validate a model on a given dataset .

        Args:
            data (str): The dataset to validate on. Accepts all formats accepted by yolo
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        overrides["mode"] = "val"
        args = get_config(config=DEFAULT_CFG_PATH, overrides=overrides)
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
        args = get_config(config=DEFAULT_CFG_PATH, overrides=overrides)
        args.task = self.task

        print(args)
        exporter = Exporter(overrides=args)
        exporter(model=self.model)

    def train(self, **kwargs):
        """
        Trains the model on a given dataset.

        Args:
            **kwargs (Any): Any number of arguments representing the training configuration. List of all args can be found in 'config' section.
                            You can pass all arguments as a yaml file in `cfg`. Other args are ignored if `cfg` file is passed
        """
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        if kwargs.get("cfg"):
            LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
            overrides = yaml_load(check_yaml(kwargs["cfg"]), append_filename=True)
        overrides["task"] = self.task
        overrides["mode"] = "train"
        if not overrides.get("data"):
            raise AttributeError("dataset not provided! Please define `data` in config.yaml or pass as an argument.")
        if overrides.get("resume"):
            overrides["resume"] = self.ckpt_path

        self.trainer = self.TrainerClass(overrides=overrides)
        if not overrides.get("resume"):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
        self.trainer.train()
        # update model and configs after training
        self.model, _ = attempt_load_one_weight(str(self.trainer.best))
        self.overrides = self.model.args

    def to(self, device):
        """
        Sends the model to the given device.

        Args:
            device (str): device
        """
        self.model.to(device)

    def _guess_ops_from_task(self, task):
        model_class, train_lit, val_lit, pred_lit = MODEL_MAP[task]
        # warning: eval is unsafe. Use with caution
        trainer_class = eval(train_lit.replace("TYPE", f"{self.type}"))
        validator_class = eval(val_lit.replace("TYPE", f"{self.type}"))
        predictor_class = eval(pred_lit.replace("TYPE", f"{self.type}"))

        return model_class, trainer_class, validator_class, predictor_class

    @staticmethod
    def _reset_ckpt_args(args):
        args.pop("project", None)
        args.pop("name", None)
        args.pop("exist_ok", None)
        args.pop("resume", None)
        args.pop("batch", None)
        args.pop("epochs", None)
        args.pop("cache", None)
        args.pop("save_json", None)
        args.pop("half", None)
        args.pop("v5loader", None)

        # set device to '' to prevent from auto DDP usage
        args["device"] = ''
