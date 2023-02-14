# Ultralytics YOLO 🚀, GPL-3.0 license

import sys
from pathlib import Path
from typing import List

from ultralytics import yolo  # noqa
from ultralytics.nn.tasks import (ClassificationModel, DetectionModel, SegmentationModel, attempt_load_one_weight,
                                  guess_model_task, nn)
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.engine.exporter import Exporter
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, RANK, callbacks, yaml_load
from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_yaml
from ultralytics.yolo.utils.downloads import GITHUB_ASSET_STEMS
from ultralytics.yolo.utils.torch_utils import smart_inference_mode

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

    def __init__(self, model='yolov8n.pt', type="v8") -> None:
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
        self.metrics_data = None

        # Load or create new YOLO model
        suffix = Path(model).suffix
        if not suffix and Path(model).stem in GITHUB_ASSET_STEMS:
            model, suffix = Path(model).with_suffix('.pt'), '.pt'  # add suffix, i.e. yolov8n -> yolov8n.pt
        if suffix == '.yaml':
            self._new(model)
        else:
            self._load(model)

    def __call__(self, source=None, stream=False, **kwargs):
        return self.predict(source, stream, **kwargs)

    def _new(self, cfg: str, verbose=True):
        """
        Initializes a new model and infers the task type from the model definitions.

        Args:
            cfg (str): model configuration file
            verbose (bool): display model info on load
        """
        self.cfg = check_yaml(cfg)  # check YAML
        cfg_dict = yaml_load(self.cfg, append_filename=True)  # model dict
        self.task = guess_model_task(cfg_dict)
        self.ModelClass, self.TrainerClass, self.ValidatorClass, self.PredictorClass = self._assign_ops_from_task()
        self.model = self.ModelClass(cfg_dict, verbose=verbose)  # initialize

    def _load(self, weights: str):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
        """
        suffix = Path(weights).suffix
        if suffix == '.pt':
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args["task"]
            self.overrides = self.model.args
            self._reset_ckpt_args(self.overrides)
        else:
            check_file(weights)
            self.model, self.ckpt = weights, None
            self.task = guess_model_task(weights)
        self.ckpt_path = weights
        self.ModelClass, self.TrainerClass, self.ValidatorClass, self.PredictorClass = self._assign_ops_from_task()

    def _check_is_pytorch_model(self):
        """
        Raises TypeError is model is not a PyTorch model
        """
        if not isinstance(self.model, nn.Module):
            raise TypeError(f"model='{self.model}' must be a *.pt PyTorch model, but is a different type. "
                            f"PyTorch models can be used to train, val, predict and export, i.e. "
                            f"'yolo export model=yolov8n.pt', but exported formats like ONNX, TensorRT etc. only "
                            f"support 'predict' and 'val' modes, i.e. 'yolo predict model=yolov8n.onnx'.")

    def reset(self):
        """
        Resets the model modules.
        """
        self._check_is_pytorch_model()
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
        self._check_is_pytorch_model()
        self.model.info(verbose=verbose)

    def fuse(self):
        self._check_is_pytorch_model()
        self.model.fuse()

    def predict(self, source=None, stream=False, **kwargs):
        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                          Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            **kwargs : Additional keyword arguments passed to the predictor.
                       Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[ultralytics.yolo.engine.results.Results]): The prediction results.
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
            self.predictor.args = get_cfg(self.predictor.args, overrides)
        is_cli = sys.argv[0].endswith('yolo') or sys.argv[0].endswith('ultralytics')
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    @smart_inference_mode()
    def val(self, data=None, **kwargs):
        """
        Validate a model on a given dataset .

        Args:
            data (str): The dataset to validate on. Accepts all formats accepted by yolo
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        overrides = self.overrides.copy()
        overrides["rect"] = True  # rect batches as default
        overrides.update(kwargs)
        overrides["mode"] = "val"
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.data = data or args.data
        args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz and not isinstance(self.model, (str, Path)):
            args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        args.imgsz = check_imgsz(args.imgsz, max_dim=1)

        validator = self.ValidatorClass(args=args)
        validator(model=self.model)
        self.metrics_data = validator.metrics

        return validator.metrics

    @smart_inference_mode()
    def export(self, **kwargs):
        """
        Export model.

        Args:
            **kwargs : Any other args accepted by the predictors. To see all args check 'configuration' section in docs
        """
        self._check_is_pytorch_model()
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz:
            args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        if args.batch == DEFAULT_CFG.batch:
            args.batch = 1  # default to 1 if not modified
        exporter = Exporter(overrides=args)
        return exporter(model=self.model)

    def train(self, **kwargs):
        """
        Trains the model on a given dataset.

        Args:
            **kwargs (Any): Any number of arguments representing the training configuration.
        """
        self._check_is_pytorch_model()
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        if kwargs.get("cfg"):
            LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
            overrides = yaml_load(check_yaml(kwargs["cfg"]), append_filename=True)
        overrides["task"] = self.task
        overrides["mode"] = "train"
        if not overrides.get("data"):
            raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
        if overrides.get("resume"):
            overrides["resume"] = self.ckpt_path

        self.trainer = self.TrainerClass(overrides=overrides)
        if not overrides.get("resume"):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
        self.trainer.train()
        # update model and cfg after training
        if RANK in {0, -1}:
            self.model, _ = attempt_load_one_weight(str(self.trainer.best))
            self.overrides = self.model.args
            self.metrics_data = self.trainer.validator.metrics

    def to(self, device):
        """
        Sends the model to the given device.

        Args:
            device (str): device
        """
        self._check_is_pytorch_model()
        self.model.to(device)

    def _assign_ops_from_task(self):
        model_class, train_lit, val_lit, pred_lit = MODEL_MAP[self.task]
        trainer_class = eval(train_lit.replace("TYPE", f"{self.type}"))
        validator_class = eval(val_lit.replace("TYPE", f"{self.type}"))
        predictor_class = eval(pred_lit.replace("TYPE", f"{self.type}"))
        return model_class, trainer_class, validator_class, predictor_class

    @property
    def names(self):
        """
         Returns class names of the loaded model.
        """
        return self.model.names if hasattr(self.model, 'names') else None

    @property
    def transforms(self):
        """
         Returns transform of the loaded model.
        """
        return self.model.transforms if hasattr(self.model, 'transforms') else None

    @property
    def metrics(self):
        """
        Returns metrics if computed
        """
        if not self.metrics_data:
            LOGGER.info("No metrics data found! Run training or validation operation first.")

        return self.metrics_data

    @staticmethod
    def add_callback(event: str, func):
        """
        Add callback
        """
        callbacks.default_callbacks[event].append(func)

    @staticmethod
    def _reset_ckpt_args(args):
        for arg in 'augment', 'verbose', 'project', 'name', 'exist_ok', 'resume', 'batch', 'epochs', 'cache', \
                'save_json', 'half', 'v5loader', 'device', 'cfg', 'save', 'rect', 'plots', 'opset':
            args.pop(arg, None)
