# Ultralytics YOLO 🚀, AGPL-3.0 license

import sys
from pathlib import Path
from typing import Union

from ultralytics import yolo  # noqa
from ultralytics.nn.tasks import (ClassificationModel, DetectionModel, PoseModel, SegmentationModel,
                                  attempt_load_one_weight, guess_model_task, nn, yaml_model_load)
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.engine.exporter import Exporter
from ultralytics.yolo.utils import (DEFAULT_CFG, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, RANK, ROOT, callbacks,
                                    is_git_dir, yaml_load)
from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_pip_update_available, check_yaml
from ultralytics.yolo.utils.downloads import GITHUB_ASSET_STEMS
from ultralytics.yolo.utils.torch_utils import smart_inference_mode

# Map head to model, trainer, validator, and predictor classes
TASK_MAP = {
    'classify': [
        ClassificationModel, yolo.v8.classify.ClassificationTrainer, yolo.v8.classify.ClassificationValidator,
        yolo.v8.classify.ClassificationPredictor],
    'detect': [
        DetectionModel, yolo.v8.detect.DetectionTrainer, yolo.v8.detect.DetectionValidator,
        yolo.v8.detect.DetectionPredictor],
    'segment': [
        SegmentationModel, yolo.v8.segment.SegmentationTrainer, yolo.v8.segment.SegmentationValidator,
        yolo.v8.segment.SegmentationPredictor],
    'pose': [PoseModel, yolo.v8.pose.PoseTrainer, yolo.v8.pose.PoseValidator, yolo.v8.pose.PosePredictor]}


class YOLO:
    """
    YOLO (You Only Look Once) object detection model.

    Args:
        model (str, Path): Path to the model file to load or create.
        task (Any, optional): Task type for the YOLO model. Defaults to None.

    Attributes:
        predictor (Any): The predictor object.
        model (Any): The model object.
        trainer (Any): The trainer object.
        task (str): The type of model task.
        ckpt (Any): The checkpoint object if the model loaded from *.pt file.
        cfg (str): The model configuration if loaded from *.yaml file.
        ckpt_path (str): The checkpoint file path.
        overrides (dict): Overrides for the trainer object.
        metrics (Any): The data for metrics.

    Methods:
        __call__(source=None, stream=False, **kwargs):
            Alias for the predict method.
        _new(cfg:str, verbose:bool=True) -> None:
            Initializes a new model and infers the task type from the model definitions.
        _load(weights:str, task:str='') -> None:
            Initializes a new model and infers the task type from the model head.
        _check_is_pytorch_model() -> None:
            Raises TypeError if the model is not a PyTorch model.
        reset() -> None:
            Resets the model modules.
        info(verbose:bool=False) -> None:
            Logs the model info.
        fuse() -> None:
            Fuses the model for faster inference.
        predict(source=None, stream=False, **kwargs) -> List[ultralytics.yolo.engine.results.Results]:
            Performs prediction using the YOLO model.

    Returns:
        list(ultralytics.yolo.engine.results.Results): The prediction results.
    """

    def __init__(self, model: Union[str, Path] = 'yolov8n.pt', task=None) -> None:
        """
        Initializes the YOLO model.

        Args:
            model (Union[str, Path], optional): Path or name of the model to load or create. Defaults to 'yolov8n.pt'.
            task (Any, optional): Task type for the YOLO model. Defaults to None.
        """
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.task = None  # task type
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object
        self.metrics = None  # validation/training metrics
        self.session = None  # HUB session
        model = str(model).strip()  # strip spaces

        # Check if Ultralytics HUB model from https://hub.ultralytics.com
        if self.is_hub_model(model):
            from ultralytics.hub.session import HUBTrainingSession
            self.session = HUBTrainingSession(model)
            model = self.session.model_file

        # Load or create new YOLO model
        suffix = Path(model).suffix
        if not suffix and Path(model).stem in GITHUB_ASSET_STEMS:
            model, suffix = Path(model).with_suffix('.pt'), '.pt'  # add suffix, i.e. yolov8n -> yolov8n.pt
        if suffix == '.yaml':
            self._new(model, task)
        else:
            self._load(model, task)

    def __call__(self, source=None, stream=False, **kwargs):
        """Calls the 'predict' function with given arguments to perform object detection."""
        return self.predict(source, stream, **kwargs)

    def __getattr__(self, attr):
        """Raises error if object has no requested attribute."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    @staticmethod
    def is_hub_model(model):
        """Check if the provided model is a HUB model."""
        return any((
            model.startswith('https://hub.ultra'),  # i.e. https://hub.ultralytics.com/models/MODEL_ID
            [len(x) for x in model.split('_')] == [42, 20],  # APIKEY_MODELID
            len(model) == 20 and not Path(model).exists() and all(x not in model for x in './\\')))  # MODELID

    def _new(self, cfg: str, task=None, verbose=True):
        """
        Initializes a new model and infers the task type from the model definitions.

        Args:
            cfg (str): model configuration file
            task (str) or (None): model task
            verbose (bool): display model info on load
        """
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = TASK_MAP[self.task][0](cfg_dict, verbose=verbose and RANK == -1)  # build model
        self.overrides['model'] = self.cfg

        # Below added to allow export from yamls
        args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine model and default args, preferring model args
        self.model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
        self.model.task = self.task

    def _load(self, weights: str, task=None):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
            task (str) or (None): model task
        """
        suffix = Path(weights).suffix
        if suffix == '.pt':
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args['task']
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = check_file(weights)
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides['model'] = weights
        self.overrides['task'] = self.task

    def _check_is_pytorch_model(self):
        """
        Raises TypeError is model is not a PyTorch model
        """
        pt_str = isinstance(self.model, (str, Path)) and Path(self.model).suffix == '.pt'
        pt_module = isinstance(self.model, nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(f"model='{self.model}' must be a *.pt PyTorch model, but is a different type. "
                            f'PyTorch models can be used to train, val, predict and export, i.e. '
                            f"'yolo export model=yolov8n.pt', but exported formats like ONNX, TensorRT etc. only "
                            f"support 'predict' and 'val' modes, i.e. 'yolo predict model=yolov8n.onnx'.")

    @smart_inference_mode()
    def reset_weights(self):
        """
        Resets the model modules parameters to randomly initialized values, losing all training information.
        """
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    @smart_inference_mode()
    def load(self, weights='yolov8n.pt'):
        """
        Transfers parameters with matching names and shapes from 'weights' to model.
        """
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            weights, self.ckpt = attempt_load_one_weight(weights)
        self.model.load(weights)
        return self

    def info(self, detailed=False, verbose=True):
        """
        Logs model info.

        Args:
            detailed (bool): Show detailed information about model.
            verbose (bool): Controls verbosity.
        """
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self):
        """Fuse PyTorch Conv2d and BatchNorm2d layers."""
        self._check_is_pytorch_model()
        self.model.fuse()

    @smart_inference_mode()
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
        if source is None:
            source = ROOT / 'assets' if is_git_dir() else 'https://ultralytics.com/images/bus.jpg'
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")
        is_cli = (sys.argv[0].endswith('yolo') or sys.argv[0].endswith('ultralytics')) and any(
            x in sys.argv for x in ('predict', 'track', 'mode=predict', 'mode=track'))
        overrides = self.overrides.copy()
        overrides['conf'] = 0.25
        overrides.update(kwargs)  # prefer kwargs
        overrides['mode'] = kwargs.get('mode', 'predict')
        assert overrides['mode'] in ['track', 'predict']
        if not is_cli:
            overrides['save'] = kwargs.get('save', False)  # do not save by default if called in Python
        if not self.predictor:
            self.task = overrides.get('task') or self.task
            self.predictor = TASK_MAP[self.task][3](overrides=overrides, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, overrides)
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    def track(self, source=None, stream=False, persist=False, **kwargs):
        """
        Perform object tracking on the input source using the registered trackers.

        Args:
            source (str, optional): The input source for object tracking. Can be a file path or a video stream.
            stream (bool, optional): Whether the input source is a video stream. Defaults to False.
            persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
            **kwargs (optional): Additional keyword arguments for the tracking process.

        Returns:
            (List[ultralytics.yolo.engine.results.Results]): The tracking results.

        """
        if not hasattr(self.predictor, 'trackers'):
            from ultralytics.tracker import register_tracker
            register_tracker(self, persist)
        # ByteTrack-based method needs low confidence predictions as input
        conf = kwargs.get('conf') or 0.1
        kwargs['conf'] = conf
        kwargs['mode'] = 'track'
        return self.predict(source=source, stream=stream, **kwargs)

    @smart_inference_mode()
    def val(self, data=None, **kwargs):
        """
        Validate a model on a given dataset.

        Args:
            data (str): The dataset to validate on. Accepts all formats accepted by yolo
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        overrides = self.overrides.copy()
        overrides['rect'] = True  # rect batches as default
        overrides.update(kwargs)
        overrides['mode'] = 'val'
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.data = data or args.data
        if 'task' in overrides:
            self.task = args.task
        else:
            args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz and not isinstance(self.model, (str, Path)):
            args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        args.imgsz = check_imgsz(args.imgsz, max_dim=1)

        validator = TASK_MAP[self.task][2](args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics

        return validator.metrics

    @smart_inference_mode()
    def benchmark(self, **kwargs):
        """
        Benchmark a model on all export formats.

        Args:
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        self._check_is_pytorch_model()
        from ultralytics.yolo.utils.benchmarks import benchmark
        overrides = self.model.args.copy()
        overrides.update(kwargs)
        overrides['mode'] = 'benchmark'
        overrides = {**DEFAULT_CFG_DICT, **overrides}  # fill in missing overrides keys with defaults
        return benchmark(model=self, imgsz=overrides['imgsz'], half=overrides['half'], device=overrides['device'])

    def export(self, **kwargs):
        """
        Export model.

        Args:
            **kwargs : Any other args accepted by the predictors. To see all args check 'configuration' section in docs
        """
        self._check_is_pytorch_model()
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        overrides['mode'] = 'export'
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz:
            args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        if args.batch == DEFAULT_CFG.batch:
            args.batch = 1  # default to 1 if not modified
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def train(self, **kwargs):
        """
        Trains the model on a given dataset.

        Args:
            **kwargs (Any): Any number of arguments representing the training configuration.
        """
        self._check_is_pytorch_model()
        if self.session:  # Ultralytics HUB session
            if any(kwargs):
                LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
            kwargs = self.session.train_args
        check_pip_update_available()
        overrides = self.overrides.copy()
        overrides.update(kwargs)
        if kwargs.get('cfg'):
            LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
            overrides = yaml_load(check_yaml(kwargs['cfg']))
        overrides['mode'] = 'train'
        if not overrides.get('data'):
            raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
        if overrides.get('resume'):
            overrides['resume'] = self.ckpt_path
        self.task = overrides.get('task') or self.task
        self.trainer = TASK_MAP[self.task][1](overrides=overrides, _callbacks=self.callbacks)
        if not overrides.get('resume'):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.train()
        # Update model and cfg after training
        if RANK in (-1, 0):
            self.model, _ = attempt_load_one_weight(str(self.trainer.best))
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, 'metrics', None)  # TODO: no metrics returned by DDP

    def to(self, device):
        """
        Sends the model to the given device.

        Args:
            device (str): device
        """
        self._check_is_pytorch_model()
        self.model.to(device)

    def tune(self,
             data: str,
             space: dict = None,
             grace_period: int = 10,
             gpu_per_trial: int = None,
             max_samples: int = 10,
             train_args: dict = {}):
        """
        Runs hyperparameter tuning using Ray Tune.

        Args:
            data (str): The dataset to run the tuner on.
            space (dict, optional): The hyperparameter search space. Defaults to None.
            grace_period (int, optional): The grace period in epochs of the ASHA scheduler. Defaults to 10.
            gpu_per_trial (int, optional): The number of GPUs to allocate per trial. Defaults to None.
            max_samples (int, optional): The maximum number of trials to run. Defaults to 10.
            train_args (dict, optional): Additional arguments to pass to the `train()` method. Defaults to {}.

        Returns:
            (dict): A dictionary containing the results of the hyperparameter search.

        Raises:
            ModuleNotFoundError: If Ray Tune is not installed.
        """

        try:
            from ultralytics.yolo.utils.tuner import (ASHAScheduler, RunConfig, WandbLoggerCallback, default_space,
                                                      task_metric_map, tune)
        except ImportError:
            raise ModuleNotFoundError("Install Ray Tune: `pip install 'ray[tune]'`")

        try:
            import wandb
            from wandb import __version__  # noqa
        except ImportError:
            wandb = False

        def _tune(config):
            """
            Trains the YOLO model with the specified hyperparameters and additional arguments.

            Args:
                config (dict): A dictionary of hyperparameters to use for training.

            Returns:
                None.
            """
            self._reset_callbacks()
            config.update(train_args)
            self.train(**config)

        if not space:
            LOGGER.warning('WARNING: search space not provided. Using default search space')
            space = default_space

        space['data'] = data

        # Define the trainable function with allocated resources
        trainable_with_resources = tune.with_resources(_tune, {'cpu': 8, 'gpu': gpu_per_trial if gpu_per_trial else 0})

        # Define the ASHA scheduler for hyperparameter search
        asha_scheduler = ASHAScheduler(time_attr='epoch',
                                       metric=task_metric_map[self.task],
                                       mode='max',
                                       max_t=train_args.get('epochs') or 100,
                                       grace_period=grace_period,
                                       reduction_factor=3)

        # Define the callbacks for the hyperparameter search
        tuner_callbacks = [WandbLoggerCallback(project='yolov8_tune')] if wandb else []

        # Create the Ray Tune hyperparameter search tuner
        tuner = tune.Tuner(trainable_with_resources,
                           param_space=space,
                           tune_config=tune.TuneConfig(scheduler=asha_scheduler, num_samples=max_samples),
                           run_config=RunConfig(callbacks=tuner_callbacks, local_dir='./runs'))

        # Run the hyperparameter search
        tuner.fit()

        # Return the results of the hyperparameter search
        return tuner.get_results()

    @property
    def names(self):
        """Returns class names of the loaded model."""
        return self.model.names if hasattr(self.model, 'names') else None

    @property
    def device(self):
        """Returns device if PyTorch model."""
        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

    @property
    def transforms(self):
        """Returns transform of the loaded model."""
        return self.model.transforms if hasattr(self.model, 'transforms') else None

    def add_callback(self, event: str, func):
        """Add a callback."""
        self.callbacks[event].append(func)

    def clear_callback(self, event: str):
        """Clear all event callbacks."""
        self.callbacks[event] = []

    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {'imgsz', 'data', 'task', 'single_cls'}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    def _reset_callbacks(self):
        """Reset all registered callbacks."""
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]
