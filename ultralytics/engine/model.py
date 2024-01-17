# Ultralytics YOLO 🚀, AGPL-3.0 license

import inspect
import sys
from pathlib import Path
from typing import Union

from ultralytics.cfg import TASK2DATA, get_cfg, get_save_dir
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load
from ultralytics.utils import ASSETS, DEFAULT_CFG_DICT, LOGGER, RANK, SETTINGS, callbacks, checks, emojis, yaml_load
from ultralytics.hub.utils import HUB_WEB_ROOT


class Model(nn.Module):
    """
    A base class to unify APIs for all models.

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
        predict(source=None, stream=False, **kwargs) -> List[ultralytics.engine.results.Results]:
            Performs prediction using the YOLO model.

    Returns:
        list(ultralytics.engine.results.Results): The prediction results.
    """

    def __init__(self, model: Union[str, Path] = "yolov8n.pt", task=None) -> None:
        """
        Initializes the YOLO model.

        Args:
            model (Union[str, Path], optional): Path or name of the model to load or create. Defaults to 'yolov8n.pt'.
            task (Any, optional): Task type for the YOLO model. Defaults to None.
        """
        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object
        self.metrics = None  # validation/training metrics
        self.session = None  # HUB session
        self.task = task  # task type
        self.model_name = model = str(model).strip()  # strip spaces

        # Check if Ultralytics HUB model from https://hub.ultralytics.com
        if self.is_hub_model(model):
            # Fetch model from HUB
            self.session = self._get_hub_session(model)
            model = self.session.model_file

        # Check if Triton Server model
        elif self.is_triton_model(model):
            self.model = model
            self.task = task
            return

        # Load or create new YOLO model
        model = checks.check_model_file_from_stem(model)  # add suffix, i.e. yolov8n -> yolov8n.pt
        if Path(model).suffix in (".yaml", ".yml"):
            self._new(model, task)
        else:
            self._load(model, task)

        self.model_name = model

    def __call__(self, source=None, stream=False, **kwargs):
        """Calls the predict() method with given arguments to perform object detection."""
        return self.predict(source, stream, **kwargs)

    @staticmethod
    def _get_hub_session(model: str):
        """Creates a session for Hub Training."""
        from ultralytics.hub.session import HUBTrainingSession

        session = HUBTrainingSession(model)
        return session if session.client.authenticated else None

    @staticmethod
    def is_triton_model(model):
        """Is model a Triton Server URL string, i.e. <scheme>://<netloc>/<endpoint>/<task_name>"""
        from urllib.parse import urlsplit

        url = urlsplit(model)
        return url.netloc and url.path and url.scheme in {"http", "grpc"}

    @staticmethod
    def is_hub_model(model):
        """Check if the provided model is a HUB model."""
        return any(
            (
                model.startswith(f"{HUB_WEB_ROOT}/models/"),  # i.e. https://hub.ultralytics.com/models/MODEL_ID
                [len(x) for x in model.split("_")] == [42, 20],  # APIKEY_MODELID
                len(model) == 20 and not Path(model).exists() and all(x not in model for x in "./\\"),  # MODELID
            )
        )

    def _new(self, cfg: str, task=None, model=None, verbose=True):
        """
        Initializes a new model and infers the task type from the model definitions.

        Args:
            cfg (str): model configuration file
            task (str | None): model task
            model (BaseModel): Customized model.
            verbose (bool): display model info on load
        """
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = (model or self._smart_load("model"))(cfg_dict, verbose=verbose and RANK == -1)  # build model
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task

        # Below added to allow export from YAMLs
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine default and model args (prefer model args)
        self.model.task = self.task

    def _load(self, weights: str, task=None):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
            task (str | None): model task
        """
        suffix = Path(weights).suffix
        if suffix == ".pt":
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args["task"]
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = checks.check_file(weights)
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides["model"] = weights
        self.overrides["task"] = self.task

    def _check_is_pytorch_model(self):
        """Raises TypeError is model is not a PyTorch model."""
        pt_str = isinstance(self.model, (str, Path)) and Path(self.model).suffix == ".pt"
        pt_module = isinstance(self.model, nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. "
                f"PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported "
                f"formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, "
                f"i.e. 'yolo predict model=yolov8n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'"
            )

    def reset_weights(self):
        """Resets the model modules parameters to randomly initialized values, losing all training information."""
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    def load(self, weights="yolov8n.pt"):
        """Transfers parameters with matching names and shapes from 'weights' to model."""
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

    def embed(self, source=None, stream=False, **kwargs):
        """
        Calls the predict() method and returns image embeddings.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            **kwargs : Additional keyword arguments passed to the predictor.
                Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[torch.Tensor]): A list of image embeddings.
        """
        if not kwargs.get("embed"):
            kwargs["embed"] = [len(self.model.model) - 2]  # embed second-to-last layer if no indices passed
        return self.predict(source, stream, **kwargs)

    def predict(self, source=None, stream=False, predictor=None, **kwargs):
        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            predictor (BasePredictor): Customized predictor.
            **kwargs : Additional keyword arguments passed to the predictor.
                Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[ultralytics.engine.results.Results]): The prediction results.
        """
        if source is None:
            source = ASSETS
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")

        is_cli = (sys.argv[0].endswith("yolo") or sys.argv[0].endswith("ultralytics")) and any(
            x in sys.argv for x in ("predict", "track", "mode=predict", "mode=track")
        )

        custom = {"conf": 0.25, "save": is_cli}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "predict"}  # highest priority args on the right
        prompts = args.pop("prompts", None)  # for SAM-type models

        if not self.predictor:
            self.predictor = predictor or self._smart_load("predictor")(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, "set_prompts"):  # for SAM-type models
            self.predictor.set_prompts(prompts)
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
            (List[ultralytics.engine.results.Results]): The tracking results.
        """
        if not hasattr(self.predictor, "trackers"):
            from ultralytics.trackers import register_tracker

            register_tracker(self, persist)
        kwargs["conf"] = kwargs.get("conf") or 0.1  # ByteTrack-based method needs low confidence predictions as input
        kwargs["mode"] = "track"
        return self.predict(source=source, stream=stream, **kwargs)

    def val(self, validator=None, **kwargs):
        """
        Validate a model on a given dataset.

        Args:
            validator (BaseValidator): Customized validator.
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        custom = {"rect": True}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def benchmark(self, **kwargs):
        """
        Benchmark a model on all export formats.

        Args:
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        self._check_is_pytorch_model()
        from ultralytics.utils.benchmarks import benchmark

        custom = {"verbose": False}  # method defaults
        args = {**DEFAULT_CFG_DICT, **self.model.args, **custom, **kwargs, "mode": "benchmark"}
        return benchmark(
            model=self,
            data=kwargs.get("data"),  # if no 'data' argument passed set data=None for default datasets
            imgsz=args["imgsz"],
            half=args["half"],
            int8=args["int8"],
            device=args["device"],
            verbose=kwargs.get("verbose"),
        )

    def export(self, **kwargs):
        """
        Export model.

        Args:
            **kwargs : Any other args accepted by the Exporter. To see all args check 'configuration' section in docs.
        """
        self._check_is_pytorch_model()
        from .exporter import Exporter

        custom = {"imgsz": self.model.args["imgsz"], "batch": 1, "data": None, "verbose": False}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}  # highest priority args on the right
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def train(self, trainer=None, **kwargs):
        """
        Trains the model on a given dataset.

        Args:
            trainer (BaseTrainer, optional): Customized trainer.
            **kwargs (Any): Any number of arguments representing the training configuration.
        """
        self._check_is_pytorch_model()
        if hasattr(self.session, "model") and self.session.model.id:  # Ultralytics HUB session with loaded model
            if any(kwargs):
                LOGGER.warning("WARNING ⚠️ using HUB training arguments, ignoring local training arguments.")
            kwargs = self.session.train_args  # overwrite kwargs

        checks.check_pip_update_available()

        overrides = yaml_load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
        custom = {"data": DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task]}  # method defaults
        args = {**overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right
        if args.get("resume"):
            args["resume"] = self.ckpt_path

        self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)
        if not args.get("resume"):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model

            if SETTINGS["hub"] is True and not self.session:
                # Create a model in HUB
                try:
                    self.session = self._get_hub_session(self.model_name)
                    if self.session:
                        self.session.create_model(args)
                        # Check model was created
                        if not getattr(self.session.model, "id", None):
                            self.session = None
                except PermissionError:
                    # Ignore permission error
                    pass

        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.train()
        # Update model and cfg after training
        if RANK in (-1, 0):
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, _ = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, "metrics", None)  # TODO: no metrics returned by DDP
        return self.metrics

    def tune(self, use_ray=False, iterations=10, *args, **kwargs):
        """
        Runs hyperparameter tuning, optionally using Ray Tune. See ultralytics.utils.tuner.run_ray_tune for Args.

        Returns:
            (dict): A dictionary containing the results of the hyperparameter search.
        """
        self._check_is_pytorch_model()
        if use_ray:
            from ultralytics.utils.tuner import run_ray_tune

            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)
        else:
            from .tuner import Tuner

            custom = {}  # method defaults
            args = {**self.overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right
            return Tuner(args=args, _callbacks=self.callbacks)(model=self, iterations=iterations)

    def _apply(self, fn):
        """Apply to(), cpu(), cuda(), half(), float() to model tensors that are not parameters or registered buffers."""
        self._check_is_pytorch_model()
        self = super()._apply(fn)  # noqa
        self.predictor = None  # reset predictor as device may have changed
        self.overrides["device"] = self.device  # was str(self.device) i.e. device(type='cuda', index=0) -> 'cuda:0'
        return self

    @property
    def names(self):
        """Returns class names of the loaded model."""
        return self.model.names if hasattr(self.model, "names") else None

    @property
    def device(self):
        """Returns device if PyTorch model."""
        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

    @property
    def transforms(self):
        """Returns transform of the loaded model."""
        return self.model.transforms if hasattr(self.model, "transforms") else None

    def add_callback(self, event: str, func):
        """Add a callback."""
        self.callbacks[event].append(func)

    def clear_callback(self, event: str):
        """Clear all event callbacks."""
        self.callbacks[event] = []

    def reset_callbacks(self):
        """Reset all registered callbacks."""
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]

    @staticmethod
    def _reset_ckpt_args(args):
        """Reset arguments when loading a PyTorch model."""
        include = {"imgsz", "data", "task", "single_cls"}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    # def __getattr__(self, attr):
    #    """Raises error if object has no requested attribute."""
    #    name = self.__class__.__name__
    #    raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    def _smart_load(self, key):
        """Load model/trainer/validator/predictor."""
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # get the function name.
            raise NotImplementedError(
                emojis(f"WARNING ⚠️ '{name}' model does not support '{mode}' mode for '{self.task}' task yet.")
            ) from e

    @property
    def task_map(self):
        """
        Map head to model, trainer, validator, and predictor classes.

        Returns:
            task_map (dict): The map of model task to mode classes.
        """
        raise NotImplementedError("Please provide task map for your model!")
