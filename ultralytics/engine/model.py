# Ultralytics YOLO 🚀, AGPL-3.0 license

import inspect
from pathlib import Path
from typing import Union

import numpy as np
import torch

from ultralytics.cfg import TASK2DATA, get_cfg, get_save_dir
from ultralytics.hub.utils import HUB_WEB_ROOT
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load
from ultralytics.utils import (
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    SETTINGS,
    callbacks,
    checks,
    emojis,
    yaml_load,
)


class Model(nn.Module):
    """
    A base class for implementing YOLO models, unifying APIs across different model types.

    This class provides a common interface for various operations related to YOLO models, such as training,
    validation, prediction, exporting, and benchmarking. It handles different types of models, including those
    loaded from local files, Ultralytics HUB, or Triton Server. The class is designed to be flexible and
    extendable for different tasks and model configurations.

    Args:
        model (Union[str, Path], optional): Path or name of the model to load or create. This can be a local file
            path, a model name from Ultralytics HUB, or a Triton Server model. Defaults to 'yolov8n.pt'.
        task (Any, optional): The task type associated with the YOLO model. This can be used to specify the model's
            application domain, such as object detection, segmentation, etc. Defaults to None.
        verbose (bool, optional): If True, enables verbose output during the model's operations. Defaults to False.

    Attributes:
        callbacks (dict): A dictionary of callback functions for various events during model operations.
        predictor (BasePredictor): The predictor object used for making predictions.
        model (nn.Module): The underlying PyTorch model.
        trainer (BaseTrainer): The trainer object used for training the model.
        ckpt (dict): The checkpoint data if the model is loaded from a *.pt file.
        cfg (str): The configuration of the model if loaded from a *.yaml file.
        ckpt_path (str): The path to the checkpoint file.
        overrides (dict): A dictionary of overrides for model configuration.
        metrics (dict): The latest training/validation metrics.
        session (HUBTrainingSession): The Ultralytics HUB session, if applicable.
        task (str): The type of task the model is intended for.
        model_name (str): The name of the model.

    Methods:
        __call__: Alias for the predict method, enabling the model instance to be callable.
        _new: Initializes a new model based on a configuration file.
        _load: Loads a model from a checkpoint file.
        _check_is_pytorch_model: Ensures that the model is a PyTorch model.
        reset_weights: Resets the model's weights to their initial state.
        load: Loads model weights from a specified file.
        save: Saves the current state of the model to a file.
        info: Logs or returns information about the model.
        fuse: Fuses Conv2d and BatchNorm2d layers for optimized inference.
        predict: Performs object detection predictions.
        track: Performs object tracking.
        val: Validates the model on a dataset.
        benchmark: Benchmarks the model on various export formats.
        export: Exports the model to different formats.
        train: Trains the model on a dataset.
        tune: Performs hyperparameter tuning.
        _apply: Applies a function to the model's tensors.
        add_callback: Adds a callback function for an event.
        clear_callback: Clears all callbacks for an event.
        reset_callbacks: Resets all callbacks to their default functions.
        _get_hub_session: Retrieves or creates an Ultralytics HUB session.
        is_triton_model: Checks if a model is a Triton Server model.
        is_hub_model: Checks if a model is an Ultralytics HUB model.
        _reset_ckpt_args: Resets checkpoint arguments when loading a PyTorch model.
        _smart_load: Loads the appropriate module based on the model task.
        task_map: Provides a mapping from model tasks to corresponding classes.

    Raises:
        FileNotFoundError: If the specified model file does not exist or is inaccessible.
        ValueError: If the model file or configuration is invalid or unsupported.
        ImportError: If required dependencies for specific model types (like HUB SDK) are not installed.
        TypeError: If the model is not a PyTorch model when required.
        AttributeError: If required attributes or methods are not implemented or available.
        NotImplementedError: If a specific model task or mode is not supported.
    """

    def __init__(
        self,
        model: Union[str, Path] = "yolov8n.pt",
        task: str = None,
        verbose: bool = False,
    ) -> None:
        """
        Initializes a new instance of the YOLO model class.

        This constructor sets up the model based on the provided model path or name. It handles various types of model
        sources, including local files, Ultralytics HUB models, and Triton Server models. The method initializes several
        important attributes of the model and prepares it for operations like training, prediction, or export.

        Args:
            model (Union[str, Path], optional): The path or model file to load or create. This can be a local
                file path, a model name from Ultralytics HUB, or a Triton Server model. Defaults to 'yolov8n.pt'.
            task (Any, optional): The task type associated with the YOLO model, specifying its application domain.
                Defaults to None.
            verbose (bool, optional): If True, enables verbose output during the model's initialization and subsequent
                operations. Defaults to False.

        Raises:
            FileNotFoundError: If the specified model file does not exist or is inaccessible.
            ValueError: If the model file or configuration is invalid or unsupported.
            ImportError: If required dependencies for specific model types (like HUB SDK) are not installed.
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
        model = str(model).strip()

        # Check if Ultralytics HUB model from https://hub.ultralytics.com
        if self.is_hub_model(model):
            # Fetch model from HUB
            checks.check_requirements("hub-sdk>=0.0.6")
            self.session = self._get_hub_session(model)
            model = self.session.model_file

        # Check if Triton Server model
        elif self.is_triton_model(model):
            self.model_name = self.model = model
            self.task = task
            return

        # Load or create new YOLO model
        if Path(model).suffix in {".yaml", ".yml"}:
            self._new(model, task=task, verbose=verbose)
        else:
            self._load(model, task=task)

    def __call__(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs,
    ) -> list:
        """
        An alias for the predict method, enabling the model instance to be callable.

        This method simplifies the process of making predictions by allowing the model instance to be called directly
        with the required arguments for prediction.

        Args:
            source (str | Path | int | PIL.Image | np.ndarray, optional): The source of the image for making
                predictions. Accepts various types, including file paths, URLs, PIL images, and numpy arrays.
                Defaults to None.
            stream (bool, optional): If True, treats the input source as a continuous stream for predictions.
                Defaults to False.
            **kwargs (any): Additional keyword arguments for configuring the prediction process.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of prediction results, encapsulated in the Results class.
        """
        return self.predict(source, stream, **kwargs)

    @staticmethod
    def _get_hub_session(model: str):
        """Creates a session for Hub Training."""
        from ultralytics.hub.session import HUBTrainingSession

        session = HUBTrainingSession(model)
        return session if session.client.authenticated else None

    @staticmethod
    def is_triton_model(model: str) -> bool:
        """Is model a Triton Server URL string, i.e. <scheme>://<netloc>/<endpoint>/<task_name>"""
        from urllib.parse import urlsplit

        url = urlsplit(model)
        return url.netloc and url.path and url.scheme in {"http", "grpc"}

    @staticmethod
    def is_hub_model(model: str) -> bool:
        """Check if the provided model is a HUB model."""
        return any(
            (
                model.startswith(f"{HUB_WEB_ROOT}/models/"),  # i.e. https://hub.ultralytics.com/models/MODEL_ID
                [len(x) for x in model.split("_")] == [42, 20],  # APIKEY_MODEL
                len(model) == 20 and not Path(model).exists() and all(x not in model for x in "./\\"),  # MODEL
            )
        )

    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:
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
        self.model_name = cfg

    def _load(self, weights: str, task=None) -> None:
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
            task (str | None): model task
        """
        if weights.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):
            weights = checks.check_file(weights)  # automatically download and return local filename
        weights = checks.check_model_file_from_stem(weights)  # add suffix, i.e. yolov8n -> yolov8n.pt

        if Path(weights).suffix == ".pt":
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.args["task"]
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = checks.check_file(weights)  # runs in all cases, not redundant with above call
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides["model"] = weights
        self.overrides["task"] = self.task
        self.model_name = weights

    def _check_is_pytorch_model(self) -> None:
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

    def reset_weights(self) -> "Model":
        """
        Resets the model parameters to randomly initialized values, effectively discarding all training information.

        This method iterates through all modules in the model and resets their parameters if they have a
        'reset_parameters' method. It also ensures that all parameters have 'requires_grad' set to True, enabling them
        to be updated during training.

        Returns:
            self (ultralytics.engine.model.Model): The instance of the class with reset weights.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    def load(self, weights: Union[str, Path] = "yolov8n.pt") -> "Model":
        """
        Loads parameters from the specified weights file into the model.

        This method supports loading weights from a file or directly from a weights object. It matches parameters by
        name and shape and transfers them to the model.

        Args:
            weights (str | Path): Path to the weights file or a weights object. Defaults to 'yolov8n.pt'.

        Returns:
            self (ultralytics.engine.model.Model): The instance of the class with loaded weights.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            weights, self.ckpt = attempt_load_one_weight(weights)
        self.model.load(weights)
        return self

    def save(self, filename: Union[str, Path] = "saved_model.pt", use_dill=True) -> None:
        """
        Saves the current model state to a file.

        This method exports the model's checkpoint (ckpt) to the specified filename.

        Args:
            filename (str | Path): The name of the file to save the model to. Defaults to 'saved_model.pt'.
            use_dill (bool): Whether to try using dill for serialization if available. Defaults to True.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        from datetime import datetime

        from ultralytics import __version__

        updates = {
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }
        torch.save({**self.ckpt, **updates}, filename, use_dill=use_dill)

    def info(self, detailed: bool = False, verbose: bool = True):
        """
        Logs or returns model information.

        This method provides an overview or detailed information about the model, depending on the arguments passed.
        It can control the verbosity of the output.

        Args:
            detailed (bool): If True, shows detailed information about the model. Defaults to False.
            verbose (bool): If True, prints the information. If False, returns the information. Defaults to True.

        Returns:
            (list): Various types of information about the model, depending on the 'detailed' and 'verbose' parameters.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self):
        """
        Fuses Conv2d and BatchNorm2d layers in the model.

        This method optimizes the model by fusing Conv2d and BatchNorm2d layers, which can improve inference speed.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        self.model.fuse()

    def embed(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs,
    ) -> list:
        """
        Generates image embeddings based on the provided source.

        This method is a wrapper around the 'predict()' method, focusing on generating embeddings from an image source.
        It allows customization of the embedding process through various keyword arguments.

        Args:
            source (str | int | PIL.Image | np.ndarray): The source of the image for generating embeddings.
                The source can be a file path, URL, PIL image, numpy array, etc. Defaults to None.
            stream (bool): If True, predictions are streamed. Defaults to False.
            **kwargs (any): Additional keyword arguments for configuring the embedding process.

        Returns:
            (List[torch.Tensor]): A list containing the image embeddings.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        if not kwargs.get("embed"):
            kwargs["embed"] = [len(self.model.model) - 2]  # embed second-to-last layer if no indices passed
        return self.predict(source, stream, **kwargs)

    def predict(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        predictor=None,
        **kwargs,
    ) -> list:
        """
        Performs predictions on the given image source using the YOLO model.

        This method facilitates the prediction process, allowing various configurations through keyword arguments.
        It supports predictions with custom predictors or the default predictor method. The method handles different
        types of image sources and can operate in a streaming mode. It also provides support for SAM-type models
        through 'prompts'.

        The method sets up a new predictor if not already present and updates its arguments with each call.
        It also issues a warning and uses default assets if the 'source' is not provided. The method determines if it
        is being called from the command line interface and adjusts its behavior accordingly, including setting defaults
        for confidence threshold and saving behavior.

        Args:
            source (str | int | PIL.Image | np.ndarray, optional): The source of the image for making predictions.
                Accepts various types, including file paths, URLs, PIL images, and numpy arrays. Defaults to ASSETS.
            stream (bool, optional): Treats the input source as a continuous stream for predictions. Defaults to False.
            predictor (BasePredictor, optional): An instance of a custom predictor class for making predictions.
                If None, the method uses a default predictor. Defaults to None.
            **kwargs (any): Additional keyword arguments for configuring the prediction process. These arguments allow
                for further customization of the prediction behavior.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of prediction results, encapsulated in the Results class.

        Raises:
            AttributeError: If the predictor is not properly set up.
        """
        if source is None:
            source = ASSETS
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")

        is_cli = (ARGV[0].endswith("yolo") or ARGV[0].endswith("ultralytics")) and any(
            x in ARGV for x in ("predict", "track", "mode=predict", "mode=track")
        )

        custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict"}  # method defaults
        args = {**self.overrides, **custom, **kwargs}  # highest priority args on the right
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

    def track(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        persist: bool = False,
        **kwargs,
    ) -> list:
        """
        Conducts object tracking on the specified input source using the registered trackers.

        This method performs object tracking using the model's predictors and optionally registered trackers. It is
        capable of handling different types of input sources such as file paths or video streams. The method supports
        customization of the tracking process through various keyword arguments. It registers trackers if they are not
        already present and optionally persists them based on the 'persist' flag.

        The method sets a default confidence threshold specifically for ByteTrack-based tracking, which requires low
        confidence predictions as input. The tracking mode is explicitly set in the keyword arguments.

        Args:
            source (str, optional): The input source for object tracking. It can be a file path, URL, or video stream.
            stream (bool, optional): Treats the input source as a continuous video stream. Defaults to False.
            persist (bool, optional): Persists the trackers between different calls to this method. Defaults to False.
            **kwargs (any): Additional keyword arguments for configuring the tracking process. These arguments allow
                for further customization of the tracking behavior.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of tracking results, encapsulated in the Results class.

        Raises:
            AttributeError: If the predictor does not have registered trackers.
        """
        if not hasattr(self.predictor, "trackers"):
            from ultralytics.trackers import register_tracker

            register_tracker(self, persist)
        kwargs["conf"] = kwargs.get("conf") or 0.1  # ByteTrack-based method needs low confidence predictions as input
        kwargs["batch"] = kwargs.get("batch") or 1  # batch-size 1 for tracking in videos
        kwargs["mode"] = "track"
        return self.predict(source=source, stream=stream, **kwargs)

    def val(
        self,
        validator=None,
        **kwargs,
    ):
        """
        Validates the model using a specified dataset and validation configuration.

        This method facilitates the model validation process, allowing for a range of customization through various
        settings and configurations. It supports validation with a custom validator or the default validation approach.
        The method combines default configurations, method-specific defaults, and user-provided arguments to configure
        the validation process. After validation, it updates the model's metrics with the results obtained from the
        validator.

        The method supports various arguments that allow customization of the validation process. For a comprehensive
        list of all configurable options, users should refer to the 'configuration' section in the documentation.

        Args:
            validator (BaseValidator, optional): An instance of a custom validator class for validating the model. If
                None, the method uses a default validator. Defaults to None.
            **kwargs (any): Arbitrary keyword arguments representing the validation configuration. These arguments are
                used to customize various aspects of the validation process.

        Returns:
            (dict): Validation metrics obtained from the validation process.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        custom = {"rect": True}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def benchmark(
        self,
        **kwargs,
    ):
        """
        Benchmarks the model across various export formats to evaluate performance.

        This method assesses the model's performance in different export formats, such as ONNX, TorchScript, etc.
        It uses the 'benchmark' function from the ultralytics.utils.benchmarks module. The benchmarking is configured
        using a combination of default configuration values, model-specific arguments, method-specific defaults, and
        any additional user-provided keyword arguments.

        The method supports various arguments that allow customization of the benchmarking process, such as dataset
        choice, image size, precision modes, device selection, and verbosity. For a comprehensive list of all
        configurable options, users should refer to the 'configuration' section in the documentation.

        Args:
            **kwargs (any): Arbitrary keyword arguments to customize the benchmarking process. These are combined with
                default configurations, model-specific arguments, and method defaults.

        Returns:
            (dict): A dictionary containing the results of the benchmarking process.

        Raises:
            AssertionError: If the model is not a PyTorch model.
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

    def export(
        self,
        **kwargs,
    ):
        """
        Exports the model to a different format suitable for deployment.

        This method facilitates the export of the model to various formats (e.g., ONNX, TorchScript) for deployment
        purposes. It uses the 'Exporter' class for the export process, combining model-specific overrides, method
        defaults, and any additional arguments provided. The combined arguments are used to configure export settings.

        The method supports a wide range of arguments to customize the export process. For a comprehensive list of all
        possible arguments, refer to the 'configuration' section in the documentation.

        Args:
            **kwargs (any): Arbitrary keyword arguments to customize the export process. These are combined with the
                model's overrides and method defaults.

        Returns:
            (object): The exported model in the specified format, or an object related to the export process.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        from .exporter import Exporter

        custom = {"imgsz": self.model.args["imgsz"], "batch": 1, "data": None, "verbose": False}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}  # highest priority args on the right
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def train(
        self,
        trainer=None,
        **kwargs,
    ):
        """
        Trains the model using the specified dataset and training configuration.

        This method facilitates model training with a range of customizable settings and configurations. It supports
        training with a custom trainer or the default training approach defined in the method. The method handles
        different scenarios, such as resuming training from a checkpoint, integrating with Ultralytics HUB, and
        updating model and configuration after training.

        When using Ultralytics HUB, if the session already has a loaded model, the method prioritizes HUB training
        arguments and issues a warning if local arguments are provided. It checks for pip updates and combines default
        configurations, method-specific defaults, and user-provided arguments to configure the training process. After
        training, it updates the model and its configurations, and optionally attaches metrics.

        Args:
            trainer (BaseTrainer, optional): An instance of a custom trainer class for training the model. If None, the
                method uses a default trainer. Defaults to None.
            **kwargs (any): Arbitrary keyword arguments representing the training configuration. These arguments are
                used to customize various aspects of the training process.

        Returns:
            (dict | None): Training metrics if available and training is successful; otherwise, None.

        Raises:
            AssertionError: If the model is not a PyTorch model.
            PermissionError: If there is a permission issue with the HUB session.
            ModuleNotFoundError: If the HUB SDK is not installed.
        """
        self._check_is_pytorch_model()
        if hasattr(self.session, "model") and self.session.model.id:  # Ultralytics HUB session with loaded model
            if any(kwargs):
                LOGGER.warning("WARNING ⚠️ using HUB training arguments, ignoring local training arguments.")
            kwargs = self.session.train_args  # overwrite kwargs

        checks.check_pip_update_available()

        overrides = yaml_load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
        custom = {
            # NOTE: handle the case when 'cfg' includes 'data'.
            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task],
            "model": self.overrides["model"],
            "task": self.task,
        }  # method defaults
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
                except (PermissionError, ModuleNotFoundError):
                    # Ignore PermissionError and ModuleNotFoundError which indicates hub-sdk not installed
                    pass

        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.train()
        # Update model and cfg after training
        if RANK in {-1, 0}:
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, _ = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, "metrics", None)  # TODO: no metrics returned by DDP
        return self.metrics

    def tune(
        self,
        use_ray=False,
        iterations=10,
        *args,
        **kwargs,
    ):
        """
        Conducts hyperparameter tuning for the model, with an option to use Ray Tune.

        This method supports two modes of hyperparameter tuning: using Ray Tune or a custom tuning method.
        When Ray Tune is enabled, it leverages the 'run_ray_tune' function from the ultralytics.utils.tuner module.
        Otherwise, it uses the internal 'Tuner' class for tuning. The method combines default, overridden, and
        custom arguments to configure the tuning process.

        Args:
            use_ray (bool): If True, uses Ray Tune for hyperparameter tuning. Defaults to False.
            iterations (int): The number of tuning iterations to perform. Defaults to 10.
            *args (list): Variable length argument list for additional arguments.
            **kwargs (any): Arbitrary keyword arguments. These are combined with the model's overrides and defaults.

        Returns:
            (dict): A dictionary containing the results of the hyperparameter search.

        Raises:
            AssertionError: If the model is not a PyTorch model.
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

    def _apply(self, fn) -> "Model":
        """Apply to(), cpu(), cuda(), half(), float() to model tensors that are not parameters or registered buffers."""
        self._check_is_pytorch_model()
        self = super()._apply(fn)  # noqa
        self.predictor = None  # reset predictor as device may have changed
        self.overrides["device"] = self.device  # was str(self.device) i.e. device(type='cuda', index=0) -> 'cuda:0'
        return self

    @property
    def names(self) -> list:
        """
        Retrieves the class names associated with the loaded model.

        This property returns the class names if they are defined in the model. It checks the class names for validity
        using the 'check_class_names' function from the ultralytics.nn.autobackend module.

        Returns:
            (list | None): The class names of the model if available, otherwise None.
        """
        from ultralytics.nn.autobackend import check_class_names

        if hasattr(self.model, "names"):
            return check_class_names(self.model.names)
        else:
            if not self.predictor:  # export formats will not have predictor defined until predict() is called
                self.predictor = self._smart_load("predictor")(overrides=self.overrides, _callbacks=self.callbacks)
                self.predictor.setup_model(model=self.model, verbose=False)
            return self.predictor.model.names

    @property
    def device(self) -> torch.device:
        """
        Retrieves the device on which the model's parameters are allocated.

        This property is used to determine whether the model's parameters are on CPU or GPU. It only applies to models
        that are instances of nn.Module.

        Returns:
            (torch.device | None): The device (CPU/GPU) of the model if it is a PyTorch model, otherwise None.
        """
        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

    @property
    def transforms(self):
        """
        Retrieves the transformations applied to the input data of the loaded model.

        This property returns the transformations if they are defined in the model.

        Returns:
            (object | None): The transform object of the model if available, otherwise None.
        """
        return self.model.transforms if hasattr(self.model, "transforms") else None

    def add_callback(self, event: str, func) -> None:
        """
        Adds a callback function for a specified event.

        This method allows the user to register a custom callback function that is triggered on a specific event during
        model training or inference.

        Args:
            event (str): The name of the event to attach the callback to.
            func (callable): The callback function to be registered.

        Raises:
            ValueError: If the event name is not recognized.
        """
        self.callbacks[event].append(func)

    def clear_callback(self, event: str) -> None:
        """
        Clears all callback functions registered for a specified event.

        This method removes all custom and default callback functions associated with the given event.

        Args:
            event (str): The name of the event for which to clear the callbacks.

        Raises:
            ValueError: If the event name is not recognized.
        """
        self.callbacks[event] = []

    def reset_callbacks(self) -> None:
        """
        Resets all callbacks to their default functions.

        This method reinstates the default callback functions for all events, removing any custom callbacks that were
        added previously.
        """
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]

    @staticmethod
    def _reset_ckpt_args(args: dict) -> dict:
        """Reset arguments when loading a PyTorch model."""
        include = {"imgsz", "data", "task", "single_cls"}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    # def __getattr__(self, attr):
    #    """Raises error if object has no requested attribute."""
    #    name = self.__class__.__name__
    #    raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    def _smart_load(self, key: str):
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
    def task_map(self) -> dict:
        """
        Map head to model, trainer, validator, and predictor classes.

        Returns:
            task_map (dict): The map of model task to mode classes.
        """
        raise NotImplementedError("Please provide task map for your model!")
