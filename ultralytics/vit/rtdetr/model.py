# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
# RT-DETR model interface
"""

from pathlib import Path

from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight, yaml_model_load
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.engine.exporter import Exporter
from ultralytics.yolo.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, LOGGER, ROOT, is_git_dir
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.torch_utils import model_info

from ...yolo.utils.torch_utils import smart_inference_mode
from .predict import RTDETRPredictor
from .val import RTDETRValidator


class RTDETR:

    def __init__(self, model='rtdetr-l.pt') -> None:
        if model and not model.endswith('.pt') and not model.endswith('.yaml'):
            raise NotImplementedError('RT-DETR only supports creating from pt file or yaml file.')
        # Load or create new YOLO model
        self.predictor = None
        suffix = Path(model).suffix
        if suffix == '.yaml':
            self._new(model)
        else:
            self._load(model)

    def _new(self, cfg: str, verbose=True):
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = 'detect'
        self.model = DetectionModel(cfg_dict, verbose=verbose)  # build model

        # Below added to allow export from yamls
        self.model.args = DEFAULT_CFG_DICT  # attach args to model
        self.model.task = self.task

    @smart_inference_mode()
    def _load(self, weights: str):
        self.model, _ = attempt_load_one_weight(weights)
        self.model.args = DEFAULT_CFG_DICT  # attach args to model
        self.task = self.model.args['task']

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
        overrides = dict(conf=0.25, task='detect', mode='predict')
        overrides.update(kwargs)  # prefer kwargs
        if not self.predictor:
            self.predictor = RTDETRPredictor(overrides=overrides)
            self.predictor.setup_model(model=self.model)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, overrides)
        return self.predictor(source, stream=stream)

    def train(self, **kwargs):
        """Function trains models but raises an error as RTDETR models do not support training."""
        raise NotImplementedError("RTDETR models don't support training")

    def val(self, **kwargs):
        """Run validation given dataset."""
        overrides = dict(task='detect', mode='val')
        overrides.update(kwargs)  # prefer kwargs
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.imgsz = check_imgsz(args.imgsz, max_dim=1)
        validator = RTDETRValidator(args=args)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def info(self, verbose=True):
        """Get model info"""
        return model_info(self.model, verbose=verbose)

    @smart_inference_mode()
    def export(self, **kwargs):
        """
        Export model.

        Args:
            **kwargs : Any other args accepted by the predictors. To see all args check 'configuration' section in docs
        """
        overrides = dict(task='detect')
        overrides.update(kwargs)
        overrides['mode'] = 'export'
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz:
            args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        if args.batch == DEFAULT_CFG.batch:
            args.batch = 1  # default to 1 if not modified
        return Exporter(overrides=args)(model=self.model)
