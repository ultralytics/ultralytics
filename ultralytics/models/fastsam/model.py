# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
FastSAM model interface.

Usage - Predict:
    from ultralytics import FastSAM

    model = FastSAM('last.pt')
    results = model.predict('ultralytics/assets/bus.jpg')
"""

from ultralytics.cfg import get_cfg
from ultralytics.engine.exporter import Exporter
from ultralytics.engine.model import YOLO
from ultralytics.utils import DEFAULT_CFG, LOGGER, ROOT, is_git_dir
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import model_info, smart_inference_mode

from .predict import FastSAMPredictor


class FastSAM(YOLO):

    def __init__(self, model='FastSAM-x.pt'):
        """Call the __init__ method of the parent class (YOLO) with the updated default model"""
        if model == 'FastSAM.pt':
            model = 'FastSAM-x.pt'
        super().__init__(model=model)
        # any additional initialization code for FastSAM

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
            (List[ultralytics.engine.results.Results]): The prediction results.
        """
        if source is None:
            source = ROOT / 'assets' if is_git_dir() else 'https://ultralytics.com/images/bus.jpg'
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")
        overrides = self.overrides.copy()
        overrides['conf'] = 0.25
        overrides.update(kwargs)  # prefer kwargs
        overrides['mode'] = kwargs.get('mode', 'predict')
        assert overrides['mode'] in ['track', 'predict']
        overrides['save'] = kwargs.get('save', False)  # do not save by default if called in Python
        self.predictor = FastSAMPredictor(overrides=overrides)
        self.predictor.setup_model(model=self.model, verbose=False)

        return self.predictor(source, stream=stream)

    def train(self, **kwargs):
        """Function trains models but raises an error as FastSAM models do not support training."""
        raise NotImplementedError("FastSAM models don't support training")

    def val(self, **kwargs):
        """Run validation given dataset."""
        overrides = dict(task='segment', mode='val')
        overrides.update(kwargs)  # prefer kwargs
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.imgsz = check_imgsz(args.imgsz, max_dim=1)
        validator = FastSAM(args=args)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

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

    def info(self, detailed=False, verbose=True):
        """
        Logs model info.

        Args:
            detailed (bool): Show detailed information about model.
            verbose (bool): Controls verbosity.
        """
        return model_info(self.model, detailed=detailed, verbose=verbose, imgsz=640)

    def __call__(self, source=None, stream=False, **kwargs):
        """Calls the 'predict' function with given arguments to perform object detection."""
        return self.predict(source, stream, **kwargs)

    def __getattr__(self, attr):
        """Raises error if object has no requested attribute."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")
