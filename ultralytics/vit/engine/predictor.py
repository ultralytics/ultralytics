# Draft of predictor engine for Vits.
from pathlib import Path

from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.checks import check_imgsz, check_imshow

class BasePredictor:
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        self.args = get_cfg(cfg, overrides)
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.plotted_img = None
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)
        
        self.setup()

    def setup(self, model=None):
        pass
    
    def __call__(self):
        pass