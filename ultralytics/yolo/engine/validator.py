import logging

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ultralytics.yolo.engine.trainer import DEFAULT_CONFIG
from ultralytics.yolo.utils import TQDM_BAR_FORMAT, LOGGER
from ultralytics.yolo.utils.ops import Profile
from ultralytics.yolo.utils.torch_utils import de_parallel, select_device, check_img_size
from ultralytics.yolo.utils.modeling.autobackend import AutoBackend
from ultralytics.yolo.utils.modeling import get_model
from ultralytics.yolo.data.utils import check_dataset, check_dataset_yaml


class BaseValidator:
    """
    Base validator class.
    """

    def __init__(self, dataloader=None, pbar=None, logger=None, args=None):
        self.dataloader = dataloader
        self.pbar = pbar
        self.logger = logger or LOGGER
        self.args = args or OmegaConf.load(DEFAULT_CONFIG)
        self.device = None
        self.model=None
        self.data = None
        self.cuda = None
        self.batch_i = None
        self.training = True
        self.loss = None

    def __call__(self, trainer=None, model=None):
        """
        Supports validation of a pre-trained model if passed or a model being trained
        if trainer is passed (trainer gets priority).
        """
        self.training = trainer is not None
        if self.training:
            self.device = trainer.device
            self.cuda = self.device.type != 'cpu'
            self.data = trainer.data
            model = trainer.ema.ema or trainer.model
            self.args.half &= self.device.type != 'cpu'
            # NOTE: half() inference in evaluation will make training stuck,
            # so I comment it out for now, I think we can reuse half mode after we add EMA.
            model = model.half() if self.args.half else model.float()
            self.model = model
        else:  # TODO: handle this when detectMultiBackend is supported
            assert model is not None, "Either trainer or model is needed for validation"
            self.device = select_device(self.args.device, self.args.batch_size)
            self.args.half &= self.device.type != 'cpu'
            model = AutoBackend(model, device=self.device, dnn=self.args.dnn, fp16=self.args.half)
            self.model = model
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_img_size(self.args.img_size, s=stride)
            half = model.fp16  # FP16 supported on limited backends with CUDA
            if engine:
                self.args.batch_size = model.batch_size
            else:
                self.device = model.device
                if not (pt or jit):
                    self.args.batch_size = 1  # export.py models default to batch-size 1
                    self.logger.info(f'Forcing --batch-size 1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')
            
            if self.args.data.endswith(".yaml"):
                data = check_dataset_yaml(self.args.data)
            else:
                data = check_dataset(self.args.data)
            self.dataloader = self.get_dataloader(data.get("val") or data.set("test"), self.args.batch_size)
            
        model.eval()
        dt = Profile(), Profile(), Profile(), Profile()
        self.loss = 0
        n_batches = len(self.dataloader)
        desc = self.get_desc()
        bar = tqdm(self.dataloader, desc, n_batches, not self.training, bar_format=TQDM_BAR_FORMAT)
        self.init_metrics(de_parallel(model))
        with torch.no_grad():
            for batch_i, batch in enumerate(bar):
                self.batch_i = batch_i
                # pre-process
                with dt[0]:
                    batch = self.preprocess(batch)

                # inference
                with dt[1]:
                    preds = model(batch["img"].float())
                    # TODO: remember to add native augmentation support when implementing model, like:
                    #  preds, train_out = model(im, augment=augment)

                # loss
                with dt[2]:
                    if self.training:
                        self.loss += trainer.criterion(preds, batch)[0]

                # pre-process predictions
                with dt[3]:
                    preds = self.postprocess(preds)

                self.update_metrics(preds, batch)

        stats = self.get_stats()
        self.check_stats(stats)

        self.print_results()

        # print speeds
        if not self.training:
            t = tuple(x.t / len(self.dataloader.dataset) * 1E3 for x in dt)  # speeds per image
            # shape = (self.dataloader.batch_size, 3, imgsz, imgsz)
            self.logger.info(
                'Speed: %.1fms pre-process, %.1fms inference, %.1fms loss, %.1fms post-process per image at shape ' % t)

        if self.training:
            model.float()
        # TODO: implement save json

        return stats

    def get_dataloader(self, dataset_path, batch_size):
        raise Exception("get_dataloder function not implemented for this validator")

    def preprocess(self, batch):
        return batch

    def postprocess(self, preds):
        return preds

    def init_metrics(self):
        pass

    def update_metrics(self, preds, batch):
        pass

    def get_stats(self):
        pass

    def check_stats(self, stats):
        pass

    def print_results(self):
        pass

    def get_desc(self):
        pass
