import logging

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ultralytics.yolo.engine.trainer import DEFAULT_CONFIG
from ultralytics.yolo.utils.ops import Profile
from ultralytics.yolo.utils.torch_utils import de_parallel, select_device


class BaseValidator:
    """
    Base validator class.
    """

    def __init__(self, dataloader, pbar=None, logger=None, args=None):
        self.dataloader = dataloader
        self.pbar = pbar
        self.logger = logger or logging.getLogger()
        self.args = args or OmegaConf.load(DEFAULT_CONFIG)
        self.device = select_device(self.args.device, dataloader.batch_size)
        self.cuda = self.device.type != 'cpu'
        self.batch_i = None
        self.training = True

    def __call__(self, trainer=None, model=None):
        """
        Supports validation of a pre-trained model if passed or a model being trained
        if trainer is passed (trainer gets priority).
        """
        training = trainer is not None
        self.training = training
        # trainer = trainer or self.trainer_class.get_trainer()
        assert training or model is not None, "Either trainer or model is needed for validation"
        if training:
            model = trainer.model
            self.args.half &= self.device.type != 'cpu'
            # NOTE: half() inference in evaluation will make training stuck,
            # so I comment it out for now, I think we can reuse half mode after we add EMA.
            # model = model.half() if self.args.half else model
        else:  # TODO: handle this when detectMultiBackend is supported
            # model = DetectMultiBacked(model)
            pass
            # TODO: implement init_model_attributes()

        model.eval()
        dt = Profile(), Profile(), Profile(), Profile()
        loss = 0
        n_batches = len(self.dataloader)
        desc = self.get_desc()
        bar = tqdm(self.dataloader, desc, n_batches, not training, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
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
                    if training:
                        loss += trainer.criterion(preds, batch)[0]

                # pre-process predictions
                with dt[3]:
                    preds = self.postprocess(preds)

                self.update_metrics(preds, batch)

        stats = self.get_stats()
        self.check_stats(stats)

        self.print_results()

        # print speeds
        if not training:
            t = tuple(x.t / len(self.dataloader.dataset.samples) * 1E3 for x in dt)  # speeds per image
            # shape = (self.dataloader.batch_size, 3, imgsz, imgsz)
            self.logger.info(
                'Speed: %.1fms pre-process, %.1fms inference, %.1fms loss, %.1fms post-process per image at shape ' % t)

        if self.training:
            model.float()
        # TODO: implement save json

        return stats

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
