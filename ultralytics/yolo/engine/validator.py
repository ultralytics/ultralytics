import logging

import torch
from tqdm import tqdm

from ultralytics.yolo.utils.ops import Profile
from ultralytics.yolo.utils.torch_utils import select_device


class BaseValidator:
    """
    Base validator class.
    """

    def __init__(self, dataloader, device='', half=False, pbar=None, logger=None):
        self.dataloader = dataloader
        self.half = half
        self.device = select_device(device, dataloader.batch_size)
        self.pbar = pbar
        self.logger = logger or logging.getLogger()

    def __call__(self, trainer=None, model=None):
        """
        Supports validation of a pre-trained model if passed or a model being trained
        if trainer is passed (trainer gets priority).
        """
        training = trainer is not None
        # trainer = trainer or self.trainer_class.get_trainer()
        assert training or model is not None, "Either trainer or model is needed for validation"
        if training:
            model = trainer.model
            self.half &= self.device.type != 'cpu'
            model = model.half() if self.half else model
        else:  # TODO: handle this when detectMultiBackend is supported
            # model = DetectMultiBacked(model)
            pass

        model.eval()
        dt = Profile(), Profile(), Profile(), Profile()
        loss = 0
        n_batches = len(self.dataloader)
        desc = self.set_desc()
        bar = tqdm(self.dataloader, desc, n_batches, not training, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        self.init_metrics()
        with torch.cuda.amp.autocast(enabled=self.device.type != 'cpu'):
            for images, labels in bar:
                # pre-process
                with dt[0]:
                    images, labels = self.preprocess_batch(images, labels)

                # inference
                with dt[1]:
                    preds = model(images)
                    # TODO: remember to add native augmentation support when implementing model, like:
                    #  preds, train_out = model(im, augment=augment)

                # loss
                with dt[2]:
                    if training:
                        loss += trainer.criterion(preds, labels) / images.shape[0]

                # pre-process predictions
                with dt[3]:
                    preds = self.preprocess_preds(preds)

                self.update_metrics(preds, labels)

        stats = self.get_stats()
        self.check_stats(stats)

        self.print_results()

        # print speeds
        if not training:
            t = tuple(x.t / len(self.dataloader.dataset.samples) * 1E3 for x in dt)  # speeds per image
            # shape = (self.dataloader.batch_size, 3, imgsz, imgsz)
            self.logger.info(
                'Speed: %.1fms pre-process, %.1fms inference, %.1fms loss, %.1fms post-process per image at shape ' % t)

        # TODO: implement save json

        return stats

    def preprocess_batch(self, images, labels):
        return images.to(self.device, non_blocking=True), labels.to(self.device)

    def preprocess_preds(self, preds):
        return preds

    def init_metrics(self):
        pass

    def update_metrics(self, preds, targets):
        pass

    def get_stats(self):
        pass

    def check_stats(self, stats):
        pass

    def print_results(self):
        pass

    def set_desc(self):
        pass
