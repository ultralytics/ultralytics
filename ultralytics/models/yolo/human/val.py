# Ultralytics YOLO 🚀, AGPL-3.0 license
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.data.dataset import HumanDataset
from ultralytics.utils import colorstr


class HumanValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a human model.

    Example:
        ```python
        from ultralytics.models.yolo.human import HumanValidator

        args = dict(model='yolov8n-human.pt', data='coco8.yaml')
        validator = HumanValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)

    def build_dataset(self, img_path, mode="val", batch=None):
        cfg = self.args
        return HumanDataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == "train",  # augmentation
            hyp=cfg,
            rect=cfg.rect or mode == "val",  # rectangular batches
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=int(self.stride),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            task=cfg.task,
            classes=cfg.classes,
            data=self.data,
            fraction=cfg.fraction if mode == "train" else 1.0,
        )
