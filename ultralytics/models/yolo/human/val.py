# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.data.dataset import HumanDataset
from ultralytics.engine.results import Human, Results
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import colorstr
from ultralytics.utils.metrics import HumanMetrics
from pathlib import Path
import numpy as np


class HumanValidator(BaseValidator):
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
        self.args.task = "human"
        self.metrics = HumanMetrics()

    def build_dataset(self, img_path, mode="val", batch=None):
        return HumanDataset(
            img_path=img_path,
            augment=mode == "train",  # augmentation
            hyp=self.args,
            prefix=colorstr(f"{mode}: "),
        )

    def preprocess(self, batch):
        """Preprocesses batch by converting masks to float and sending to device."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["attributes"] = batch["attributes"].to(self.device).float()
        return batch

    def update_metrics(self, preds, batch):
        """
        Process Human Attributes and compute the accuracy.

        Args:
            predn (torch.Tensor): The predictions of attributes with shape [M, 11].
            batch (Dict): A dict contains grounding truth of attributes with shape [N, 5].

        Returns:
            The accuracy for each human attribute.
        """
        preds = Human(preds)
        gt_attrs = batch["attributes"]
        weight = gt_attrs[:, 0]
        height = gt_attrs[:, 1]
        gender = gt_attrs[:, 2]
        age = gt_attrs[:, 3]
        ethnicity = gt_attrs[:, 4]
        acc_w = 1 - (preds.weight - weight).abs() / weight
        acc_h = 1 - (preds.height - height).abs() / height
        acc_g = (preds.cls_gender == gender).float()
        acc_a = 1 - (preds.age - age).abs() / age
        acc_e = (preds.cls_ethnicity == ethnicity).float()

        self.metrics.attrs_stats["weight"].append(acc_w.clip(0, 1))
        self.metrics.attrs_stats["height"].append(acc_h.clip(0, 1))
        self.metrics.attrs_stats["gender"].append(acc_g)
        self.metrics.attrs_stats["age"].append(acc_a.clip(0, 1))
        self.metrics.attrs_stats["ethnicity"].append(acc_e)

        if self.args.save_txt:
            for i, pred in enumerate(preds):
                file = self.save_dir / "labels" / f'{Path(batch["im_file"][i]).stem}.txt'
                self.save_one_txt(pred, self.args.save_conf, batch["ori_shape"], file)

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        im = np.zeros((shape[0], shape[1]), dtype=np.uint8)
        result = Results(im, path=None, names=self.names, human=predn)
        result.save_txt(file, save_conf=save_conf)

    def get_desc(self):
        """Return a formatted description of evaluation metrics."""
        return ("%22s" + "%11s" * 11) % (
            "Class",
            "acc(W)",  # weight
            "acc(H)",  # height
            "acc(G)",  # gender
            "acc(A)",  # age
            "acc(E)",  # ethnicity
        )
