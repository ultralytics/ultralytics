from ultralytics.nn.tasks import DFINEDetectModel
from ultralytics.utils import RANK
from ..rtdetr.train import RTDETRTrainer


class DEIMModelTrainer(RTDETRTrainer):
    """
    DEIMModelTrainer is a subclass of RTDETRTrainer for training DEIM models.

    Examples:
        >>> from ultralytics.models.rtdetr.train import DEIMModelTrainer
        >>> args = dict(model="rtdetr-l.yaml", data="coco8.yaml", imgsz=640, epochs=3)
        >>> trainer = DEIMModelTrainer(overrides=args)
        >>> trainer.train()
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Initialize and return an RT-DETR model for object detection tasks.

        Args:
            cfg (dict, optional): Model configuration.
            weights (str, optional): Path to pre-trained model weights.
            verbose (bool): Verbose logging if True.

        Returns:
            (RTDETRDetectionModel): Initialized model.
        """
        model = DFINEDetectModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
