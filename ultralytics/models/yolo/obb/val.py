# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import OBBMetrics, batch_probiou


class OBBValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on an Oriented Bounding Box (OBB) model.

    Example:
        ```python
        from ultralytics.models.yolo.obb import OBBValidator

        args = dict(model='yolov8n-obb.pt', data='coco8-seg.yaml')
        validator = OBBValidator(args=args)
        validator(model=args['model'])
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize OBBValidator and set task to 'obb', metrics to OBBMetrics."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.plot_masks = None
        self.process = None
        self.args.task = 'obb'
        self.metrics = OBBMetrics(save_dir=self.save_dir, plot=True, on_plot=self.on_plot)

    def _process_batch(self, detections, labels):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        iou = batch_probiou(labels[:, 1:], detections[:, :4])
        return self.match_predictions(detections[:, 5], labels[:, 0], iou)

    # TODO
    def update_metrics(self, preds, batch):
        pass

    # TODO
    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        pass
