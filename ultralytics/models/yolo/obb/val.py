# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import OBBMetrics


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
        self.metrics = OBBMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
