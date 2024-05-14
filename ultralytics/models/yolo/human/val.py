# Ultralytics YOLO 🚀, AGPL-3.0 license
from ultralytics.models.yolo.detect import DetectionValidator


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
