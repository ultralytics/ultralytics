# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo.detect import DetectionValidator


class YOLOv10DetectionValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a YOLOv10.

    Example:
        ```python
        from ultralytics.models.yolov10.detect import YOLOv10DetectionValidator

        args = dict(model='yolov10n.pt', data='coco8.yaml')
        validator = YOLOv10DetectionValidator(args=args)
        validator()
        ```
    """
    def postprocess(self, preds):
        """Apply postprocess for yolov10 models."""
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        return preds
