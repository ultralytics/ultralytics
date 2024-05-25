# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo.detect import DetectionValidator


class YOLOv10DetectionValidator(DetectionValidator):
    def postprocess(self, preds):
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        return preds
